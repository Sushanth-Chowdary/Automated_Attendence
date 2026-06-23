# 1. Imports
import torch
torch.cuda.set_per_process_memory_fraction(0.5, 0)
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import pickle
import pandas as pd
from datetime import datetime
import os
import shutil  
from tqdm import tqdm  
from collections import Counter
import faiss
import threading
import queue

# Tracker Import
from deep_sort_realtime.deepsort_tracker import DeepSort


torch.cuda.set_per_process_memory_fraction(0.5, 0)
# ==========================================
# THREADED VIDEO I/O HELPER
# ==========================================
class ThreadedVideoReader:
    def __init__(self, path, queue_size=128):
        # Initialize standard capture
        self.cap = cv2.VideoCapture(path)
        
        # Get video properties right away so we can use them later
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # The Queue is our "waiting line" for frames
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def start(self):
        # Start the background thread
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # This loop runs constantly in the background
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                return
            
            # .put() naturally blocks/pauses the thread if the queue is at maxsize (128).
            # This prevents the CPU from spinning infinitely and locking up the script!
            self.q.put(frame)

    def read(self):
        # The main script calls this to instantly grab the next ready frame
        return self.q.get()

    def more(self):
        # Check if there are still frames left
        return self.q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        self.cap.release()

# ==========================================
# ALIGNMENT HELPER FUNCTIONS
# ==========================================
def crop_standard(img, box):
    """Fallback standard crop with 15% margin."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img.shape[1], x2 + margin_x)
    y2 = min(img.shape[0], y2 + margin_y)
    
    return img[y1:y2, x1:x2]

def align_face(img, box, keypoints):
    """Rotates the face to align the eyes horizontally."""
    if keypoints is None or len(keypoints) < 2:
        return crop_standard(img, box)
        
    x1, y1, x2, y2 = map(int, box)
    left_eye, right_eye = keypoints[0], keypoints[1]
    
    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye
        
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    
    if dx == 0:
        return crop_standard(img, box)
        
    angle = np.degrees(np.arctan2(dy, dx))
    
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    margin_x, margin_y = int(w * 0.5), int(h * 0.5)
    X1 = max(0, cx - w - margin_x)
    Y1 = max(0, cy - h - margin_y)
    X2 = min(img.shape[1], cx + w + margin_x)
    Y2 = min(img.shape[0], cy + h + margin_y)
    
    large_crop = img[Y1:Y2, X1:X2]
    if large_crop.size == 0:
        return crop_standard(img, box)
        
    eye_center = ((left_eye[0] + right_eye[0]) / 2 - X1, (left_eye[1] + right_eye[1]) / 2 - Y1)
    
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated_crop = cv2.warpAffine(large_crop, M, (large_crop.shape[1], large_crop.shape[0]), flags=cv2.INTER_CUBIC)
    
    local_cx, local_cy = cx - X1, cy - Y1
    new_cx = M[0, 0] * local_cx + M[0, 1] * local_cy + M[0, 2]
    new_cy = M[1, 0] * local_cx + M[1, 1] * local_cy + M[1, 2]
    
    fin_margin_x, fin_margin_y = int(w * 0.15), int(h * 0.15)
    fx1 = int(new_cx - w/2 - fin_margin_x)
    fy1 = int(new_cy - h/2 - fin_margin_y)
    fx2 = int(new_cx + w/2 + fin_margin_x)
    fy2 = int(new_cy + h/2 + fin_margin_y)
    
    fx1, fy1 = max(0, fx1), max(0, fy1)
    fx2, fy2 = min(rotated_crop.shape[1], fx2), min(rotated_crop.shape[0], fy2)
    
    final_crop = rotated_crop[fy1:fy2, fx1:fx2]
    if final_crop.size == 0:
        return crop_standard(img, box)
        
    return final_crop

# 2. Setup Devices and Models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# --- UPDATED TO USE TENSORRT ENGINE ---
yolo_model = YOLO('yolov8n-face.engine', task='detect')

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device).half()

# 3. Load the FAISS Index and Mappings
faiss_index_path = './face_attendance_faiss.bin'
index = faiss.read_index(faiss_index_path)
index.nprobe = 20

with open('./face_attendance_meta.pkl', 'rb') as f:
    saved_data = pickle.load(f)

target_names = saved_data['target_names']
y_real = saved_data['y_real']

# 4. Define Parameters
CONFIDENCE_THRESHOLD = 0.84 
FRAME_SKIP = 3                
FRAMES_PER_VOTE = 5          
REQUIRED_VOTES = 20           

input_dir = 'VIDEOS'
output_dir = os.path.abspath('ATTENDENCE RESULTS/MINE')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

target_videos = [
    '2026-04-27_10.02.44.mkv', 
    '2026-02-25_11.23.07.mkv', 
    '2026-03-05_11.02.28.mkv', 
    '2026-03-09_10.03.16.mkv',
    '2026-04-07_09.18.02.mkv',
    'video2.mkv',
    '2026-02-25_11.21.17.mkv',
    '2026-03-09_10.04.35.mkv',
    '2026-02-25_11.03.43.mkv',
    '2026-02-18_11.02.03.mkv',
    'video1_uajX8qg0.mp4',
    '2026-02-25_11.00.04.mkv',
    '2026-02-25_11.15.41.mkv',
    '2026-03-02_09.55.37.mkv'
]

video_files = [v for v in target_videos if os.path.exists(os.path.join(input_dir, v))]

to_tensor = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# 5. Iterate through each video
for video_filename in video_files:
    print(f"\n{'='*50}")
    print(f"Starting processing for: {video_filename}")
    print(f"{'='*50}")
    
    tracker = DeepSort(max_age=10, n_init=3, embedder_gpu=True, half=True)

    network_input_path = os.path.join(input_dir, video_filename)
    file_name_no_ext, _ = os.path.splitext(video_filename)
    
    local_input_path = f"/tmp/input_{video_filename}"
    video_output_filename = f"{file_name_no_ext}_output.mp4"
    temp_local_path = f"/tmp/temp_{video_output_filename}" 
    
    final_network_path = os.path.join(output_dir, video_output_filename)
    attendance_csv_path = os.path.join(output_dir, f"{file_name_no_ext}_output.csv")
    debug_csv_path = os.path.join(output_dir, f"{file_name_no_ext}_DEBUG_Tracks.csv")

    print(f"[1/4] Sending video {video_filename} to /tmp/ for local processing...")
    try:
        shutil.copy(network_input_path, local_input_path)
    except Exception as e:
        print(f"Error copying input video locally: {e}")
        continue

    video_stream = ThreadedVideoReader(local_input_path, queue_size=128).start()
    
    frame_width = video_stream.frame_width
    frame_height = video_stream.frame_height
    fps = video_stream.fps
    total_frames = video_stream.total_frames

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_local_path, fourcc, fps, (frame_width, frame_height))

    track_visibility = {}  
    track_identities = {}  
    global_student_votes = {name: 0 for name in target_names}
    
    active_track_memory = {} 
    archived_tracks = {} 
    
    frame_count = 0
    
    print(f"[2/4] Processing frames...")
    try:
        with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
            while video_stream.more():
                frame = video_stream.read()
                
                if frame is None:
                    break 

                is_keyframe = (frame_count % FRAME_SKIP == 0)

                # --- 1. TRACKING PHASE (Run on EVERY frame) ---
                results = yolo_model(frame, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                raw_kpts = results[0].keypoints.xy.cpu().numpy() if hasattr(results[0], 'keypoints') and results[0].keypoints is not None else None
                    
                detections = []
                for box, prob in zip(boxes, confs):
                    if prob > 0.60: 
                        x1, y1, x2, y2 = map(int, box)
                        w, h = x2 - x1, y2 - y1
                        detections.append(([x1, y1, w, h], prob, 'face'))

                tracks = tracker.update_tracks(detections, frame=frame)

                # --- 2. LOGIC & VISIBILITY PHASE ---
                if is_keyframe:
                    batch_tensors = []
                    batch_track_ids = []

                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                            
                        t_id = track.track_id
                        
                        if t_id not in active_track_memory:
                            video_seconds = frame_count / fps if fps > 0 else 0
                            time_str = f"{int(video_seconds // 60)}m {int(video_seconds % 60)}s"
                            active_track_memory[t_id] = {'start_time': time_str, 'buffer': [], 'all_preds': []}

                        if track.time_since_update == 0:
                            track_visibility[t_id] = True
                            track_box = track.to_ltrb() # x1, y1, x2, y2
                            
                            x1, y1, x2, y2 = map(int, track_box)
                            if (x2 - x1) < 50 or (y2 - y1) < 50: 
                                continue

                            best_kpt = None
                            if raw_kpts is not None and len(boxes) > 0:
                                txc, tyc = (x1 + x2) / 2, (y1 + y2) / 2
                                min_dist = float('inf')
                                for i, rbox in enumerate(boxes):
                                    rxc, ryc = (rbox[0] + rbox[2]) / 2, (rbox[1] + rbox[3]) / 2
                                    dist = (txc - rxc)**2 + (tyc - ryc)**2
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_kpt = raw_kpts[i]
                                if min_dist > 2500: 
                                    best_kpt = None

                            try:
                                face_crop_bgr = align_face(frame, track_box, best_kpt)
                                
                                # --- NEW QUALITY GATE ---
                                sharpness = cv2.Laplacian(face_crop_bgr, cv2.CV_64F).var()
                                if sharpness < 50.0:
                                    continue  # Skip this blurry face
                                # ------------------------

                                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                                face_crop_pil = Image.fromarray(face_crop_rgb)
                                
                                face_tensor = to_tensor(face_crop_pil).unsqueeze(0).to(device)
                                face_tensor = (face_tensor - 0.5) * 2
                                
                                batch_tensors.append(face_tensor)
                                batch_track_ids.append(t_id)
                            except Exception:
                                pass 
                        else:
                            track_visibility[t_id] = False

                    if len(batch_tensors) > 0:
                        combined_tensors = torch.cat(batch_tensors, dim=0).half() 
                        
                        with torch.no_grad():
                            embeddings = resnet(combined_tensors).detach().cpu().numpy().astype('float32')

                        faiss.normalize_L2(embeddings)
                        distances, indices = index.search(embeddings, k=1)

                        for i in range(len(batch_track_ids)):
                            t_id = batch_track_ids[i]
                            
                            similarity_score = distances[i][0]
                            matched_faiss_row = indices[i][0]
                            predicted_label_index = y_real[matched_faiss_row]
                            
                            if similarity_score > CONFIDENCE_THRESHOLD:
                                predicted_name = target_names[predicted_label_index]
                            else:
                                predicted_name = "Unknown"

                            active_track_memory[t_id]['buffer'].append(predicted_name)
                            active_track_memory[t_id]['all_preds'].append(predicted_name)

                            if len(active_track_memory[t_id]['buffer']) >= FRAMES_PER_VOTE:
                                vote_counts = Counter(active_track_memory[t_id]['buffer'])
                                winner = vote_counts.most_common(1)[0][0]
                                
                                if winner != "Unknown":
                                    global_student_votes[winner] += 1
                                    
                                active_track_memory[t_id]['buffer'] = [] 
                                
                                if t_id not in track_identities or winner != "Unknown":
                                    track_identities[t_id] = winner

                # --- 3. DRAWING PHASE (Run Every Frame) ---
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                        
                    t_id = track.track_id
                    
                    if track_visibility.get(t_id, False) == True:
                        x1, y1, x2, y2 = map(int, track.to_ltrb())
                        display_name = track_identities.get(t_id, "Analyzing...")
                        color = (0, 255, 0) if display_name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID:{t_id} {display_name}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                out.write(frame)
                
                # --- GARBAGE COLLECTION ---
                alive_ids = set(t.track_id for t in tracks if t.is_confirmed())
                stale_ids = [t_id for t_id in active_track_memory.keys() if t_id not in alive_ids]
                
                for t_id in stale_ids:
                    archived_tracks[t_id] = active_track_memory[t_id]
                    del active_track_memory[t_id]
                    if t_id in track_visibility:
                        del track_visibility[t_id]
                    if t_id in track_identities:
                        del track_identities[t_id]
                        
                frame_count += 1
                pbar.update(1) 

    except Exception as e:
        print(f"\n[!] Error encountered during frame processing: {e}")
        
    finally:
        print(f"\n[3/4] Forcing release of resources...")
        
        if 'video_stream' in locals():
            video_stream.stop()
            
        if out is not None:
            out.release()
        
        final_memory = {**archived_tracks, **active_track_memory}

        print(f"      Generating Attendance and Debug CSVs...")
        attendance_records = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for student in target_names:
            votes = global_student_votes.get(student, 0)
            status = 'Present' if votes >= REQUIRED_VOTES else 'Absent'
            attendance_records.append({'Name': student, 'Time': current_time, 'Status': status, 'Detection Count': votes})

        pd.DataFrame(attendance_records).to_csv(attendance_csv_path, index=False)

        debug_records = []
        for t_id, data in final_memory.items():
            if len(data['all_preds']) == 0:
                continue
                
            overall_counts = Counter(data['all_preds'])
            true_identity = overall_counts.most_common(1)[0][0]
            
            debug_records.append({
                'Track ID': t_id,
                'First Seen': data['start_time'],
                'Total Frames': len(data['all_preds']),
                'Predicted Identity': true_identity,
                'Breakdown': dict(overall_counts)
            })
            
        pd.DataFrame(debug_records).to_csv(debug_csv_path, index=False)

        if os.path.exists(temp_local_path):
            print(f"      Copying processed video back to output folder: {final_network_path}")
            
            os.makedirs(os.path.dirname(final_network_path), exist_ok=True)
            
            try:
                shutil.copyfile(temp_local_path, final_network_path) 
            except Exception as e:
                print(f"      [!] Failed to copy output video back: {e}")
                
                rescue_path = os.path.join(os.path.expanduser('~'), f"RESCUED_{video_output_filename}")
                print(f"      [!] Rescuing file to local drive: {rescue_path}")
                try:
                    shutil.copyfile(temp_local_path, rescue_path)
                    print("      [+] File successfully rescued to your local home folder.")
                except Exception as rescue_e:
                    print(f"      [!] Failsafe failed: {rescue_e}")

        print(f"[4/4] Cleaning up temporary /tmp/ files...")
        try:
            if os.path.exists(local_input_path):
                os.remove(local_input_path)
                print(f"      Deleted {local_input_path}")
            if os.path.exists(temp_local_path):
                os.remove(temp_local_path)
                print(f"      Deleted {temp_local_path}")
        except Exception as e:
            print(f"      [!] Error deleting temp files: {e}")
            
        print(f"\n*** Process completely finished for {video_filename}! ***\n")

print("All videos processed successfully.")
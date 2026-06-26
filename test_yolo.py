# 1. Imports
import torch
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
import gc
import time 
import subprocess # <-- Required for native OS file transfers

# Tracker Import
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==========================================
# THREADED VIDEO I/O HELPER 
# ==========================================
class ThreadedVideoReader:
    def __init__(self, path, queue_size=128):
        self.cap = cv2.VideoCapture(path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            
            while not self.stopped:
                try:
                    self.q.put(frame, timeout=0.5)
                    break 
                except queue.Full:
                    continue
                    
        self.cap.release()

    def read(self):
        try:
            return self.q.get(timeout=2.0)
        except queue.Empty:
            return None

    def more(self):
        return self.q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break


# ==========================================
# CROP HELPER FUNCTION (Alignment Removed)
# ==========================================
def crop_standard(img, box):
    """Grabs the raw face with a 15% margin for context."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img.shape[1], x2 + margin_x)
    y2 = min(img.shape[0], y2 + margin_y)
    
    return img[y1:y2, x1:x2]


# 2. Setup Devices and Models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

yolo_model = YOLO('yolov8n-face.pt', task='detect')
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
CONFIDENCE_THRESHOLD = 0.80  # Lowered to accommodate unaligned faces
FRAME_SKIP = 2                
FRAMES_PER_VOTE = 8          
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

    video_output_filename = f"{file_name_no_ext}_output.mp4"

    # LOCAL temporary paths (Writing directly to your Ubuntu machine)
    temp_local_vid = f"./temp_{video_output_filename}" 
    temp_local_att_csv = f"./temp_{file_name_no_ext}_output.csv"
    temp_local_dbg_csv = f"./temp_{file_name_no_ext}_DEBUG_Tracks.csv"
    
    # FINAL network paths (Where they go when finished)
    final_network_vid = os.path.join(output_dir, video_output_filename)
    final_network_att_csv = os.path.join(output_dir, f"{file_name_no_ext}_output.csv")
    final_network_dbg_csv = os.path.join(output_dir, f"{file_name_no_ext}_DEBUG_Tracks.csv")

    print(f"[1/4] Connecting directly to {video_filename} on the network...")
    # Reading directly from the network to save time
    video_stream = ThreadedVideoReader(network_input_path, queue_size=128).start()
    
    frame_width = video_stream.frame_width
    frame_height = video_stream.frame_height
    fps = video_stream.fps
    total_frames = video_stream.total_frames

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_local_vid, fourcc, fps, (frame_width, frame_height))

    track_visibility = {}  
    track_identities = {}  
    global_student_votes = {name: 0 for name in target_names}
    
    active_track_memory = {} 
    archived_tracks = {} 
    
    frame_count = 0
    
    profiler = {
        'yolo': 0.0,
        'deepsort': 0.0,
        'cropping': 0.0, 
        'facenet': 0.0,
        'faiss': 0.0,
        'drawing': 0.0
    }
    
    print(f"[2/4] Processing frames...")
    try:
        with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
            while video_stream.more():
                frame = video_stream.read()
                
                if frame is None:
                    break 

                is_keyframe = (frame_count % FRAME_SKIP == 0)

                # --- 1. TRACKING PHASE ---
                t_start = time.time()
                results = yolo_model(frame, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                profiler['yolo'] += (time.time() - t_start)
                    
                detections = []
                for box, prob in zip(boxes, confs):
                    if prob > 0.70: # Raised to 0.70 to prevent hallucination lag
                        x1, y1, x2, y2 = map(int, box)
                        w, h = x2 - x1, y2 - y1
                        
                        if w >= 50 and h >= 50: 
                            detections.append(([x1, y1, w, h], prob, 'face'))

                t_start = time.time()
                tracks = tracker.update_tracks(detections, frame=frame)
                profiler['deepsort'] += (time.time() - t_start)

                # --- 2. LOGIC & VISIBILITY PHASE ---
                if is_keyframe:
                    batch_tensors = []
                    batch_track_ids = []

                    t_start = time.time()
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
                            track_box = track.to_ltrb()
                            
                            x1, y1, x2, y2 = map(int, track_box)
                            if (x2 - x1) < 50 or (y2 - y1) < 50: 
                                continue

                            # Grab raw face crop
                            face_crop_bgr = crop_standard(frame, track_box)
                            if face_crop_bgr.size == 0:
                                continue
                                
                            # Blur check - Loosened to prevent "Analyzing..." limbo
                            sharpness = cv2.Laplacian(face_crop_bgr, cv2.CV_64F).var()
                            if sharpness < 4.0:
                                continue  
                                
                            try:
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
                    profiler['cropping'] += (time.time() - t_start)

                    if len(batch_tensors) > 0:
                        t_start = time.time()
                        combined_tensors = torch.cat(batch_tensors, dim=0).half() 
                        with torch.no_grad():
                            embeddings = resnet(combined_tensors).detach().cpu().numpy().astype('float32')
                        profiler['facenet'] += (time.time() - t_start)

                        t_start = time.time()
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
                        profiler['faiss'] += (time.time() - t_start)

                # --- 3. DRAWING PHASE ---
                t_start = time.time()
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
                profiler['drawing'] += (time.time() - t_start)
                
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

                if frame_count % 5400 == 0:
                    tqdm.write(
                        f"\n[DEBUG Profiler - Last 5400 Frames] "
                        f"YOLO: {profiler['yolo']:.2f}s | "
                        f"DeepSORT: {profiler['deepsort']:.2f}s | "
                        f"Crop: {profiler['cropping']:.2f}s | "
                        f"FaceNet: {profiler['facenet']:.2f}s | "
                        f"FAISS: {profiler['faiss']:.2f}s"
                    )
                    for k in profiler:
                        profiler[k] = 0.0

    except Exception as e:
        print(f"\n[!] Error encountered during frame processing: {e}")
        
    finally:
        print(f"\n[3/4] Forcing release of resources...")
        
        if 'video_stream' in locals():
            video_stream.stop()
            
        if out is not None:
            out.release()
        
        final_memory = {**archived_tracks, **active_track_memory}

        # ==========================================
        # PHASE 4: SAVING AND TRANSFERRING
        # ==========================================
        print(f"      Generating Attendance and Debug CSVs locally...")
        
        attendance_records = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for student in target_names:
            votes = global_student_votes.get(student, 0)
            status = 'Present' if votes >= REQUIRED_VOTES else 'Absent'
            attendance_records.append({'Name': student, 'Time': current_time, 'Status': status, 'Detection Count': votes})

        debug_records = []
        for t_id, data in final_memory.items():
            if len(data['all_preds']) == 0:
                debug_records.append({
                    'Track ID': t_id,
                    'First Seen': data['start_time'],
                    'Total Frames': 0,
                    'Predicted Identity': "Rejected (Blurry)",
                    'Breakdown': {}
                })
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
            
        # WRITE EVERYTHING LOCALLY FIRST
        pd.DataFrame(attendance_records).to_csv(temp_local_att_csv, index=False)
        pd.DataFrame(debug_records).to_csv(temp_local_dbg_csv, index=False)

        # MOVE EVERYTHING TO THE NETWORK USING NATIVE OS COMMANDS
        print(f"      Moving all processed files to network folder...")
        os.makedirs(os.path.dirname(final_network_vid), exist_ok=True)
        
        files_to_move = [
            (temp_local_vid, final_network_vid),
            (temp_local_att_csv, final_network_att_csv),
            (temp_local_dbg_csv, final_network_dbg_csv)
        ]
        
        for local_src, net_dest in files_to_move:
            if os.path.exists(local_src):
                try:
                    subprocess.run(['mv', local_src, net_dest], check=True)
                except subprocess.CalledProcessError:
                    print(f"      [!] Native move failed for {local_src}. Rescuing to local home drive...")
                    rescue_path = os.path.join(os.path.expanduser('~'), f"RESCUED_{os.path.basename(local_src)}")
                    try:
                        subprocess.run(['mv', local_src, rescue_path], check=True)
                    except Exception:
                        pass

        print(f"[4/4] Cleaning up local temporary files and resetting memory...")
        try:
            for local_src, _ in files_to_move:
                if os.path.exists(local_src):
                    os.remove(local_src)
        except Exception:
            pass
            
        if 'tracker' in locals():
            del tracker
        if 'video_stream' in locals():
            del video_stream
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"\n*** Process completely finished for {video_filename}! ***\n")

print("All videos processed successfully.")
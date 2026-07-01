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
from tqdm import tqdm  
from collections import Counter
import faiss
import subprocess 
import threading
import queue

# ==========================================
# THREADED VIDEO I/O HELPER 
# ==========================================
class ThreadedVideoReader:
    def __init__(self, path, queue_size=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = queue.Queue(maxsize=queue_size)
        
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 30.0
            
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                break
            if not self.Q.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stop()
                    break
                self.Q.put(frame)
            else:
                import time
                time.sleep(0.001)
        self.stream.release()

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True

# ==========================================
# CROP HELPER FUNCTION (Alignment Removed)
# ==========================================
def crop_standard(img, box):
    """Grabs the raw face with a 15% margin for context, matching train_yolo.py"""
    h, w, _ = img.shape
    x1, y1, x2, y2 = map(int, box)
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    margin_w = int(box_w * 0.15)
    margin_h = int(box_h * 0.15)
    
    nx1 = max(0, x1 - margin_w)
    ny1 = max(0, y1 - margin_h)
    nx2 = min(w, x2 + margin_w)
    ny2 = min(h, y2 + margin_h)
    
    return img[ny1:ny2, nx1:nx2]

def format_timestamp(frame_idx, fps):
    total_seconds = int(frame_idx / fps)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# ==========================================
# UPDATED ATTENDANCE LOGIC & POST-PROCESSING
# ==========================================
def save_attendance_results(video_filename, archived_tracks, active_track_memory, target_names, output_dir):
    final_mem = {**archived_tracks, **active_track_memory}
    debug_data = []
    
    student_presence = {name: False for name in target_names}
    student_detection_count = {name: 0 for name in target_names}

    for t_id, data in final_mem.items():
        total_frames = data.get('frames_alive', 0)
        all_preds = data['all_preds']
        
        # Filter out "Unknown" for the STRICT BACKEND logic 
        valid_preds = [p for p in all_preds if p != "Unknown"]
        valid_votes_count = len(valid_preds)
        
        if valid_votes_count > 0:
            counts = Counter(valid_preds)
            winner = counts.most_common(1)[0][0]
            
            # 1. Quality Consensus (Out of clean name frames only) 
            win_ratio = counts.get(winner, 0) / valid_votes_count
            
            # 2. Total Exposure Lifecycle Ratio (Out of ALL sampled frames, including Unknowns) 
            total_samples = len(all_preds)
            sample_ratio = counts.get(winner, 0) / total_samples if total_samples > 0 else 0
            
            # 3. The Ultimate 4-Condition Strict Gate 
            status = "Passed" if (total_frames >= 45 and 
                                  valid_votes_count >= 5 and 
                                  sample_ratio >= 0.25 and 
                                  win_ratio >= 0.60) else "Failed"
        else:
            winner = "Unknown"
            status = "Failed"
            counts = Counter(all_preds) 

        if status == "Passed" and winner != "Unknown":
            student_presence[winner] = True
            student_detection_count[winner] += counts.get(winner, 0)

        debug_data.append({
            'Track ID': t_id, 
            'Start Time': data.get('start_time', ''),
            'Total Frames': total_frames, 
            'Valid Votes': valid_votes_count,
            'Total Preds (inc. Unknown)': len(all_preds),
            'Predicted Identity': winner, 
            'Gate Status': status,
            'Breakdown': dict(Counter(all_preds))
        })

    stem = os.path.splitext(video_filename)[0]
    pd.DataFrame(debug_data).to_csv(os.path.join(output_dir, f"{stem}_DEBUG_Tracks.csv"), index=False)
    
    output_data = [
        {
            'Name': s, 
            'Status': 'Present' if student_presence[s] else 'Absent',
            'Detection Count': student_detection_count[s]
        }
        for s in target_names
    ]
    pd.DataFrame(output_data).to_csv(os.path.join(output_dir, f"{stem}_output.csv"), index=False)
    
    print(f"  -> Saved attendance + debug CSVs for {video_filename}")

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
CONFIDENCE_THRESHOLD = 0.76     [cite: 21]
FRAME_SKIP = 3                 [cite: 21]
FRAMES_PER_VOTE = 5            [cite: 21]

input_dir = './test_videos'
output_dir = './output_results'
video_staging_dir = './output_results/staging'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(video_staging_dir, exist_ok=True)

target_videos = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

if not target_videos:
    print(f"No videos found in '{input_dir}'. Please add test videos.")
    exit()

print(f"Found {len(target_videos)} video(s) for batch processing.")

# ==========================================
# RESOURCE INITIALIZATION
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load FAISS Knowledge Base
faiss_index_path = './face_attendance_faiss.bin'
metadata_path = './face_attendance_meta.pkl'

if not (os.path.exists(faiss_index_path) and os.path.exists(metadata_path)):
    print("Error: Missing FAISS components. Please run train_yolo.py first.")
    exit()

index = faiss.read_index(faiss_index_path)

with open(metadata_path, 'rb') as f:
    meta = pickle.load(f)
target_names = meta['target_names']
y_real = meta['y_real']

# Load Embedder Network (FP32 Mode)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
to_tensor = transforms.ToTensor()

# ==========================================
# BATCH EXECUTION LOOP
# ==========================================
interrupted = False

for video_filename in target_videos:
    if not os.path.exists(os.path.join(input_dir, video_filename)): 
        continue

    print(f"\nProcessing: {video_filename}")
    
    # Reset tracking environments per video
    yolo_model = YOLO('yolov8n-face.pt', task='detect')
    
    video_stream = ThreadedVideoReader(os.path.join(input_dir, video_filename)).start()

    video_stem = os.path.splitext(video_filename)[0]
    staging_video_path = os.path.join(video_staging_dir, f"{video_stem}_output.mp4")
    out = cv2.VideoWriter(staging_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_stream.fps, (video_stream.frame_width, video_stream.frame_height))

    active_track_memory, archived_tracks, track_identities = {}, {}, {}
    frame_count = 0

    try:
        with tqdm(total=video_stream.total_frames, unit="frame") as pbar:
            while video_stream.more():
                frame = video_stream.read()
                if frame is None: 
                    break 

                # --- BYTE TRACK ---
                results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

                has_detections = results[0].boxes.id is not None
                if has_detections:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.int().cpu().numpy()

                    for t_id in ids:
                        if t_id not in active_track_memory:
                            active_track_memory[t_id] = {
                                'start_time': format_timestamp(frame_count, video_stream.fps),
                                'frames_alive': 0, 
                                'buffer': [], 
                                'all_preds': []
                            }
                        active_track_memory[t_id]['frames_alive'] += 1

                # --- RECOGNITION SAMPLING ENGINE ---
                if frame_count % FRAME_SKIP == 0 and has_detections:
                    batch_tensors, batch_track_ids = [], []

                    for i, t_id in enumerate(ids):
                        crop = crop_standard(frame, boxes[i])
                        # Dynamic variance blur check
                        if crop.size > 0 and cv2.Laplacian(crop, cv2.CV_64F).var() > 4.0:
                            tensor = (to_tensor(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device) - 0.5) * 2
                            batch_tensors.append(tensor)
                            batch_track_ids.append(t_id)

                    if batch_tensors:
                        with torch.no_grad():
                            embeddings = resnet(torch.cat(batch_tensors, dim=0)).cpu().numpy().astype('float32')
                        faiss.normalize_L2(embeddings)
                        sims, indices = index.search(embeddings, k=1)

                        for i, t_id in enumerate(batch_track_ids):
                            name = target_names[y_real[indices[i][0]]] if sims[i][0] > CONFIDENCE_THRESHOLD else "Unknown"
                            active_track_memory[t_id]['buffer'].append(name)
                            active_track_memory[t_id]['all_preds'].append(name)

                            # Smooth Display UI update
                            if len(active_track_memory[t_id]['buffer']) >= FRAMES_PER_VOTE:
                                valid_history = [v for v in active_track_memory[t_id]['all_preds'] if v != "Unknown"]
                                winner = Counter(valid_history).most_common(1)[0][0] if valid_history else "Unknown"
                                track_identities[t_id] = winner
                                active_track_memory[t_id]['buffer'] = []

                # Render Canvas Bounding Elements
                if has_detections:
                    for i in range(len(ids)):
                        t_id = ids[i]
                        box = boxes[i]
                        name = track_identities.get(t_id, "Analyzing...")
                        color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                        cv2.putText(frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                out.write(frame)

                # Garbage Collection / Memory Management
                alive_ids = set(ids) if has_detections else set()
                for t_id in list(active_track_memory.keys()):
                    if t_id not in alive_ids:
                        archived_tracks[t_id] = active_track_memory.pop(t_id)

                frame_count += 1
                pbar.update(1)

    except KeyboardInterrupt:
        print(f"\n[Interrupted] Stopping '{video_filename}' early. Saving partial session data data...")
        interrupted = True
    finally:
        video_stream.stop()
        out.release()

        final_video_path = os.path.join(output_dir, f"{video_stem}_output.mp4")
        try:
            subprocess.run(['mv', staging_video_path, final_video_path], check=True)
            print(f"  -> Moved video to {final_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"  -> WARNING: Could not move session output video to final directory ({e}). Staging path: {staging_video_path}")

        save_attendance_results(video_filename, archived_tracks, active_track_memory, target_names, output_dir)

    if interrupted:
        break

if interrupted:
    print("\nBatch operation killed early. Remaining workspace items skipped.")
else:
    print("\nAll target videos processed successfully.")
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
        self.cap = cv2.VideoCapture(path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
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
        try: return self.q.get(timeout=2.0)
        except queue.Empty: return None

    def more(self): return self.q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        while not self.q.empty():
            try: self.q.get_nowait()
            except queue.Empty: break

def crop_standard(img, box):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    x1 = max(0, x1 - margin_x); y1 = max(0, y1 - margin_y)
    x2 = min(img.shape[1], x2 + margin_x); y2 = min(img.shape[0], y2 + margin_y)
    return img[y1:y2, x1:x2]

def format_timestamp(frame_count, fps):
    total_seconds = frame_count // fps
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# 2. Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# CHANGED: Removed .half() to keep FaceNet in FP32
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

faiss_index_path = './face_attendance_faiss.bin'
index = faiss.read_index(faiss_index_path)
index.nprobe = 20

with open('./face_attendance_meta.pkl', 'rb') as f:
    saved_data = pickle.load(f)
target_names, y_real = saved_data['target_names'], saved_data['y_real']

# 3. Parameters
CONFIDENCE_THRESHOLD = 0.76     
FRAME_SKIP = 3                 
FRAMES_PER_VOTE = 5          

input_dir = 'VIDEOS'
output_dir = os.path.abspath('ATTENDENCE RESULTS/MINE')
os.makedirs(output_dir, exist_ok=True)
video_staging_dir = os.path.abspath('.')

to_tensor = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])

# 4. Processing Loop
target_videos = ['2026-04-27_10.02.44.mkv', '2026-02-25_11.23.07.mkv', '2026-03-05_11.02.28.mkv', '2026-03-09_10.03.16.mkv', '2026-04-07_09.18.02.mkv', 'video2.mkv', '2026-02-25_11.21.17.mkv', '2026-03-09_10.04.35.mkv', '2026-02-25_11.03.43.mkv', '2026-02-18_11.02.03.mkv', 'video1_uajX8qg0.mp4', '2026-02-25_11.00.04.mkv', '2026-02-25_11.15.41.mkv', '2026-03-02_09.55.37.mkv']

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
            
            win_ratio = counts.get(winner, 0) / valid_votes_count
            # Strict Gate: 45 frames, 3 valid votes minimum, 60% consensus
            status = "Passed" if (total_frames >= 45 and valid_votes_count >= 5 and win_ratio >= 0.6) else "Failed"
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
    pd.DataFrame(debug_data).to_csv(os.path.join(output_dir, f"{stem}_DEBUG_Tracks.csv"))
    
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

interrupted = False

for video_filename in target_videos:
    if not os.path.exists(os.path.join(input_dir, video_filename)): continue

    print(f"\nProcessing: {video_filename}")
    
    # Initialize YOLO inside the loop to reset track IDs for every video
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
                if frame is None: break 

                # --- BYTE TRACK ---
                results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

                has_detections = results[0].boxes.id is not None
                if has_detections:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.int().cpu().numpy()

                if has_detections:
                    for t_id in ids:
                        if t_id not in active_track_memory:
                            active_track_memory[t_id] = {'start_time': format_timestamp(frame_count, video_stream.fps),
                                                          'frames_alive': 0, 'buffer': [], 'all_preds': []}
                        active_track_memory[t_id]['frames_alive'] += 1

                if frame_count % FRAME_SKIP == 0 and has_detections:
                    batch_tensors, batch_track_ids = [], []

                    for i, t_id in enumerate(ids):
                        crop = crop_standard(frame, boxes[i])
                        if crop.size > 0 and cv2.Laplacian(crop, cv2.CV_64F).var() > 4.0:
                            batch_tensors.append((to_tensor(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device) - 0.5) * 2)
                            batch_track_ids.append(t_id)

                    if batch_tensors:
                        with torch.no_grad():
                            # CHANGED: Removed .half() before passing to model to process in FP32
                            embeddings = resnet(torch.cat(batch_tensors, dim=0)).cpu().numpy().astype('float32')
                        faiss.normalize_L2(embeddings)
                        sims, indices = index.search(embeddings, k=1)

                        for i, t_id in enumerate(batch_track_ids):
                            name = target_names[y_real[indices[i][0]]] if sims[i][0] > CONFIDENCE_THRESHOLD else "Unknown"
                            active_track_memory[t_id]['buffer'].append(name)
                            active_track_memory[t_id]['all_preds'].append(name)

                            if len(active_track_memory[t_id]['buffer']) >= FRAMES_PER_VOTE:
                                # LOOSE UI: Look at ALL valid predictions for this track so far
                                valid_history = [v for v in active_track_memory[t_id]['all_preds'] if v != "Unknown"]
                                
                                if valid_history:
                                    winner = Counter(valid_history).most_common(1)[0][0]
                                else:
                                    winner = "Unknown"
                                    
                                track_identities[t_id] = winner
                                active_track_memory[t_id]['buffer'] = []

                # Drawing
                if has_detections:
                    for i in range(len(ids)):
                        t_id = ids[i]
                        box = boxes[i]
                        name = track_identities.get(t_id, "Analyzing...")
                        color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                        cv2.putText(frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                out.write(frame)

                # Memory Cleanup
                alive_ids = set(ids) if has_detections else set()
                for t_id in list(active_track_memory.keys()):
                    if t_id not in alive_ids:
                        archived_tracks[t_id] = active_track_memory.pop(t_id)

                frame_count += 1
                pbar.update(1)

    except KeyboardInterrupt:
        print(f"\n[Interrupted] Stopping '{video_filename}' early. "
              f"Saving whatever attendance data was collected before exiting.")
        interrupted = True
    finally:
        video_stream.stop()
        out.release()

        final_video_path = os.path.join(output_dir, f"{video_stem}_output.mp4")
        try:
            subprocess.run(['mv', staging_video_path, final_video_path], check=True)
            print(f"  -> Moved video to {final_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"  -> WARNING: video finished but couldn't be moved to {output_dir} ({e}). "
                  f"It's still saved locally at {staging_video_path} — move it manually.")

        save_attendance_results(video_filename, archived_tracks, active_track_memory, target_names, output_dir)

    if interrupted:
        break

if interrupted:
    print("\nBatch stopped early by user. Remaining videos in the list were not processed.")
else:
    print("\nAll videos processed.")
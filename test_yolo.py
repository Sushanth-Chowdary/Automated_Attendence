# 1. Imports
import torch
import numpy as np
import cv2
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
import torchvision.transforms as transforms
import torchvision.ops as ops
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

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

def format_timestamp(frame_count, fps):
    total_seconds = frame_count // fps
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# 2. Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_half = torch.cuda.is_available() 

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
if use_half:
    resnet = resnet.half()

faiss_index_path = './face_attendance_faiss.bin'
ref_embeddings_tensor = None
if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)
    index.nprobe = 20
    try:
        try:
            ref_embeddings = index.reconstruct_n(0, index.ntotal)
        except AttributeError:
            index.make_direct_map()
            ref_embeddings = np.array([index.reconstruct(i) for i in range(index.ntotal)])
        
        ref_embeddings_tensor = torch.from_numpy(ref_embeddings).to(device).float()
        ref_embeddings_tensor = torch.nn.functional.normalize(ref_embeddings_tensor, p=2, dim=1)
        if use_half:
            ref_embeddings_tensor = ref_embeddings_tensor.half()
    except Exception as e:
        print(f"Failed to extract FAISS vectors to GPU: {e}")
        
with open('./face_attendance_meta.pkl', 'rb') as f:
    saved_data = pickle.load(f)
target_names, y_real = saved_data['target_names'], saved_data['y_real']

# 3. Parameters
CONFIDENCE_THRESHOLD = 0.79     
FRAME_SKIP = 1                  
FRAMES_PER_VOTE = 5          

input_dir = 'VIDEOS'
output_dir = os.path.abspath('ATTENDENCE RESULTS/MINE')
os.makedirs(output_dir, exist_ok=True)
video_staging_dir = os.path.abspath('.')

target_videos = ['2026-04-27_10.02.44.mkv', '2026-02-25_11.23.07.mkv', '2026-03-05_11.02.28.mkv', '2026-03-09_10.03.16.mkv', '2026-04-07_09.18.02.mkv', 'video2.mkv', '2026-02-25_11.21.17.mkv', '2026-03-09_10.04.35.mkv', '2026-02-25_11.03.43.mkv', '2026-02-18_11.02.03.mkv', 'video1_uajX8qg0.mp4', '2026-02-25_11.00.04.mkv', '2026-02-25_11.15.41.mkv', '2026-03-02_09.55.37.mkv']

def save_attendance_results(video_filename, archived_tracks, active_track_memory, target_names, output_dir):
    final_mem = {**archived_tracks, **active_track_memory}
    debug_data = []
    student_presence = {name: False for name in target_names}
    student_detection_count = {name: 0 for name in target_names}

    for t_id, data in final_mem.items():
        total_frames = data.get('frames_alive', 0)
        all_preds = data['all_preds']
        valid_preds = [p for p in all_preds if p != "Unknown"]
        valid_votes_count = len(valid_preds)
        
        if valid_votes_count > 0:
            counts = Counter(valid_preds)
            winner = counts.most_common(1)[0][0]
            win_ratio = counts.get(winner, 0) / valid_votes_count
            total_samples = len(all_preds)
            sample_ratio = counts.get(winner, 0) / total_samples if total_samples > 0 else 0
            
            status = "Passed" if (total_frames >= 45 and 
                                  total_samples >= 15 and 
                                  sample_ratio >= 0.33 and 
                                  win_ratio >= 0.52) else "Failed"
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
    
    output_data = [{'Name': s, 'Status': 'Present' if student_presence[s] else 'Absent', 'Detection Count': student_detection_count[s]} for s in target_names]
    pd.DataFrame(output_data).to_csv(os.path.join(output_dir, f"{stem}_output.csv"), index=False)
    print(f"  -> Saved attendance + debug CSVs for {video_filename}")

interrupted = False
for video_filename in target_videos:
    if not os.path.exists(os.path.join(input_dir, video_filename)): continue
    print(f"\nProcessing: {video_filename}")
    yolo_model = YOLO('yolov8n-face.pt', task='detect')
    yolo_model.to(device)
    
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
                
                # Move frame to GPU inherently 
                frame_tensor = torch.from_numpy(frame).to(device, non_blocking=True).float()
                
                results = yolo_model.track(frame, persist=True, tracker="custom_bytetrack.yaml", verbose=False, quantize=16 if use_half else None, imgsz=640)
                has_detections = results[0].boxes.id is not None
                
                if has_detections and ref_embeddings_tensor is not None:
                    boxes = results[0].boxes.xyxy.to(device) 
                    ids = results[0].boxes.id.to(device).int() 
                    
                    ids_list = ids.cpu().tolist()
                    for t_id in ids_list:
                        if t_id not in active_track_memory:
                            active_track_memory[t_id] = {
                                'start_time': format_timestamp(frame_count, video_stream.fps),
                                'frames_alive': 0, 'buffer': [], 'all_preds': [], 'missing_frames': 0,
                                'crop_buffer': []
                            }
                        active_track_memory[t_id]['frames_alive'] += 1

                    if frame_count % FRAME_SKIP == 0:
                        batch_tensors, batch_track_ids = [], []
                        
                        box_w = boxes[:, 2] - boxes[:, 0]
                        box_h = boxes[:, 3] - boxes[:, 1]
                        aspect_ratios = box_w / (box_h + 1e-6)
                        
                        valid_mask = (box_w >= 65) & (box_h >= 65) & (aspect_ratios >= 0.55) & (aspect_ratios <= 1.55)
                        
                        if valid_mask.any():
                            valid_boxes = boxes[valid_mask].clone()
                            valid_ids = ids[valid_mask]
                            
                            margin_x = (valid_boxes[:, 2] - valid_boxes[:, 0]) * 0.15
                            margin_y = (valid_boxes[:, 3] - valid_boxes[:, 1]) * 0.15
                            
                            valid_boxes[:, 0] = torch.clamp(valid_boxes[:, 0] - margin_x, min=0)
                            valid_boxes[:, 1] = torch.clamp(valid_boxes[:, 1] - margin_y, min=0)
                            valid_boxes[:, 2] = torch.clamp(valid_boxes[:, 2] + margin_x, max=float(frame.shape[1]))
                            valid_boxes[:, 3] = torch.clamp(valid_boxes[:, 3] + margin_y, max=float(frame.shape[0]))
                            
                            batch_idx = torch.zeros((valid_boxes.size(0), 1), device=device, dtype=valid_boxes.dtype)
                            roi_boxes = torch.cat((batch_idx, valid_boxes), dim=1)
                            
                            frame_tensor_chw = frame_tensor.permute(2, 0, 1).unsqueeze(0)
                            crops = ops.roi_align(frame_tensor_chw, roi_boxes, output_size=(160, 160)).detach()
                            
                            valid_ids_list = valid_ids.cpu().tolist()
                            for i, t_id in enumerate(valid_ids_list):
                                active_track_memory[t_id]['crop_buffer'].append(crops[i:i+1])
                                
                                if len(active_track_memory[t_id]['crop_buffer']) >= FRAMES_PER_VOTE:
                                    batch_tensors.extend(active_track_memory[t_id]['crop_buffer'])
                                    batch_track_ids.extend([t_id] * len(active_track_memory[t_id]['crop_buffer']))
                                    active_track_memory[t_id]['crop_buffer'] = []
                        
                        if batch_tensors:
                            with torch.inference_mode():
                                batch_tensor = torch.cat(batch_tensors, dim=0) 
                                batch_tensor = batch_tensor[:, [2, 1, 0], :, :] 
                                
                                gray = transforms.functional.rgb_to_grayscale(batch_tensor)
                                laplacian_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=device, dtype=batch_tensor.dtype)
                                laplacian_out = torch.nn.functional.conv2d(gray, laplacian_kernel, padding=1)
                                laplacian_var = torch.var(laplacian_out, dim=(1, 2, 3))
                                
                                mask = laplacian_var > 5.0
                                mask_list = mask.cpu().tolist()
                                valid_batch_tensor = batch_tensor[mask]
                                valid_batch_track_ids = [batch_track_ids[k] for k in range(len(batch_track_ids)) if mask_list[k]]
                                
                                if valid_batch_tensor.size(0) > 0:
                                    valid_batch_tensor = (valid_batch_tensor / 127.5) - 1.0
                                    if use_half:
                                        valid_batch_tensor = valid_batch_tensor.half()
                                    
                                    embeddings = resnet(valid_batch_tensor)
                                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                                    
                                    sim_matrix = torch.mm(embeddings, ref_embeddings_tensor.t())
                                    max_sims, max_indices = torch.max(sim_matrix, dim=1)
                                    
                                    sims_list = max_sims.cpu().tolist()
                                    indices_list = max_indices.cpu().tolist()
                                    
                                    for i, t_id in enumerate(valid_batch_track_ids):
                                        name = target_names[y_real[indices_list[i]]] if sims_list[i] > CONFIDENCE_THRESHOLD else "Unknown"
                                        active_track_memory[t_id]['buffer'].append(name)
                                        active_track_memory[t_id]['all_preds'].append(name)
                                        
                                        if len(active_track_memory[t_id]['buffer']) >= FRAMES_PER_VOTE:
                                            valid_history = [v for v in active_track_memory[t_id]['all_preds'] if v != "Unknown"]
                                            winner = Counter(valid_history).most_common(1)[0][0] if valid_history else "Unknown"
                                            track_identities[t_id] = winner
                                            active_track_memory[t_id]['buffer'] = []

                    boxes_cpu = boxes.cpu().numpy()
                    for i in range(len(ids_list)):
                        t_id = ids_list[i]
                        box = boxes_cpu[i]
                        name = track_identities.get(t_id, "Analyzing...")
                        color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                        cv2.putText(frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                out.write(frame)
                
                alive_ids = set(ids_list) if has_detections else set()
                for t_id in list(active_track_memory.keys()):
                    if t_id not in alive_ids:
                        active_track_memory[t_id]['missing_frames'] += 1
                        if active_track_memory[t_id]['missing_frames'] > 50:
                            archived_tracks[t_id] = active_track_memory.pop(t_id)
                    else:
                        active_track_memory[t_id]['missing_frames'] = 0

                frame_count += 1
                pbar.update(1)

    except KeyboardInterrupt:
        interrupted = True
    finally:
        video_stream.stop()
        out.release()
        
        final_video_path = os.path.join(output_dir, f"{video_stem}_output.mp4")
        try: subprocess.run(['mv', staging_video_path, final_video_path], check=True)
        except: pass
        save_attendance_results(video_filename, archived_tracks, active_track_memory, target_names, output_dir)
    if interrupted: break
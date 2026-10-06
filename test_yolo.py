# 1. Imports
import os
import gc
import time
import queue
import random
import pickle
import warnings
import threading
from datetime import datetime
from collections import Counter
import concurrent.futures
import multiprocessing as mp

import cv2
import faiss
import numpy as np
import pandas as pd

import torch
import torchvision.ops as ops
import torchvision.transforms as transforms
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn

# Suppress serialization warnings from third-party weights
warnings.filterwarnings("ignore", category=FutureWarning)


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
            try:
                self.q.put(frame, timeout=2.0)
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


def video_writer_worker(write_queue, output_path, fps, width, height):
    """Handles CPU drawing and video encoding in the background."""
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    while True:
        item = write_queue.get()
        if item is None:
            break
            
        frame, boxes_cpu, ids_list, names = item

        for i in range(len(ids_list)):
            box = boxes_cpu[i]
            t_id = ids_list[i]
            name = names[i]
            color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        out.write(frame)
    out.release()


def format_timestamp(frame_count, fps):
    total_seconds = frame_count // fps
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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


# 2. Worker Lifecycle & Models
yolo_model = None
resnet = None
ref_embeddings_tensor = None
target_names = []
y_real = []
device = None
use_half = False


def init_worker():
    """Initializes models once per worker process to prevent redundant reload overhead."""
    global yolo_model, resnet, ref_embeddings_tensor, target_names, y_real, device, use_half
    
    time.sleep(random.uniform(0.2, 2.5))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_half = torch.cuda.is_available()

    # Load YOLO
    yolo_model = YOLO('yolov8n-face.pt', task='detect')
    yolo_model.to(device)

    # Load InceptionResnetV1
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    if use_half:
        resnet = resnet.half()

    # Load FAISS Gallery
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
        except Exception:
            ref_embeddings_tensor = None

    # Load Metadata
    with open('./face_attendance_meta.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    target_names, y_real = saved_data['target_names'], saved_data['y_real']


def process_video(video_filename, input_dir, output_dir, progress_q):
    """Processes a single video and pushes live telemetry through progress_q."""
    global yolo_model, resnet, ref_embeddings_tensor, target_names, y_real, device, use_half

    video_path = os.path.join(input_dir, video_filename)
    video_stem = os.path.splitext(video_filename)[0]
    output_csv = os.path.join(output_dir, f"{video_stem}_output.csv")
    final_video_path = os.path.join(output_dir, f"{video_stem}_output.mp4")

    # Resume capability: Skip if already finished
    if os.path.exists(output_csv):
        progress_q.put({"type": "skip", "file": video_filename})
        return video_filename, "Skipped"

    CONFIDENCE_THRESHOLD = 0.79     
    FRAME_SKIP = 1                  
    FRAMES_PER_VOTE = 5

    video_stream = ThreadedVideoReader(video_path).start()
    total_frames = max(video_stream.total_frames, 1)
    
    progress_q.put({"type": "start", "file": video_filename, "total_frames": total_frames})

    write_queue = queue.Queue(maxsize=128)
    writer_thread = threading.Thread(
        target=video_writer_worker, 
        args=(write_queue, final_video_path, video_stream.fps, video_stream.frame_width, video_stream.frame_height)
    )
    writer_thread.daemon = True
    writer_thread.start()
    
    active_track_memory, archived_tracks, track_identities = {}, {}, {}
    frame_count = 0
    t_start = time.perf_counter()
    last_q_time = t_start

    try:
        while video_stream.more():
            frame = video_stream.read()
            if frame is None: 
                break 
            
            frame_tensor = torch.from_numpy(frame).to(device, non_blocking=True).float()
            
            results = yolo_model.track(
                frame, 
                persist=True, 
                tracker="custom_bytetrack.yaml", 
                verbose=False, 
                half=use_half, 
                imgsz=640
            )
            has_detections = results[0].boxes.id is not None
            
            boxes_cpu = []
            ids_list = []
            frame_names = []
            
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
                    frame_names.append(track_identities.get(t_id, "Analyzing..."))
                    
            write_queue.put((frame, boxes_cpu, ids_list, frame_names))
            
            alive_ids = set(ids_list) if has_detections else set()
            for t_id in list(active_track_memory.keys()):
                if t_id not in alive_ids:
                    active_track_memory[t_id]['missing_frames'] += 1
                    if active_track_memory[t_id]['missing_frames'] > 50:
                        archived_tracks[t_id] = active_track_memory.pop(t_id)
                else:
                    active_track_memory[t_id]['missing_frames'] = 0

            frame_count += 1
            
            # Send status update every 10 frames to keep queue overhead negligible
            if frame_count % 10 == 0:
                now = time.perf_counter()
                elapsed = now - t_start
                curr_fps = frame_count / elapsed if elapsed > 0 else 0.0
                progress_q.put({
                    "type": "progress",
                    "file": video_filename,
                    "frame": frame_count,
                    "total": total_frames,
                    "fps": curr_fps
                })

    except Exception as e:
        progress_q.put({"type": "finish", "file": video_filename})
        return video_filename, f"Error: {str(e)}"
    finally:
        video_stream.stop()
        write_queue.put(None)
        writer_thread.join()  
        
        save_attendance_results(video_filename, archived_tracks, active_track_memory, target_names, output_dir)
        progress_q.put({"type": "finish", "file": video_filename})

        del active_track_memory, archived_tracks, track_identities
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return video_filename, "Processed"


# 3. Terminal Live Dashboard
def run_live_dashboard(progress_q, total_videos, stop_event):
    """Renders a real-time console table displaying progress and FPS per video."""
    active_videos = {}
    completed = 0

    with Live(auto_refresh=False, refresh_per_second=4) as live:
        while not stop_event.is_set() or not progress_q.empty():
            # Flush queue messages
            while True:
                try:
                    msg = progress_q.get_nowait()
                    msg_type = msg.get("type")
                    filename = msg.get("file")

                    if msg_type == "start":
                        active_videos[filename] = {
                            "frame": 0,
                            "total": msg.get("total_frames", 1),
                            "fps": 0.0
                        }
                    elif msg_type == "progress":
                        if filename in active_videos:
                            active_videos[filename]["frame"] = msg["frame"]
                            active_videos[filename]["total"] = msg["total"]
                            active_videos[filename]["fps"] = msg["fps"]
                    elif msg_type in ("finish", "skip"):
                        active_videos.pop(filename, None)
                        completed += 1
                except queue.Empty:
                    break

            # Construct dynamic table
            table = Table(
                title=f"[bold green]Batch Processing Progress ({completed}/{total_videos} Videos Finished)[/bold green]",
                box=None
            )
            table.add_column("Running File", style="cyan", width=35, no_wrap=True)
            table.add_column("Frames", style="magenta", width=18, justify="right")
            table.add_column("Completion", style="yellow", width=12, justify="right")
            table.add_column("FPS", style="green", width=10, justify="right")

            if active_videos:
                for fname, stat in sorted(active_videos.items()):
                    pct = min(100.0, (stat["frame"] / stat["total"]) * 100) if stat["total"] > 0 else 0
                    table.add_row(
                        fname,
                        f"{stat['frame']}/{stat['total']}",
                        f"{pct:5.1f}%",
                        f"{stat['fps']:5.1f}"
                    )
            else:
                table.add_row("[italic dim]Waiting for worker threads...[/italic dim]", "-", "-", "-")

            live.update(table, refresh=True)
            time.sleep(0.2)


# 4. Main Entry Point
if __name__ == '__main__':
    mp.freeze_support()
    ctx = mp.get_context('spawn')

    input_dir = 'VIDEOS'
    output_dir = os.path.abspath('ATTENDENCE RESULTS/MINE')
    os.makedirs(output_dir, exist_ok=True)

    video_extensions = ('.mp4', '.mkv', '.avi', '.mov')
    target_videos = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(video_extensions) and not f.startswith('.')
    ])

    total_files = len(target_videos)
    print(f"Found {total_files} video files in '{input_dir}'.")
    
    # 6 parallel pipelines for RTX A4000
    MAX_WORKERS = 6 

    if target_videos:
        # Multiprocessing Manager Queue for telemetry updates
        manager = ctx.Manager()
        progress_q = manager.Queue()
        stop_event = threading.Event()

        # Start background dashboard rendering thread
        dashboard_thread = threading.Thread(
            target=run_live_dashboard,
            args=(progress_q, total_files, stop_event),
            daemon=True
        )
        dashboard_thread.start()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS, 
            mp_context=ctx, 
            initializer=init_worker
        ) as executor:
            futures = {
                executor.submit(process_video, video, input_dir, output_dir, progress_q): video 
                for video in target_videos
            }
            
            for future in concurrent.futures.as_completed(futures):
                video_filename = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"\n  -> [CRASH] {video_filename} error: {exc}")

        stop_event.set()
        dashboard_thread.join(timeout=1.0)
        print("\nAll videos processed successfully.")
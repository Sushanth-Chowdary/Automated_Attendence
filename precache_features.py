# precache_features.py – Standalone Stage 1 YOLO+AdaFace Feature Pre-Cacher
# ==============================================================================
# This script strictly checks the VIDEOS folder against the feature_cache folder.
# It extracts bounding boxes, 5-point landmarks, AdaFace embeddings, and norms 
# into .pt files for any missing videos so sweep_yolo_sota and test_yolo_sota 
# can instantly skip Stage 1.
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import sys
import time
import queue
import random
import threading
import concurrent.futures
import multiprocessing as mp

import cv2
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from ultralytics import YOLO

from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

warnings = __import__('warnings')
warnings.filterwarnings("ignore", category=FutureWarning)
console = Console()

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
FRAME_SKIP = 1
BASE_MIN_BOX_SIZE = 45.0
MAX_STAGE1_WORKERS = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEOS_DIR = os.path.join(BASE_DIR, "VIDEOS")

candidate_roots = [
    "/mnt/DATA/SUSHANTH/EE23B044/ATTENDENCE RESULTS",
    "/mnt/DATA/SUSHANTH/EE23B044/ATTENDENCE_RESULTS",
    os.path.join(BASE_DIR, "ATTENDENCE RESULTS"),
    os.path.join(BASE_DIR, "ATTENDENCE_RESULTS"),
]
ATTENDANCE_ROOT = next((cr for cr in candidate_roots if os.path.exists(cr)), os.path.join(BASE_DIR, "ATTENDENCE RESULTS"))
FEATURE_CACHE_DIR = os.path.join(ATTENDANCE_ROOT, "feature_cache")
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

# ==========================================
# 2. MODELS & UTILITIES
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
        t = threading.Thread(target=self.update, args=(), daemon=True)
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

class BasicBlockIR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth)
        )
    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)

class AdaFaceIResNet50(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        blocks = [(64, 64, 2, 3), (64, 128, 2, 4), (128, 256, 2, 14), (256, 512, 2, 3)]
        modules = []
        for in_c, out_c, stride, num_blocks in blocks:
            modules.append(BasicBlockIR(in_c, out_c, stride))
            for _ in range(num_blocks - 1):
                modules.append(BasicBlockIR(out_c, out_c, 1))
        self.body = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_size),
            nn.BatchNorm1d(embedding_size, affine=False)
        )
    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        raw_emb = self.output_layer(x)
        norms = torch.norm(raw_emb, p=2, dim=1, keepdim=True)
        normed_emb = raw_emb / (norms + 1e-6)
        return normed_emb, norms

def load_adaface_model(device, use_half=False):
    model = AdaFaceIResNet50(embedding_size=512)
    for wf in ['adaface_ir50_webface4m.ckpt', 'adaface_ir50_webface4m.pth', 'adaface_ir50_webface4m.pt']:
        wf_path = os.path.join(BASE_DIR, wf) if not os.path.exists(wf) else wf
        if os.path.exists(wf_path):
            ckpt = torch.load(wf_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            clean_sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
            if not clean_sd:
                clean_sd = {k: v for k, v in sd.items() if k in model.state_dict()}
            model.load_state_dict(clean_sd, strict=False)
            model.eval().to(device)
            if use_half:
                model = model.half()
            return model
    raise FileNotFoundError("AdaFace checkpoint not found!")

def load_yolo_face(device):
    for model_path in ['yolo26n-face.pt', 'yolov8n-face.pt']:
        mp_full = os.path.join(BASE_DIR, model_path) if not os.path.exists(model_path) else model_path
        if os.path.exists(mp_full):
            model = YOLO(mp_full, task='detect')
            model.to(device)
            return model
    raise FileNotFoundError("No YOLO face model found!")

OCSORT_TRACKER_PATH = os.path.join(BASE_DIR, "custom_ocsort.yaml")

def ensure_ocsort_tracker():
    tracker_cfg = {
        'tracker_type': 'bytetrack',
        'track_high_thresh': 0.5,
        'track_low_thresh': 0.1,
        'new_track_thresh': 0.6,
        'track_buffer': 50,
        'match_thresh': 0.8,
        'fuse_score': True,
    }
    with open(OCSORT_TRACKER_PATH, 'w') as f:
        yaml.dump(tracker_cfg, f, default_flow_style=False)

ARCFACE_DST = torch.tensor([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=torch.float32)
_dst_norm_const = (ARCFACE_DST / 111.0) * 2.0 - 1.0
_X_const = torch.cat([_dst_norm_const, torch.ones((5, 1), dtype=torch.float32)], dim=1)
ARCFACE_PINV = torch.linalg.pinv(_X_const).contiguous()

def extract_faces_112(frame_tensor_chw, results, valid_mask, valid_boxes, arcface_pinv, frame_h, frame_w, device):
    has_real_kpts = (
        results[0].keypoints is not None and
        results[0].keypoints.xy is not None and
        results[0].keypoints.xy.numel() > 0 and
        results[0].keypoints.xy.shape[-2] >= 5 and
        (results[0].keypoints.xy.abs().sum() > 1.0).item()
    )
    if has_real_kpts:
        kpts = results[0].keypoints.xy.to(device).float()[valid_mask][:, :5, :]
        N = kpts.size(0)
        src_norm = torch.stack([
            (kpts[:, :, 0] / (frame_w - 1.0)) * 2.0 - 1.0,
            (kpts[:, :, 1] / (frame_h - 1.0)) * 2.0 - 1.0
        ], dim=-1)
        pinv = arcface_pinv.to(device).float()
        theta_T = torch.matmul(pinv.unsqueeze(0), src_norm)
        theta = theta_T.permute(0, 2, 1).contiguous()
        grid = F.affine_grid(theta, size=(N, 3, 112, 112), align_corners=True)
        sampled = F.grid_sample(frame_tensor_chw, grid.view(1, N * 112, 112, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
        return sampled.squeeze(0).view(3, N, 112, 112).permute(1, 0, 2, 3).contiguous().detach()
    else:
        boxes_m = valid_boxes.clone().float()
        margin_x = (boxes_m[:, 2] - boxes_m[:, 0]) * 0.15
        margin_y = (boxes_m[:, 3] - boxes_m[:, 1]) * 0.15
        boxes_m[:, 0] = torch.clamp(boxes_m[:, 0] - margin_x, min=0.0)
        boxes_m[:, 1] = torch.clamp(boxes_m[:, 1] - margin_y, min=0.0)
        boxes_m[:, 2] = torch.clamp(boxes_m[:, 2] + margin_x, max=float(frame_w))
        boxes_m[:, 3] = torch.clamp(boxes_m[:, 3] + margin_y, max=float(frame_h))
        batch_idx = torch.zeros((boxes_m.size(0), 1), device=device, dtype=torch.float32)
        roi_boxes = torch.cat((batch_idx, boxes_m), dim=1)
        return ops.roi_align(frame_tensor_chw, roi_boxes, output_size=(112, 112)).detach()

def format_timestamp(frame_count, fps):
    fps = fps if fps > 0 else 30
    total_seconds = int(frame_count // fps)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ==========================================
# 3. EXTRACTION WORKER
# ==========================================
ext_yolo_model = None
ext_resnet     = None
ext_device     = None
ext_use_half   = False

def init_stage1_worker():
    global ext_yolo_model, ext_resnet, ext_device, ext_use_half
    cv2.setNumThreads(1)
    time.sleep(random.uniform(0.1, 1.0))
    ext_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ext_use_half = torch.cuda.is_available()
    ext_yolo_model = load_yolo_face(ext_device)
    ext_resnet = load_adaface_model(ext_device, use_half=ext_use_half)

def extract_video_features_worker(video_filename, input_dir, cache_dir, progress_q):
    global ext_yolo_model, ext_resnet, ext_device, ext_use_half

    video_stem = os.path.splitext(video_filename)[0]
    cache_path = os.path.join(cache_dir, f"{video_stem}_features.pt")

    if hasattr(ext_yolo_model, 'predictor') and ext_yolo_model.predictor is not None:
        if hasattr(ext_yolo_model.predictor, 'trackers') and ext_yolo_model.predictor.trackers:
            for tracker in ext_yolo_model.predictor.trackers:
                tracker.reset()

    video_path = os.path.join(input_dir, video_filename)
    video_stream = ThreadedVideoReader(video_path).start()
    total_frames = max(video_stream.total_frames, 1)
    fps = video_stream.fps

    progress_q.put({"type": "start", "file": video_filename, "total_frames": total_frames})

    local_id_counter = 0
    raw_to_local_id = {}
    active_tracks = {}
    archived_tracks = {}
    frame_count = 0
    t_start = time.perf_counter()

    try:
        while video_stream.more():
            frame = video_stream.read()
            if frame is None:
                break

            results = ext_yolo_model.track(
                frame, persist=True, tracker=OCSORT_TRACKER_PATH, verbose=False, half=ext_use_half, imgsz=640
            )
            has_detections = results[0].boxes.id is not None
            local_ids_list = []

            if has_detections:
                boxes = results[0].boxes.xyxy.to(ext_device)
                raw_ids = results[0].boxes.id.to(ext_device).int().cpu().tolist()

                for i, r_id in enumerate(raw_ids):
                    if r_id not in raw_to_local_id:
                        local_id_counter += 1
                        raw_to_local_id[r_id] = local_id_counter

                    t_id = raw_to_local_id[r_id]
                    local_ids_list.append(t_id)

                    if t_id not in active_tracks:
                        active_tracks[t_id] = {
                            'frames_alive': 0, 'missing_frames': 0, 'start_time': format_timestamp(frame_count, fps),
                            'box_w': [], 'box_h': [], 'embs': [], 'norms': [], 'sample_frames': [], 'frame_boxes': {}
                        }
                    active_tracks[t_id]['frames_alive'] += 1
                    box_coord = boxes[i].int().cpu().tolist()
                    active_tracks[t_id]['frame_boxes'][frame_count] = (
                        int(box_coord[0]), int(box_coord[1]), int(box_coord[2]), int(box_coord[3])
                    )

                if frame_count % FRAME_SKIP == 0:
                    box_w = boxes[:, 2] - boxes[:, 0]
                    box_h = boxes[:, 3] - boxes[:, 1]
                    aspect_ratios = box_w / (box_h + 1e-6)

                    valid_mask = ((box_w >= BASE_MIN_BOX_SIZE) & (box_h >= BASE_MIN_BOX_SIZE) & 
                                  (aspect_ratios >= 0.7) & (aspect_ratios <= 1.7))

                    if valid_mask.any():
                        valid_boxes = boxes[valid_mask].clone()
                        valid_w = box_w[valid_mask].cpu().tolist()
                        valid_h = box_h[valid_mask].cpu().tolist()
                        valid_mask_cpu = valid_mask.cpu().numpy()
                        valid_local_ids = [local_ids_list[k] for k in range(len(local_ids_list)) if valid_mask_cpu[k]]

                        frame_tensor_chw = (
                            torch.from_numpy(frame).to(ext_device, non_blocking=True)
                            .permute(2, 0, 1).unsqueeze(0).contiguous().float()
                        )
                        crops = extract_faces_112(frame_tensor_chw, results, valid_mask, valid_boxes, ARCFACE_PINV, frame.shape[0], frame.shape[1], ext_device)
                        del frame_tensor_chw

                        with torch.inference_mode():
                            batch_tensor = (crops / 127.5) - 1.0
                            if ext_use_half:
                                batch_tensor = batch_tensor.half()
                            normed_embeddings, feature_norms = ext_resnet(batch_tensor)
                            embs_cpu = normed_embeddings.float().cpu()
                            norms_cpu = feature_norms.squeeze(1).float().cpu()

                        for i, t_id in enumerate(valid_local_ids):
                            active_tracks[t_id]['box_w'].append(valid_w[i])
                            active_tracks[t_id]['box_h'].append(valid_h[i])
                            active_tracks[t_id]['embs'].append(embs_cpu[i:i+1])
                            active_tracks[t_id]['norms'].append(norms_cpu[i:i+1])
                            active_tracks[t_id]['sample_frames'].append(frame_count)

            alive_ids = set(local_ids_list) if has_detections else set()
            for t_id in list(active_tracks.keys()):
                if t_id not in alive_ids:
                    active_tracks[t_id]['missing_frames'] += 1
                    if active_tracks[t_id]['missing_frames'] > 50:
                        archived_tracks[t_id] = active_tracks.pop(t_id)
                else:
                    active_tracks[t_id]['missing_frames'] = 0

            frame_count += 1
            if frame_count % 15 == 0:
                elapsed = time.perf_counter() - t_start
                curr_fps = frame_count / elapsed if elapsed > 0 else 0.0
                progress_q.put({"type": "progress", "file": video_filename, "frame": frame_count, "total": total_frames, "fps": curr_fps})

        final_mem = {**archived_tracks, **active_tracks}
        serialized_tracks = {}
        for t_id, data in sorted(final_mem.items(), key=lambda x: x[0]):
            if data['embs']:
                serialized_tracks[t_id] = {
                    'frames_alive': data['frames_alive'],
                    'start_time': data.get('start_time', '00:00:00'),
                    'box_w': torch.tensor(data['box_w'], dtype=torch.float32),
                    'box_h': torch.tensor(data['box_h'], dtype=torch.float32),
                    'embs': torch.cat(data['embs'], dim=0).half(),
                    'norms': torch.cat(data['norms'], dim=0).float(),
                    'sample_frames': torch.tensor(data.get('sample_frames', []), dtype=torch.int32),
                    'frame_boxes': data.get('frame_boxes', {}),
                }

        torch.save(serialized_tracks, cache_path)
        progress_q.put({"type": "finish", "file": video_filename})
        return video_filename, cache_path, "Extracted"

    except Exception as e:
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass
        progress_q.put({"type": "error", "file": video_filename, "err": str(e)})
        return video_filename, "", f"Error: {str(e)}"
    finally:
        video_stream.stop()
        del active_tracks, archived_tracks, raw_to_local_id
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_stage1_dashboard(progress_q, total_videos, stop_event):
    active_videos = {}
    completed = 0
    with Live(auto_refresh=False, refresh_per_second=4) as live:
        while not stop_event.is_set() or not progress_q.empty():
            while True:
                try:
                    msg = progress_q.get_nowait()
                    mtype = msg.get("type")
                    fname = msg.get("file")
                    if mtype == "start":
                        active_videos[fname] = {"frame": 0, "total": msg.get("total_frames", 1), "fps": 0.0}
                    elif mtype == "progress":
                        if fname in active_videos:
                            active_videos[fname]["frame"] = msg["frame"]
                            active_videos[fname]["total"] = msg["total"]
                            active_videos[fname]["fps"] = msg["fps"]
                    elif mtype in ("finish", "skip"):
                        active_videos.pop(fname, None)
                        completed += 1
                    elif mtype == "error":
                        active_videos.pop(fname, None)
                except queue.Empty:
                    break

            table = Table(title=f"[bold cyan]Stage 1 Feature Pre-Caching ({completed}/{total_videos} Videos Ready)[/bold cyan]", box=None)
            table.add_column("Video File", style="cyan", width=35)
            table.add_column("Frames", style="magenta", width=18, justify="right")
            table.add_column("Progress", style="yellow", width=12, justify="right")
            table.add_column("FPS", style="green", width=10, justify="right")

            if active_videos:
                for fn, st in sorted(active_videos.items()):
                    pct = min(100.0, (st["frame"] / st["total"]) * 100.0) if st["total"] > 0 else 0.0
                    table.add_row(fn, f"{st['frame']}/{st['total']}", f"{pct:5.1f}%", f"{st['fps']:5.1f}")
            else:
                table.add_row("[italic dim]Initializing / Finishing extraction workers...[/italic dim]", "-", "-", "-")

            live.update(table, refresh=True)
            time.sleep(0.2)

def get_cache_path(cache_dir, stem):
    p1 = os.path.join(cache_dir, f"{stem}_features.pt")
    if os.path.exists(p1) and os.path.getsize(p1) > 1024:
        return p1
    p2 = os.path.join(cache_dir, f"{stem}_raw_features.pt")
    if os.path.exists(p2) and os.path.getsize(p2) > 1024:
        return p2
    return p1

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    mp.freeze_support()
    ctx = mp.get_context('spawn')
    
    console.print(Panel(
        f"[bold cyan]Hardware Isolation:[/bold cyan] CUDA_VISIBLE_DEVICES='0'\n"
        f"[bold green]Feature Cache Directory:[/bold green] {FEATURE_CACHE_DIR}",
        title="Standalone YOLO+AdaFace Feature Pre-Cacher",
        expand=False
    ))

    video_extensions = ('.mp4', '.mkv', '.avi', '.mov')
    try:
        target_videos = sorted([
            f for f in os.listdir(INPUT_VIDEOS_DIR)
            if f.lower().endswith(video_extensions) and not f.startswith('.')
        ])
    except FileNotFoundError:
        console.print(f"[bold red]Error: '{INPUT_VIDEOS_DIR}' folder not found![/bold red]")
        sys.exit(1)

    if not target_videos:
        console.print("[bold red]No videos found in VIDEOS folder![/bold red]")
        sys.exit(0)

    # Check which videos need caching
    ready_count = 0
    missing_cache = []
    for v in target_videos:
        stem = os.path.splitext(v)[0]
        cpath = get_cache_path(FEATURE_CACHE_DIR, stem)
        if os.path.exists(cpath) and os.path.getsize(cpath) > 1024:
            ready_count += 1
        else:
            missing_cache.append(v)

    console.print(f"\n[bold white]Total Videos:[/bold white] [bold cyan]{len(target_videos)}[/bold cyan]")
    console.print(f"[bold white]Already Cached:[/bold white] [bold green]{ready_count}[/bold green]")
    console.print(f"[bold white]Pending Extraction:[/bold white] [bold yellow]{len(missing_cache)}[/bold yellow]\n")

    if not missing_cache:
        console.print("[bold green]All videos are completely cached. You are ready to run sweep_yolo_sota.py![/bold green]")
        sys.exit(0)

    ensure_ocsort_tracker()
    console.print(f"[bold yellow]Extracting YOLO+AdaFace features for {len(missing_cache)} missing video(s)...[/bold yellow]")

    manager = ctx.Manager()
    progress_q = manager.Queue()
    stop_event = threading.Event()

    dash_t = threading.Thread(
        target=run_stage1_dashboard,
        args=(progress_q, len(missing_cache), stop_event),
        daemon=True
    )
    dash_t.start()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(MAX_STAGE1_WORKERS, len(missing_cache)),
        mp_context=ctx,
        initializer=init_stage1_worker
    ) as executor:
        futures = [
            executor.submit(extract_video_features_worker, v, INPUT_VIDEOS_DIR, FEATURE_CACHE_DIR, progress_q)
            for v in missing_cache
        ]
        for fut in concurrent.futures.as_completed(futures):
            vname, _, status = fut.result()
            if status.startswith("Error:"):
                console.print(f"\n[bold red]Failed feature extraction on {vname}: {status}[/bold red]")

    stop_event.set()
    dash_t.join(timeout=1.0)
    console.print(f"\n[bold green]Feature Pre-Caching Complete! You can now run sweep_yolo_sota.py instantly.[/bold green]")
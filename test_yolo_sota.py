# pyright: reportMissingImports=false
# test_yolo_sota.py – SOTA Multi-Process Batch Video Evaluation & Rich Dashboard
# ==============================================================================
# Upgrades:
#   1. Strict GPU 0 isolation & unified ATTENDENCE RESULTS paths.
#   2. Stage 1: One-time extraction of features, landmarks, and frame boxes to .pt cache.
#   3. Phase 2: 100% cache-driven evaluation + fast video annotation rendering (no YOLO/AdaFace reload).
#   4. Fix Detection Count column to track per-Track-ID gate passes (+1 per passing track).
#   5. Rich Live telemetry dashboards for both Stage 1 and Phase 2.
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
import pickle
import warnings
import threading
from collections import Counter
import concurrent.futures
import multiprocessing as mp

import cv2
import faiss  # type: ignore
import yaml
import numpy as np
import pandas as pd

# NumPy 2.x -> 1.x pickle compatibility shim
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = np.core
    for _attr in dir(np.core):
        sys.modules[f'numpy._core.{_attr}'] = getattr(np.core, _attr)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from ultralytics import YOLO  # type: ignore

from rich.live import Live  # type: ignore
from rich.table import Table  # type: ignore
from rich.console import Console  # type: ignore
from rich.panel import Panel  # type: ignore
from rich.progress import (  # type: ignore
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    MofNCompleteColumn, SpinnerColumn
)

warnings.filterwarnings("ignore", category=FutureWarning)
console = Console()

# ==========================================
# 1. OPTIMAL SOTA HYPERPARAMETERS & PATHS
# ==========================================
CONFIDENCE_THRESHOLD   = 0.40
FRAME_SKIP             = 1
FRAMES_PER_VOTE        = 3
NORM_QUALITY_THRESHOLD = 7.0
EMA_MOMENTUM           = 0.85
SOFTMAX_TAU            = 5.0
MIN_BOX_SIZE           = 60.0
MIN_TOTAL_FRAMES       = 30
MIN_TOTAL_SAMPLES      = 20
MIN_SAMPLE_RATIO       = 0.40
MIN_WIN_RATIO          = 0.45

BASE_MIN_BOX_SIZE      = 45.0  # Stage 1 base cache filter

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
MINE_OUTPUT_DIR   = os.path.join(ATTENDANCE_ROOT, "MINE")
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
os.makedirs(MINE_OUTPUT_DIR, exist_ok=True)

MAX_STAGE1_WORKERS = 4
MAX_PHASE2_WORKERS = min(8, max(2, (os.cpu_count() or 8) // 4))


# ==========================================
# 2. VIDEO STREAMING & WRITING UTILITIES
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
            name, conf = names[i]

            color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
            conf_str = f" {conf * 100:.1f}%" if conf > 0 else ""
            label = f"ID:{t_id} {name}{conf_str}"

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
    out.release()


def format_timestamp(frame_count, fps):
    fps = fps if fps > 0 else 30
    total_seconds = int(frame_count // fps)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==========================================
# 3. ADAFACE & YOLO MODELS (STAGE 1)
# ==========================================
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
            clean_sd = {}
            for k, v in sd.items():
                if k.startswith('model.'):
                    clean_sd[k[len('model.'):]] = v
                elif k in model.state_dict():
                    clean_sd[k] = v
            model.load_state_dict(clean_sd, strict=False)
            model.eval().to(device)
            if use_half:
                model = model.half()
            return model
    raise FileNotFoundError("AdaFace checkpoint 'adaface_ir50_webface4m.ckpt' not found!")


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
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
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
        sampled = F.grid_sample(
            frame_tensor_chw,
            grid.view(1, N * 112, 112, 2),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
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


# ==========================================
# 4. STAGE 1: ONE-TIME YOLO+ADAFACE EXTRACTION
# ==========================================
ext_yolo_model = None
ext_resnet     = None
ext_device     = None
ext_use_half   = False


def is_valid_cache_for_test(cache_path):
    """Verifies cache exists, is non-empty, and has frame_boxes for video rendering."""
    if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024):
        return False
    try:
        data = torch.load(cache_path, map_location='cpu', weights_only=False)
        if not isinstance(data, dict):
            return False
        if len(data) > 0:
            first_track = next(iter(data.values()))
            if 'frame_boxes' not in first_track:
                return False
        return True
    except Exception:
        return False


def get_cache_path(cache_dir, stem):
    p1 = os.path.join(cache_dir, f"{stem}_features.pt")
    if os.path.exists(p1) and os.path.getsize(p1) > 1024:
        return p1
    p2 = os.path.join(cache_dir, f"{stem}_raw_features.pt")
    if os.path.exists(p2) and os.path.getsize(p2) > 1024:
        return p2
    return p1


def init_stage1_worker():
    global ext_yolo_model, ext_resnet, ext_device, ext_use_half
    cv2.setNumThreads(1)
    time.sleep(random.uniform(0.1, 1.0))
    ext_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ext_use_half = torch.cuda.is_available()
    ext_yolo_model = load_yolo_face(ext_device)
    ext_resnet = load_adaface_model(ext_device, use_half=ext_use_half)


def extract_video_features_worker(video_filename, input_dir, cache_dir, progress_q):
    """
    Stage 1 worker: extracts face crops, AdaFace embeddings, norms, sample frames,
    and all per-frame bounding boxes for Phase 2 video rendering.
    """
    global ext_yolo_model, ext_resnet, ext_device, ext_use_half

    video_stem = os.path.splitext(video_filename)[0]
    cache_path = os.path.join(cache_dir, f"{video_stem}_features.pt")

    if is_valid_cache_for_test(cache_path):
        progress_q.put({"type": "skip", "file": video_filename})
        return video_filename, cache_path, "Skipped"

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
                frame,
                persist=True,
                tracker=OCSORT_TRACKER_PATH,
                verbose=False,
                half=ext_use_half,
                imgsz=640
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
                            'frames_alive': 0,
                            'missing_frames': 0,
                            'start_time': format_timestamp(frame_count, fps),
                            'box_w': [],
                            'box_h': [],
                            'embs': [],
                            'norms': [],
                            'sample_frames': [],
                            'frame_boxes': {}
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

                    valid_mask = (
                        (box_w >= BASE_MIN_BOX_SIZE) &
                        (box_h >= BASE_MIN_BOX_SIZE) &
                        (aspect_ratios >= 0.7) &
                        (aspect_ratios <= 1.7)
                    )

                    if valid_mask.any():
                        valid_boxes = boxes[valid_mask].clone()
                        valid_w = box_w[valid_mask].cpu().tolist()
                        valid_h = box_h[valid_mask].cpu().tolist()
                        valid_mask_cpu = valid_mask.cpu().numpy()
                        valid_local_ids = [local_ids_list[k] for k in range(len(local_ids_list)) if valid_mask_cpu[k]]

                        frame_tensor_chw = (
                            torch.from_numpy(frame)
                            .to(ext_device, non_blocking=True)
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .contiguous()
                            .float()
                        )
                        crops = extract_faces_112(
                            frame_tensor_chw, results, valid_mask, valid_boxes,
                            ARCFACE_PINV, frame.shape[0], frame.shape[1], ext_device
                        )
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
                progress_q.put({
                    "type": "progress",
                    "file": video_filename,
                    "frame": frame_count,
                    "total": total_frames,
                    "fps": curr_fps
                })

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
            else:
                serialized_tracks[t_id] = {
                    'frames_alive': data['frames_alive'],
                    'start_time': data.get('start_time', '00:00:00'),
                    'box_w': torch.empty(0, dtype=torch.float32),
                    'box_h': torch.empty(0, dtype=torch.float32),
                    'embs': torch.empty((0, 512), dtype=torch.float16),
                    'norms': torch.empty(0, dtype=torch.float32),
                    'sample_frames': torch.empty(0, dtype=torch.int32),
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

            table = Table(
                title=f"[bold cyan]Stage 1: Extracting YOLO + AdaFace Feature Cache ({completed}/{total_videos} Videos Ready)[/bold cyan]",
                box=None
            )
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


def ensure_all_features_cached(target_videos, input_dir, cache_dir, ctx):
    os.makedirs(cache_dir, exist_ok=True)
    ready_count = 0
    missing_cache = []

    for v in target_videos:
        stem = os.path.splitext(v)[0]
        cpath = get_cache_path(cache_dir, stem)
        if is_valid_cache_for_test(cpath):
            ready_count += 1
        else:
            missing_cache.append(v)

    console.print(Panel(
        f"[bold white]Total Videos:[/bold white] [bold cyan]{len(target_videos)}[/bold cyan]  |  "
        f"[bold white]Cached (.pt ready):[/bold white] [bold green]{ready_count}[/bold green]  |  "
        f"[bold white]Pending Extraction:[/bold white] [bold yellow]{len(missing_cache)}[/bold yellow]",
        title="Stage 1 Feature Cache Status",
        expand=False
    ))

    if not missing_cache:
        console.print("[bold green]OK All feature caches ready. Skipping Stage 1 extraction completely![/bold green]")
        return

    ensure_ocsort_tracker()
    console.print(f"[bold yellow]Extracting YOLO+AdaFace features & boxes for {len(missing_cache)} video(s)...[/bold yellow]")

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
            executor.submit(extract_video_features_worker, v, input_dir, cache_dir, progress_q)
            for v in missing_cache
        ]
        for fut in concurrent.futures.as_completed(futures):
            vname, _, status = fut.result()
            if status.startswith("Error:"):
                raise RuntimeError(f"Failed Stage 1 extraction on {vname}: {status}")

    stop_event.set()
    dash_t.join(timeout=1.0)
    console.print(f"[bold green]OK Stage 1 feature extraction complete![/bold green]")


# ==========================================
# 5. PHASE 2: CACHED EVALUATION & MP4 RENDERING
# ==========================================
p2_target_names = []
p2_y_real       = []
p2_ref_embs     = None


def init_phase2_worker():
    global p2_target_names, p2_y_real, p2_ref_embs
    cv2.setNumThreads(1)

    faiss_index_path = os.path.join(BASE_DIR, 'face_attendance_sota_faiss.bin')
    meta_path        = os.path.join(BASE_DIR, 'face_attendance_sota_meta.pkl')

    if not os.path.exists(faiss_index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Run 'python train_yolo_sota.py' first to build FAISS gallery and metadata!")

    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)
    p2_target_names = meta_data['target_names']
    p2_y_real       = meta_data['y_real']

    index = faiss.read_index(faiss_index_path)
    index.nprobe = 20
    try:
        ref_np = index.reconstruct_n(0, index.ntotal)
    except AttributeError:
        index.make_direct_map()
        ref_np = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    p2_ref_embs = F.normalize(torch.from_numpy(ref_np).float(), p=2, dim=1)


def process_video_phase2(video_filename, input_dir, cache_dir, output_dir, progress_q):
    """
    Phase 2 worker:
      1. Loads .pt features (no YOLO/AdaFace model reloads).
      2. Runs quality-weighted EMA softmax voting pipeline.
      3. Outputs {stem}_DEBUG_Tracks.csv and {stem}_output.csv (Detection Count = per-Track-ID gate passes).
      4. Streams raw video and renders annotated {stem}_output.mp4 via ThreadedVideoReader + video_writer_worker.
    """
    global p2_target_names, p2_y_real, p2_ref_embs

    video_stem = os.path.splitext(video_filename)[0]
    output_csv = os.path.join(output_dir, f"{video_stem}_output.csv")
    debug_csv  = os.path.join(output_dir, f"{video_stem}_DEBUG_Tracks.csv")
    final_mp4  = os.path.join(output_dir, f"{video_stem}_output.mp4")

    # Resume capability: skip if all 3 files exist and MP4 is valid (> 1024 bytes)
    if os.path.exists(output_csv) and os.path.exists(debug_csv) and os.path.exists(final_mp4) and os.path.getsize(final_mp4) > 1024:
        progress_q.put({"type": "skip", "file": video_filename})
        return video_filename, "Skipped"

    cpath = get_cache_path(cache_dir, video_stem)
    if not is_valid_cache_for_test(cpath):
        progress_q.put({"type": "error", "file": video_filename, "err": "Cache file missing or invalid (no frame_boxes)"})
        return video_filename, "Error: Cache missing or invalid"

    raw_tracks = torch.load(cpath, map_location='cpu', weights_only=False)

    video_path = os.path.join(input_dir, video_filename)
    video_stream = ThreadedVideoReader(video_path).start()
    total_frames = max(video_stream.total_frames, 1)

    progress_q.put({"type": "start", "file": video_filename, "total_frames": total_frames})

    # 1. Run Voting Pipeline over Cached Track Features
    debug_data = []
    student_presence = {name: False for name in p2_target_names}
    student_detection_count = {name: 0 for name in p2_target_names}

    # track_timeline: t_id -> list of (completion_frame, winner, conf)
    track_timeline = {}

    for t_id, td in sorted(raw_tracks.items(), key=lambda x: x[0]):
        frames_alive = int(td.get('frames_alive', 0))
        start_time   = td.get('start_time', '00:00:00')
        box_w        = td['box_w'].float()
        box_h        = td['box_h'].float()
        embs         = td['embs'].float()
        norms        = td['norms'].float()
        sample_frames = td.get('sample_frames', None)
        if sample_frames is not None and len(sample_frames) == 0:
            sample_frames = None

        track_timeline[t_id] = []
        all_preds = []

        if embs.numel() > 0:
            box_ok = (box_w >= MIN_BOX_SIZE) & (box_h >= MIN_BOX_SIZE)
            valid_indices = box_ok.nonzero(as_tuple=True)[0]
            S_valid = valid_indices.numel()

            num_windows = S_valid // FRAMES_PER_VOTE
            ema_emb = None

            for w in range(num_windows):
                w_idx = valid_indices[w * FRAMES_PER_VOTE : (w + 1) * FRAMES_PER_VOTE]
                w_embs  = embs[w_idx]    # (fpv, 512)
                w_norms = norms[w_idx]   # (fpv,)

                last_sample_idx = int(w_idx[-1].item())
                w_frame = int(sample_frames[last_sample_idx].item()) if sample_frames is not None else (last_sample_idx * FRAME_SKIP)

                q_mask = (w_norms >= NORM_QUALITY_THRESHOLD)
                if not q_mask.any():
                    all_preds.append("Unknown")
                    track_timeline[t_id].append((w_frame, "Unknown", 0.0))
                    continue

                weights = torch.softmax(w_norms / SOFTMAX_TAU, dim=0).unsqueeze(1)
                e_win = F.normalize(torch.sum(weights * w_embs, dim=0, keepdim=True), p=2, dim=1)

                if ema_emb is None:
                    ema_emb = e_win
                else:
                    ema_emb = F.normalize(EMA_MOMENTUM * ema_emb + (1.0 - EMA_MOMENTUM) * e_win, p=2, dim=1)

                sim = torch.mm(ema_emb, p2_ref_embs.t())
                max_sim, max_idx = sim.max(dim=1)
                conf_score = float(max_sim.item())
                winner = p2_target_names[p2_y_real[max_idx.item()]] if conf_score >= CONFIDENCE_THRESHOLD else "Unknown"

                all_preds.append(winner)
                track_timeline[t_id].append((w_frame, winner, conf_score))

        # Gate Condition Evaluation
        valid_preds = [p for p in all_preds if p != "Unknown"]
        valid_votes_count = len(valid_preds)
        total_samples = len(all_preds)

        if valid_votes_count > 0:
            counts = Counter(valid_preds)
            winner = counts.most_common(1)[0][0]
            win_ratio = counts[winner] / valid_votes_count
            sample_ratio = counts[winner] / total_samples if total_samples > 0 else 0.0

            status = "Passed" if (
                frames_alive >= MIN_TOTAL_FRAMES and
                total_samples >= MIN_TOTAL_SAMPLES and
                sample_ratio >= MIN_SAMPLE_RATIO and
                win_ratio >= MIN_WIN_RATIO
            ) else "Failed"
        else:
            winner = "Unknown"
            status = "Failed"

        # TASK 3: Detection Count = how many distinct Track IDs passed the gate
        if status == "Passed" and winner != "Unknown":
            student_presence[winner] = True
            student_detection_count[winner] += 1

        debug_data.append({
            'Track ID': t_id,
            'Start Time': start_time,
            'Total Frames': frames_alive,
            'Valid Votes': valid_votes_count,
            'Total Preds (inc. Unknown)': total_samples,
            'Predicted Identity': winner,
            'Gate Status': status,
            'Breakdown': dict(Counter(all_preds))
        })

    # Save CSVs
    pd.DataFrame(debug_data).to_csv(debug_csv, index=False)
    output_data = [
        {
            'Name': s,
            'Status': 'Present' if student_presence[s] else 'Absent',
            'Detection Count': student_detection_count[s]
        }
        for s in p2_target_names
    ]
    pd.DataFrame(output_data).to_csv(output_csv, index=False)

    # 2. Build Fast Frame Lookup for Video Annotation Rendering
    # frame_lookup: frame_idx -> list of (box, t_id, (winner, conf))
    frame_lookup = {}
    for t_id, td in raw_tracks.items():
        boxes_dict = td.get('frame_boxes', {})
        t_line = track_timeline.get(t_id, [])

        for f_idx, box in boxes_dict.items():
            label = ("Analyzing...", 0.0)
            for w_frame, w_win, w_conf in t_line:
                if f_idx >= w_frame:
                    label = (w_win, w_conf)
                else:
                    break
            frame_lookup.setdefault(f_idx, []).append((box, t_id, label))

    # 3. Stream Video and Render MP4
    write_queue = queue.Queue(maxsize=128)
    writer_thread = threading.Thread(
        target=video_writer_worker,
        args=(write_queue, final_mp4, video_stream.fps, video_stream.frame_width, video_stream.frame_height),
        daemon=True
    )
    writer_thread.start()

    frame_count = 0
    t_render_start = time.perf_counter()

    try:
        while video_stream.more():
            frame = video_stream.read()
            if frame is None:
                break

            items = frame_lookup.get(frame_count, [])
            boxes_cpu  = [it[0] for it in items]
            local_ids  = [it[1] for it in items]
            frame_names = [it[2] for it in items]

            write_queue.put((frame, boxes_cpu, local_ids, frame_names))
            frame_count += 1

            if frame_count % 15 == 0:
                elapsed = time.perf_counter() - t_render_start
                curr_fps = frame_count / elapsed if elapsed > 0 else 0.0
                progress_q.put({
                    "type": "progress",
                    "file": video_filename,
                    "frame": frame_count,
                    "total": total_frames,
                    "fps": curr_fps
                })

        write_queue.put(None)
        writer_thread.join()
        progress_q.put({"type": "finish", "file": video_filename})
        return video_filename, "Processed"

    except Exception as e:
        write_queue.put(None)
        writer_thread.join()
        progress_q.put({"type": "error", "file": video_filename, "err": str(e)})
        return video_filename, f"Error: {str(e)}"
    finally:
        video_stream.stop()
        del raw_tracks, frame_lookup, track_timeline
        gc.collect()


def run_phase2_dashboard(progress_q, total_videos, stop_event):
    active_videos = {}
    completed = 0
    failed = 0
    last_error = ""

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
                        failed += 1
                        last_error = f"{fname}: {msg.get('err', '')}"
                except queue.Empty:
                    break

            err_str = f" | [bold red]Failed: {failed} ({last_error})[/bold red]" if failed > 0 else ""
            table = Table(
                title=f"[bold green]Phase 2 Video Evaluation & MP4 Rendering ({completed}/{total_videos} Videos Finished)[/bold green]{err_str}",
                box=None
            )
            table.add_column("Running File", style="cyan", width=35, no_wrap=True)
            table.add_column("Frames", style="magenta", width=18, justify="right")
            table.add_column("Completion", style="yellow", width=12, justify="right")
            table.add_column("FPS", style="green", width=10, justify="right")

            if active_videos:
                for fn, st in sorted(active_videos.items()):
                    pct = min(100.0, (st["frame"] / st["total"]) * 100.0) if st["total"] > 0 else 0.0
                    table.add_row(fn, f"{st['frame']}/{st['total']}", f"{pct:5.1f}%", f"{st['fps']:5.1f}")
            else:
                table.add_row("[italic dim]Waiting for worker threads...[/italic dim]", "-", "-", "-")

            live.update(table, refresh=True)
            time.sleep(0.2)


# ==========================================
# 6. MAIN ENTRY POINT
# ==========================================
if __name__ == '__main__':
    mp.freeze_support()
    ctx = mp.get_context('spawn')
    main_pid = os.getpid()

    console.print(Panel(
        f"[bold yellow]Main Process ID (PID):[/bold yellow] [bold white]{main_pid}[/bold white]\n"
        f"[bold cyan]Hardware:[/bold cyan] GPU 0 (CUDA_VISIBLE_DEVICES='0')\n"
        f"[bold green]Shared Feature Cache:[/bold green] {FEATURE_CACHE_DIR}\n"
        f"[bold magenta]MINE Output Dir:[/bold magenta] {MINE_OUTPUT_DIR}",
        title="Automated Attendance SOTA Video Pipeline",
        expand=False
    ))

    video_extensions = ('.mp4', '.mkv', '.avi', '.mov')
    target_videos = sorted([
        f for f in os.listdir(INPUT_VIDEOS_DIR)
        if f.lower().endswith(video_extensions) and not f.startswith('.')
    ])

    total_videos = len(target_videos)
    console.print(f"[bold cyan]Discovered {total_videos} videos in '{INPUT_VIDEOS_DIR}'.[/bold cyan]\n")

    if not target_videos:
        console.print("[bold red]No videos found in VIDEOS folder![/bold red]")
        sys.exit(0)

    # 1. Stage 1: Build or verify shared feature cache (.pt files)
    ensure_all_features_cached(target_videos, INPUT_VIDEOS_DIR, FEATURE_CACHE_DIR, ctx)

    # 2. Phase 2: Cached Evaluation + MP4 Rendering + Gate-Pass Count CSVs
    console.print(f"\n[bold yellow]Starting Phase 2 Cached Evaluation & MP4 Rendering ({total_videos} videos)...[/bold yellow]")

    manager = ctx.Manager()
    progress_q = manager.Queue()
    stop_event = threading.Event()

    dash_t = threading.Thread(
        target=run_phase2_dashboard,
        args=(progress_q, total_videos, stop_event),
        daemon=True
    )
    dash_t.start()

    errors_list = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_PHASE2_WORKERS,
        mp_context=ctx,
        initializer=init_phase2_worker
    ) as executor:
        futures = {
            executor.submit(process_video_phase2, v, INPUT_VIDEOS_DIR, FEATURE_CACHE_DIR, MINE_OUTPUT_DIR, progress_q): v
            for v in target_videos
        }

        for fut in concurrent.futures.as_completed(futures):
            vname = futures[fut]
            try:
                fname, status = fut.result()
                if status.startswith("Error:"):
                    errors_list.append((fname, status))
            except Exception as exc:
                errors_list.append((vname, str(exc)))

    stop_event.set()
    dash_t.join(timeout=1.0)

    if errors_list:
        console.print(f"\n[bold red]Completed Phase 2 with {len(errors_list)} errors:[/bold red]")
        for fname, err in errors_list[:10]:
            console.print(f"  -> [{fname}] {err}")
    else:
        console.print("\n[bold green]All videos successfully processed in Phase 2![/bold green]")

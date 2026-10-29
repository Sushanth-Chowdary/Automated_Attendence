# pyright: reportMissingImports=false
# sweep_yolo_sota.py – DGX A100 GPU-Accelerated Hyperparameter Optimizer (Stage 2)
# ==============================================================================
# Upgrades:
#   1. Strict GPU 0 isolation (leaves other GPUs completely untouched).
#   2. Main Process ID (PID) displayed at startup for SPCLAB-DGX_Utilization.xlsx.
#   3. Standardized directory layout under ATTENDENCE RESULTS:
#      - Shared feature cache: ATTENDENCE RESULTS/feature_cache
#      - Sweep outputs: ATTENDENCE RESULTS/sweep
#   4. All discovered videos from VIDEOS/ evaluated (instead of hardcoded 4).
#   5. Unified Stage 1 (.pt cache with frame boxes, timestamps, and landmarks).
#   6. Stage 2: Epoch-Based Coordinate Ascent / Progressive Local Neighborhood Optimizer.
#      - Locks low-variance variables (SOFTMAX_TAU=5.0, NORM_QUALITY=7.0, MIN_FRAMES=30).
#      - 7 active variables with 1D +-Delta, +-2Delta and 2D pairwise interaction steps.
#      - Vectorized GPU batch evaluation across all cached videos on CUDA:0.
#      - Real-time Rich progress bar, effective FPS telemetry, sweep_progress.json & live_status.txt.
#      - Full resume support from sweep_final_accuracy.csv & leaderboard visualization.
# ==============================================================================

import os
# Strictly isolate execution to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import sys
import json
import time
import queue
import random
import pickle
import warnings
import threading
import itertools
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# 1. OPTIMIZER CONFIGURATION & ACTIVE VARIABLES
# ==========================================
# Fixed low-variance variables locked from Stage 2 analysis
SOFTMAX_TAU            = 5.0
NORM_QUALITY_THRESHOLD = 7.0
MIN_TOTAL_FRAMES       = 30

# 7 Active parameters to optimize with bounds, initial steps, and type
ACTIVE_VARS = {
    "CONFIDENCE_THRESHOLD": {"center": 0.40, "min": 0.30, "max": 0.55, "step": 0.03, "is_int": False},
    "EMA_MOMENTUM":          {"center": 0.85, "min": 0.50, "max": 0.95, "step": 0.05, "is_int": False},
    "FRAMES_PER_VOTE":       {"center": 3,    "min": 2,    "max": 8,    "step": 1,    "is_int": True},
    "MIN_BOX_SIZE":          {"center": 60.0, "min": 45.0, "max": 75.0, "step": 5.0,  "is_int": False},
    "MIN_TOTAL_SAMPLES":     {"center": 20,   "min": 10,   "max": 40,   "step": 5,    "is_int": True},
    "MIN_SAMPLE_RATIO":      {"center": 0.40, "min": 0.20, "max": 0.60, "step": 0.05, "is_int": False},
    "MIN_WIN_RATIO":         {"center": 0.45, "min": 0.35, "max": 0.60, "step": 0.04, "is_int": False},
}

TOP4_VARS = ["MIN_WIN_RATIO", "EMA_MOMENTUM", "FRAMES_PER_VOTE", "CONFIDENCE_THRESHOLD"]

ALL_PARAM_KEYS = [
    "CONFIDENCE_THRESHOLD",
    "NORM_QUALITY_THRESHOLD",
    "EMA_MOMENTUM",
    "FRAMES_PER_VOTE",
    "SOFTMAX_TAU",
    "MIN_BOX_SIZE",
    "MIN_TOTAL_FRAMES",
    "MIN_TOTAL_SAMPLES",
    "MIN_SAMPLE_RATIO",
    "MIN_WIN_RATIO",
]

MAX_EPOCHS = 6
BASE_MIN_BOX_SIZE = 45.0  # Stage 1 base feature extraction threshold
FRAME_SKIP = 1
MAX_STAGE1_WORKERS = 4
GPU_BATCH_SIZE = 256

# ==========================================
# 2. STANDARDIZED DIRECTORY PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEOS_DIR = os.path.join(BASE_DIR, "VIDEOS")

candidate_roots = [
    "/mnt/DATA/SUSHANTH/EE23B044/ATTENDENCE RESULTS",
    "/mnt/DATA/SUSHANTH/EE23B044/ATTENDENCE_RESULTS",
    os.path.join(BASE_DIR, "ATTENDENCE RESULTS"),
    os.path.join(BASE_DIR, "ATTENDENCE_RESULTS"),
]
ATTENDANCE_ROOT = next((cr for cr in candidate_roots if os.path.exists(cr)), os.path.join(BASE_DIR, "ATTENDENCE RESULTS"))

RESULTS_BASE_DIR  = os.path.join(ATTENDANCE_ROOT, "Results")
FEATURE_CACHE_DIR = os.path.join(ATTENDANCE_ROOT, "feature_cache")
SWEEP_OUTPUT_DIR  = os.path.join(ATTENDANCE_ROOT, "sweep")

os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
os.makedirs(SWEEP_OUTPUT_DIR, exist_ok=True)


# ==========================================
# 3. GROUND TRUTH PREPARATION & TELEMETRY
# ==========================================
def prepare_subset_true_attendance(target_videos, results_base_dir, sweep_dir):
    """
    Extracts ground truth attendance for all unique video dates across all discovered videos.
    Saves true_attendance_all_videos.csv in ATTENDENCE RESULTS/sweep.
    """
    os.makedirs(sweep_dir, exist_ok=True)
    subset_true_path = os.path.join(sweep_dir, "true_attendance_all_videos.csv")
    full_true_path = os.path.join(results_base_dir, "filtered_true_attendance.csv")
    name_mapping_path = os.path.join(results_base_dir, "name_mapping.csv")

    if not os.path.exists(name_mapping_path):
        raise FileNotFoundError(f"Could not find name_mapping.csv at {name_mapping_path}")
    if not os.path.exists(full_true_path):
        raise FileNotFoundError(f"Could not find filtered_true_attendance.csv at {full_true_path}")

    name_map_df = pd.read_csv(name_mapping_path).dropna(subset=['Short_Name', 'Roll_Name'])
    name_map_df = name_map_df[name_map_df['Short_Name'].astype(str).str.strip() != '']
    short_to_roll = dict(zip(
        name_map_df['Short_Name'].astype(str).str.strip(),
        name_map_df['Roll_Name'].astype(str).str.strip()
    ))

    required_dates = sorted(list({v.split('_')[0] for v in target_videos}))

    full_df = pd.read_csv(full_true_path)
    full_df.columns = [str(c).strip().replace('-', '') for c in full_df.columns]
    full_df_cols = set(full_df.columns)

    available_dates = [d for d in required_dates if d in full_df_cols]
    missing_dates = [d for d in required_dates if d not in full_df_cols]

    need_create = True
    if os.path.exists(subset_true_path):
        existing_df = pd.read_csv(subset_true_path)
        existing_cols = [str(c).strip().replace('-', '') for c in existing_df.columns]
        if all(d in existing_cols for d in available_dates):
            console.print(f"[bold green]OK True Attendance for all videos already exists:[/bold green] {subset_true_path}")
            need_create = False

    if need_create:
        if missing_dates:
            console.print(f"[bold yellow]Note:[/bold yellow] {len(missing_dates)} video dates not found in filtered_true_attendance.csv (skipped during evaluation).")

        base_cols = list(full_df.columns[:2])
        subset_df = full_df[base_cols + available_dates].copy()
        subset_df.to_csv(subset_true_path, index=False)
        console.print(f"[bold green]OK Saved True Attendance to:[/bold green] {subset_true_path}")

    true_df = pd.read_csv(subset_true_path)
    roll_col = true_df.iloc[:, 0].astype(str).str.strip()
    true_df = true_df.set_index(roll_col)
    true_df.columns = true_df.columns.astype(str).str.strip().str.replace('-', '')
    for col in true_df.columns:
        true_df[col] = true_df[col].astype(str).str.strip()

    return true_df, short_to_roll


def write_live_progress(sweep_dir, pid, epoch, max_epochs, completed_epoch, total_epoch, total_tested,
                        combos_per_sec, effective_fps, elapsed_sec, eta_sec, best_acc, best_cid, best_params):
    """Writes real-time JSON and human-readable text status files."""
    pct = (completed_epoch / total_epoch * 100.0) if total_epoch > 0 else 100.0
    status_data = {
        "pid": pid,
        "epoch": epoch,
        "max_epochs": max_epochs,
        "gpu_device": 0,
        "gpu_batch_size": GPU_BATCH_SIZE,
        "completed_in_epoch": completed_epoch,
        "total_in_epoch": total_epoch,
        "total_tested_overall": total_tested,
        "epoch_percent": round(pct, 2),
        "combos_per_sec": round(combos_per_sec, 2),
        "effective_fps": round(effective_fps, 1),
        "elapsed_min": round(elapsed_sec / 60.0, 2),
        "eta_min": round(eta_sec / 60.0, 2),
        "best_accuracy": round(best_acc, 4),
        "best_combo_id": best_cid,
        "best_params": best_params,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    json_path = os.path.join(sweep_dir, "sweep_progress.json")
    txt_path  = os.path.join(sweep_dir, "live_status.txt")

    bar_len = 30
    filled = int(bar_len * pct / 100.0)
    bar_str = "#" * filled + "." * (bar_len - filled)

    txt_content = (
        f"=================================================================\n"
        f"DGX A100 GPU HYPERPARAMETER OPTIMIZER LIVE MONITOR\n"
        f"Updated  : {status_data['updated_at']}\n"
        f"=================================================================\n"
        f"PID      : {pid}  |  GPU: 0 (CUDA:0)  |  Batch: {GPU_BATCH_SIZE} combos/batch\n"
        f"Epoch    : {epoch} / {max_epochs}  [{bar_str}] {pct:6.2f}%\n"
        f"Combos   : {completed_epoch:,} / {total_epoch:,} in Epoch  ({total_tested:,} total tested)\n"
        f"Speed    : {combos_per_sec:.2f} combos/sec\n"
        f"Throughput: {effective_fps:,.0f} Effective Video Frames/sec\n"
        f"Elapsed  : {elapsed_sec/60.0:.1f} min  |  ETA: {eta_sec/60.0:.1f} min\n"
        f"Best Acc : {best_acc:.2f}% (Combo #{best_cid})\n"
        f"Best Param: {best_params}\n"
        f"=================================================================\n"
    )
    try:
        with open(json_path, "w") as f:
            json.dump(status_data, f, indent=2)
        with open(txt_path, "w") as f:
            f.write(txt_content)
    except OSError:
        pass


def param_key(p):
    """Canonical hashable key for parameter dictionary (with rounding to avoid float drift)."""
    return (
        round(float(p["CONFIDENCE_THRESHOLD"]), 4),
        round(float(p["NORM_QUALITY_THRESHOLD"]), 4),
        round(float(p["EMA_MOMENTUM"]), 4),
        int(p["FRAMES_PER_VOTE"]),
        round(float(p["SOFTMAX_TAU"]), 4),
        round(float(p["MIN_BOX_SIZE"]), 4),
        int(p["MIN_TOTAL_FRAMES"]),
        int(p["MIN_TOTAL_SAMPLES"]),
        round(float(p["MIN_SAMPLE_RATIO"]), 4),
        round(float(p["MIN_WIN_RATIO"]), 4),
    )


# ==========================================
# 4. VIDEO READER, ALIGNMENT & MODELS (STAGE 1)
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


def format_timestamp(frame_count, fps):
    fps = fps if fps > 0 else 30
    total_seconds = int(frame_count // fps)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==========================================
# 5. STAGE 1: FEATURE EXTRACTION PIPELINE
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

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024:
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


def get_cache_path(cache_dir, stem):
    p1 = os.path.join(cache_dir, f"{stem}_features.pt")
    if os.path.exists(p1) and os.path.getsize(p1) > 1024:
        return p1
    p2 = os.path.join(cache_dir, f"{stem}_raw_features.pt")
    if os.path.exists(p2) and os.path.getsize(p2) > 1024:
        return p2
    return p1


def ensure_all_features_cached(target_videos, input_dir, cache_dir, ctx):
    os.makedirs(cache_dir, exist_ok=True)
    ready_count = 0
    missing_cache = []

    for v in target_videos:
        stem = os.path.splitext(v)[0]
        cpath = get_cache_path(cache_dir, stem)
        if os.path.exists(cpath) and os.path.getsize(cpath) > 1024:
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
    console.print(f"[bold yellow]Extracting YOLO+AdaFace embeddings for {len(missing_cache)} video(s)...[/bold yellow]")

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
                raise RuntimeError(f"Failed feature extraction on {vname}: {status}")

    stop_event.set()
    dash_t.join(timeout=1.0)
    console.print(f"[bold green]OK Stage 1 feature extraction complete![/bold green]")


# ==========================================
# 6. STAGE 2: GPU-ACCELERATED BATCH EVALUATION
# ==========================================
def _load_gpu_sweep_data(target_videos, cache_dir, base_dir, short_to_roll, true_dict, device):
    """
    Loads FAISS gallery and all valid cached video track tensors onto GPU 0 once.
    """
    faiss_index_path = os.path.join(base_dir, 'face_attendance_sota_faiss.bin')
    meta_path        = os.path.join(base_dir, 'face_attendance_sota_meta.pkl')

    if not os.path.exists(faiss_index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Run 'python train_yolo_sota.py' first to build FAISS gallery and metadata!")

    console.print(f"[cyan]  Loading FAISS gallery -> GPU 0 ({device})...[/cyan]")
    index = faiss.read_index(faiss_index_path)
    index.nprobe = 20
    try:
        ref_np = index.reconstruct_n(0, index.ntotal)
    except AttributeError:
        index.make_direct_map()
        ref_np = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    ref_embs = F.normalize(torch.from_numpy(ref_np).float(), p=2, dim=1).half().to(device)
    ref_embs_t = ref_embs.t().contiguous()

    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)
    target_names = meta_data['target_names']
    y_real       = meta_data['y_real']
    roll_names   = [short_to_roll.get(n.strip(), n.strip()) for n in target_names]

    console.print(f"[cyan]  Loading feature caches for {len(target_videos)} videos -> GPU 0...[/cyan]")
    gpu_videos = []
    total_cached_frames = 0

    for v in target_videos:
        date_str = v.split('_')[0]
        stem = os.path.splitext(v)[0]
        cpath = get_cache_path(cache_dir, stem)
        if not (os.path.exists(cpath) and os.path.getsize(cpath) > 1024):
            continue

        raw_tracks = torch.load(cpath, map_location='cpu', weights_only=False)
        gpu_tracks = []
        max_vid_frame = 0

        for t_id in sorted(raw_tracks.keys()):
            td = raw_tracks[t_id]
            embs  = td['embs'].half()
            norms = td['norms'].float()
            if embs.numel() == 0:
                continue

            frames_alive = int(td.get('frames_alive', 0))
            max_vid_frame = max(max_vid_frame, frames_alive)

            gpu_tracks.append({
                'frames_alive': frames_alive,
                'box_w': td['box_w'].to(device),
                'box_h': td['box_h'].to(device),
                'embs' : embs.to(device),
                'norms': norms.to(device),
            })

        if gpu_tracks:
            total_cached_frames += max_vid_frame
            gpu_videos.append({'date_str': date_str, 'tracks': gpu_tracks})

    console.print(f"[bold green]OK Loaded {len(gpu_videos)} video caches onto GPU ({total_cached_frames:,} total frames)[/bold green]")
    return gpu_videos, ref_embs_t, target_names, y_real, roll_names, total_cached_frames


@torch.no_grad()
def _eval_track_batch(
    embs, norms, frames_alive,
    box_w, box_h,
    ref_embs_t,
    y_real_t,
    b_norm_thresh,
    b_frames_per_vote,
    b_ema_momentum,
    b_softmax_tau,
    b_min_box_size,
    b_min_total_frames,
    b_min_total_samples,
    b_min_sample_ratio,
    b_conf_thresh,
    b_min_win_ratio,
    vote_counts,
    total_preds,
    n_classes,
):
    """
    Evaluates one track across all B parameter combinations in the batch on GPU Tensor Cores.
    vote_counts : (B, n_classes) float32 -- modified in-place
    total_preds : (B,) int32             -- modified in-place
    """
    B = b_norm_thresh.shape[0]
    S = embs.shape[0]

    # Quick skip if track cannot satisfy minimum frame requirement for any combo
    if frames_alive < b_min_total_frames.min().item() or S < b_frames_per_vote.min().item():
        return vote_counts, total_preds

    box_ok = (box_w.unsqueeze(0) >= b_min_box_size.unsqueeze(1)) & \
             (box_h.unsqueeze(0) >= b_min_box_size.unsqueeze(1))  # (B, S)

    alive_ok = (frames_alive >= b_min_total_frames)  # (B,)

    for fpv_val in b_frames_per_vote.unique():
        fpv = int(fpv_val.item())
        combo_mask = (b_frames_per_vote == fpv_val)
        active = combo_mask & alive_ok
        if not active.any():
            continue

        act_idx = active.nonzero(as_tuple=True)[0]
        A = act_idx.shape[0]

        a_norm_thresh  = b_norm_thresh[act_idx]
        a_ema_momentum = b_ema_momentum[act_idx]
        a_softmax_tau  = b_softmax_tau[act_idx]
        a_box_ok       = box_ok[act_idx]

        masked_norms = norms.unsqueeze(0) * a_box_ok.float()  # (A, S)

        num_full_windows = S // fpv
        if num_full_windows == 0:
            continue

        used_S  = num_full_windows * fpv
        embs_w  = embs[:used_S].view(num_full_windows, fpv, 512)
        norms_w = masked_norms[:, :used_S].view(A, num_full_windows, fpv)

        ema_emb  = None
        a_votes  = torch.zeros(A, n_classes, device=embs.device, dtype=torch.float32)
        a_totals = torch.zeros(A, dtype=torch.int32, device=embs.device)

        a_mom_h = a_ema_momentum.unsqueeze(1).half()
        a_inv_mom_h = (1.0 - a_ema_momentum.unsqueeze(1)).half()
        a_conf_h = b_conf_thresh[act_idx].half()

        for w in range(num_full_windows):
            a_totals += 1

            w_embs  = embs_w[w]        # (fpv, 512) half
            w_norms = norms_w[:, w, :] # (A, fpv) float

            q_mask_aw = (w_norms >= a_norm_thresh.unsqueeze(1))
            any_valid = q_mask_aw.any(dim=1)  # (A,)
            if not any_valid.any():
                continue

            weights = torch.softmax(w_norms / a_softmax_tau.unsqueeze(1), dim=1).half()  # (A, fpv)

            # High-throughput Tensor Core matrix multiplication (A, fpv) @ (fpv, 512) -> (A, 512)
            e_win = F.normalize(torch.mm(weights, w_embs), p=2, dim=1)

            if ema_emb is None:
                ema_emb = e_win
            else:
                ema_emb = F.normalize(a_mom_h * ema_emb + a_inv_mom_h * e_win, p=2, dim=1)

            # Match against gallery in FP16 on Tensor Cores: (A, 512) @ (512, G) -> (A, G)
            sim = torch.mm(ema_emb, ref_embs_t)
            max_sim, max_idx = sim.max(dim=1)
            pred_class = y_real_t[max_idx]

            is_conf = (max_sim >= a_conf_h) & any_valid

            if is_conf.any():
                conf_idx = is_conf.nonzero(as_tuple=True)[0]
                conf_cls = pred_class[conf_idx]
                a_votes[conf_idx, conf_cls] += 1.0

        vote_counts[act_idx] += a_votes
        total_preds[act_idx] += a_totals

    return vote_counts, total_preds


@torch.no_grad()
def evaluate_combo_batch_gpu(
    combo_batch,
    gpu_videos,
    ref_embs_t,
    target_names,
    y_real,
    roll_names,
    true_dict,
    short_to_roll,
    device,
):
    """
    Evaluates a batch of parameter combinations across all cached videos on GPU 0.
    Returns a list of result dictionaries with accuracy and confusion matrix stats.
    """
    B = len(combo_batch)
    n_classes = len(target_names)
    y_real_t = torch.tensor(y_real, dtype=torch.int64, device=device)

    def _pt(key, dtype=torch.float32):
        return torch.tensor([p[key] for _, p in combo_batch], dtype=dtype, device=device)

    b_conf_thresh       = _pt("CONFIDENCE_THRESHOLD")
    b_norm_thresh       = _pt("NORM_QUALITY_THRESHOLD")
    b_ema_momentum      = _pt("EMA_MOMENTUM")
    b_frames_per_vote   = _pt("FRAMES_PER_VOTE", torch.int64)
    b_softmax_tau       = _pt("SOFTMAX_TAU")
    b_min_box_size      = _pt("MIN_BOX_SIZE")
    b_min_total_frames  = _pt("MIN_TOTAL_FRAMES", torch.int64)
    b_min_total_samples = _pt("MIN_TOTAL_SAMPLES", torch.int64)
    b_min_sample_ratio  = _pt("MIN_SAMPLE_RATIO")
    b_min_win_ratio     = _pt("MIN_WIN_RATIO")

    all_dates   = sorted(true_dict.keys())
    n_dates     = len(all_dates)
    n_persons   = n_classes
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    presence = torch.zeros(B, n_dates, n_persons, dtype=torch.bool, device=device)

    for vid_info in gpu_videos:
        date_str = vid_info['date_str']
        tracks   = vid_info['tracks']
        d_idx    = date_to_idx.get(date_str, -1)
        if d_idx < 0:
            continue

        for td in tracks:
            embs         = td['embs']
            norms        = td['norms']
            frames_alive = td['frames_alive']
            box_w        = td['box_w']
            box_h        = td['box_h']
            if embs.shape[0] == 0:
                continue

            vote_counts = torch.zeros(B, n_classes, device=device, dtype=torch.float32)
            total_preds = torch.zeros(B, dtype=torch.int32, device=device)

            _eval_track_batch(
                embs, norms, frames_alive,
                box_w, box_h,
                ref_embs_t, y_real_t,
                b_norm_thresh, b_frames_per_vote, b_ema_momentum,
                b_softmax_tau, b_min_box_size, b_min_total_frames,
                b_min_total_samples, b_min_sample_ratio,
                b_conf_thresh, b_min_win_ratio,
                vote_counts, total_preds,
                n_classes,
            )

            best_class_votes, best_class_idx = vote_counts.max(dim=1)
            total_valid_votes = vote_counts.sum(dim=1)

            alive_ok     = (frames_alive >= b_min_total_frames)
            samples_ok   = (total_preds.long() >= b_min_total_samples)
            sample_ratio = best_class_votes / (total_preds.float().clamp(min=1))
            sample_ok_r  = (sample_ratio >= b_min_sample_ratio)
            win_ratio    = best_class_votes / (total_valid_votes.clamp(min=1))
            win_ok       = (win_ratio >= b_min_win_ratio)
            has_votes    = (total_valid_votes > 0)

            confirmed = alive_ok & samples_ok & sample_ok_r & win_ok & has_votes
            conf_idx = confirmed.nonzero(as_tuple=True)[0]
            if conf_idx.numel() > 0:
                cls_idx = best_class_idx[conf_idx]
                presence[conf_idx, d_idx, cls_idx] = True

    results = []
    for b_i, (combo_id, params) in enumerate(combo_batch):
        tp = fp = tn = fn = 0
        total_matches = total_valid = 0

        for d_idx, date_str in enumerate(all_dates):
            if date_str not in true_dict:
                continue
            date_true_map = true_dict[date_str]

            for cls_i, roll_name in enumerate(roll_names):
                true_val = date_true_map.get(roll_name, '')
                if true_val in ('unknown', 'nan', 'none', ''):
                    continue
                total_valid += 1
                is_pres = presence[b_i, d_idx, cls_i].item()
                est_val = 'present' if is_pres else 'absent'
                if est_val == true_val:
                    total_matches += 1
                    if est_val == 'present':
                        tp += 1
                    else:
                        tn += 1
                else:
                    if est_val == 'present':
                        fp += 1
                    else:
                        fn += 1

        overall_acc = (total_matches / total_valid * 100.0) if total_valid > 0 else 0.0
        results.append({
            "Combo_ID": combo_id,
            **params,
            "Overall_Accuracy": round(overall_acc, 4),
            "Matches": total_matches,
            "Total_Valid": total_valid,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        })

    return results


# ==========================================
# 7. VISUALIZATIONS & LEADERBOARD REPORTING
# ==========================================
def generate_visualizations_and_report(final_csv_path, output_dir, param_keys):
    if not os.path.exists(final_csv_path):
        console.print("[bold red]No sweep results found to visualize![/bold red]")
        return

    df = pd.read_csv(final_csv_path)
    if df.empty:
        return

    df_sorted = df.sort_values(by=["Overall_Accuracy", "TP", "FP"], ascending=[False, False, True]).reset_index(drop=True)
    best_row = df_sorted.iloc[0]
    varied_keys = [k for k in param_keys if k in df.columns and df[k].nunique() > 1]

    leader_table = Table(title=f"[bold green]TOP 15 OPTIMAL HYPERPARAMETER COMBINATIONS (OUT OF {len(df):,} TESTED)[/bold green]")
    leader_table.add_column("Rank", justify="center", style="bold yellow")
    leader_table.add_column("Combo ID", justify="center", style="cyan")
    for k in varied_keys:
        short_h = k.replace("_THRESHOLD", "").replace("MIN_", "")
        leader_table.add_column(short_h, justify="right")
    leader_table.add_column("TP / TN / FP / FN", justify="center", style="magenta")
    leader_table.add_column("Overall Accuracy", justify="right", style="bold green")

    for rank, (_, row) in enumerate(df_sorted.head(15).iterrows(), start=1):
        row_vals = [str(rank), f"#{int(row['Combo_ID'])}"]
        for k in varied_keys:
            val = row[k]
            row_vals.append(f"{int(val)}" if float(val).is_integer() else f"{val:.3f}")
        row_vals.append(f"{int(row['TP'])} / {int(row['TN'])} / {int(row['FP'])} / {int(row['FN'])}")
        row_vals.append(f"{row['Overall_Accuracy']:.2f}%")
        leader_table.add_row(*row_vals)

    console.print("\n")
    console.print(leader_table)

    best_params_str = "\n".join([f"  * [bold cyan]{k}[/bold cyan]: {best_row[k]}" for k in param_keys if k in best_row])
    console.print(Panel(
        f"[bold green]Max Overall Accuracy Achieved: {best_row['Overall_Accuracy']:.2f}% (Combo #{int(best_row['Combo_ID'])})[/bold green]\n"
        f"[bold magenta]Confusion Breakdown:[/bold magenta] TP={int(best_row['TP'])}, TN={int(best_row['TN'])}, FP={int(best_row['FP'])}, FN={int(best_row['FN'])}\n\n"
        f"[bold white]Optimal Generalization Parameters for test_yolo_sota.py:[/bold white]\n{best_params_str}",
        title="Stage 2 Optimizer Conclusion",
        expand=False
    ))

    fig = plt.figure(figsize=(18, 12), dpi=150)
    fig.suptitle(
        f"Stage 2 Hyperparameter Optimizer Analysis ({len(df):,} Combos) -- Best Accuracy: {best_row['Overall_Accuracy']:.2f}%",
        fontsize=16, fontweight='bold', y=0.98
    )

    # 1. Top 12 Combos
    ax1 = plt.subplot(2, 2, 1)
    top_n = df_sorted.head(12).iloc[::-1]
    labels = [f"#{int(r['Combo_ID'])}" for _, r in top_n.iterrows()]
    bars = ax1.barh(labels, top_n["Overall_Accuracy"], color="#2ecc71", edgecolor="#27ae60")
    ax1.set_xlabel("Overall Accuracy (%)")
    ax1.set_title("Top 12 Parameter Combinations", fontweight="bold")
    if len(top_n) > 0:
        min_x = max(0.0, top_n["Overall_Accuracy"].min() - 3.0)
        max_x = min(100.0, top_n["Overall_Accuracy"].max() + 1.5)
        ax1.set_xlim(min_x, max_x)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va='center', fontsize=9, fontweight='bold')

    # 2. 2D Heatmap or Epoch Progression
    ax2 = plt.subplot(2, 2, 2)
    if "Epoch" in df.columns and df["Epoch"].nunique() > 1:
        epoch_agg = df.groupby("Epoch")["Overall_Accuracy"].agg(["mean", "max"]).reset_index()
        ax2.plot(epoch_agg["Epoch"], epoch_agg["max"], marker="o", color="#e74c3c", linewidth=2.5, label="Max Accuracy (%)")
        ax2.plot(epoch_agg["Epoch"], epoch_agg["mean"], marker="s", color="#3498db", linewidth=2.0, linestyle="--", label="Mean Accuracy (%)")
        ax2.set_xlabel("Epoch", fontweight="bold")
        ax2.set_ylabel("Accuracy (%)", fontweight="bold")
        ax2.set_title("Optimization Progression Across Epochs", fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    elif len(varied_keys) >= 2:
        p1, p2 = varied_keys[0], varied_keys[1]
        pivot = df.pivot_table(index=p1, columns=p2, values="Overall_Accuracy", aggfunc="max")
        im = ax2.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax2.set_xticks(range(len(pivot.columns)))
        ax2.set_xticklabels([str(c) for c in pivot.columns], rotation=30)
        ax2.set_yticks(range(len(pivot.index)))
        ax2.set_yticklabels([str(i) for i in pivot.index])
        ax2.set_xlabel(p2, fontweight="bold")
        ax2.set_ylabel(p1, fontweight="bold")
        ax2.set_title(f"Max Accuracy Heatmap: {p1} vs {p2}", fontweight="bold")
        plt.colorbar(im, ax=ax2, label="Max Accuracy (%)")
    else:
        ax2.text(0.5, 0.5, "Single Parameter Swept", ha="center", va="center")

    # 3. Sensitivity across active variables
    ax3 = plt.subplot(2, 1, 2)
    if varied_keys:
        x_positions, x_labels, mean_vals, max_vals, group_boundaries = [], [], [], [], []
        pos = 0
        for k in varied_keys:
            grouped = df.groupby(k)["Overall_Accuracy"].agg(["mean", "max"]).reset_index()
            for _, grow in grouped.iterrows():
                x_positions.append(pos)
                val_str = f"{int(grow[k])}" if float(grow[k]).is_integer() else f"{grow[k]:.3f}"
                x_labels.append(f"{k.replace('_THRESHOLD', '').replace('MIN_', '')}\n={val_str}")
                mean_vals.append(grow["mean"])
                max_vals.append(grow["max"])
                pos += 1
            group_boundaries.append(pos - 0.5)

        width = 0.38
        x_arr = np.array(x_positions)
        ax3.bar(x_arr - width/2, mean_vals, width=width, label="Mean Accuracy (%)", color="#3498db")
        ax3.bar(x_arr + width/2, max_vals, width=width, label="Max Accuracy (%)", color="#e67e22")
        for b in group_boundaries[:-1]:
            ax3.axvline(b, color="gray", linestyle="--", alpha=0.5)
        ax3.set_xticks(x_arr)
        ax3.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_title("Active Hyperparameter Sensitivity (Mean vs. Max Accuracy across tested slices)", fontweight="bold")
        if mean_vals:
            ax3.set_ylim(max(0.0, min(mean_vals) - 3.0), min(100.0, max(max_vals) + 1.5))
        ax3.legend(loc="lower right")
        ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plot_path = os.path.join(output_dir, "sweep_visualization.png")
    plt.savefig(plot_path)
    plt.close(fig)
    console.print(f"[bold green]OK Saved Sweep Visualization Chart to:[/bold green] {plot_path}")


# ==========================================
# 8. MAIN ENTRY POINT (STAGE 1 & STAGE 2 OPTIMIZER)
# ==========================================
if __name__ == '__main__':
    mp.freeze_support()
    ctx = mp.get_context('spawn')
    main_pid = os.getpid()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    console.print(Panel(
        f"[bold yellow]Main Process ID (PID) for Excel Sheet:[/bold yellow] [bold white]{main_pid}[/bold white]\n"
        f"[bold cyan]Hardware Isolation:[/bold cyan] CUDA_VISIBLE_DEVICES='0' ({device})\n"
        f"[bold green]Shared Feature Cache:[/bold green] {FEATURE_CACHE_DIR}\n"
        f"[bold magenta]Sweep Output Directory:[/bold magenta] {SWEEP_OUTPUT_DIR}",
        title="DGX A100 GPU Hyperparameter Optimizer (Copy PID to SPCLAB-DGX_Utilization.xlsx)",
        expand=False
    ))

    # 1. Discover all videos from VIDEOS/
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov')
    target_videos = sorted([
        f for f in os.listdir(INPUT_VIDEOS_DIR)
        if f.lower().endswith(video_extensions) and not f.startswith('.')
    ])
    console.print(f"[bold cyan]Discovered {len(target_videos)} videos in '{INPUT_VIDEOS_DIR}'.[/bold cyan]\n")

    if not target_videos:
        console.print("[bold red]No videos found in VIDEOS folder![/bold red]")
        sys.exit(0)

    # 2. Extract or load ground truth for all videos
    true_df, short_to_roll = prepare_subset_true_attendance(target_videos, RESULTS_BASE_DIR, SWEEP_OUTPUT_DIR)
    true_dict = {}
    for col in true_df.columns:
        col_clean = str(col).strip()
        if col_clean.isdigit():
            true_dict[col_clean] = {
                str(idx).strip(): str(val).strip().lower()
                for idx, val in true_df[col].items()
            }

    # 3. Build or verify Stage 1 feature cache
    ensure_all_features_cached(target_videos, INPUT_VIDEOS_DIR, FEATURE_CACHE_DIR, ctx)

    # 4. Load all cached video features and FAISS gallery directly onto GPU 0 once
    console.print("\n[bold yellow]Loading video caches and gallery onto GPU 0...[/bold yellow]")
    gpu_videos, ref_embs_t, target_names, y_real, roll_names, total_cached_frames = _load_gpu_sweep_data(
        target_videos, FEATURE_CACHE_DIR, BASE_DIR, short_to_roll, true_dict, device
    )

    if not gpu_videos:
        console.print("[bold red]No valid video caches loaded onto GPU! Exiting.[/bold red]")
        sys.exit(0)

    # 5. Initialize Progressive Epoch Optimizer
    final_accuracy_csv = os.path.join(SWEEP_OUTPUT_DIR, "sweep_final_accuracy.csv")

    evaluated_cache = {}
    max_combo_id = 0
    best_acc_so_far = -1.0
    best_tp_so_far  = -1
    best_fp_so_far  = 999999
    best_cid_so_far = None
    best_params_so_far = None

    if os.path.exists(final_accuracy_csv):
        try:
            existing_df = pd.read_csv(final_accuracy_csv)
            if not existing_df.empty and "Combo_ID" in existing_df.columns:
                for _, row in existing_df.iterrows():
                    cid = int(row["Combo_ID"])
                    max_combo_id = max(max_combo_id, cid)
                    p_dict = {k: row[k] for k in ALL_PARAM_KEYS if k in row}
                    if len(p_dict) == len(ALL_PARAM_KEYS):
                        pkey = param_key(p_dict)
                        evaluated_cache[pkey] = row.to_dict()
                        acc = float(row.get("Overall_Accuracy", 0.0))
                        tp  = int(row.get("TP", 0))
                        fp  = int(row.get("FP", 0))
                        if (acc > best_acc_so_far) or (acc == best_acc_so_far and tp > best_tp_so_far) or (acc == best_acc_so_far and tp == best_tp_so_far and fp < best_fp_so_far):
                            best_acc_so_far = acc
                            best_tp_so_far  = tp
                            best_fp_so_far  = fp
                            best_cid_so_far = cid
                            best_params_so_far = p_dict
                console.print(f"[bold cyan]Found {len(evaluated_cache)} already-evaluated combinations in {final_accuracy_csv}. Resuming...[/bold cyan]")
        except Exception as e:
            console.print(f"[bold yellow]Warning reading existing sweep CSV: {e}[/bold yellow]")

    # Initial baseline center point (Epoch 0 configuration)
    if best_params_so_far is not None:
        theta_star = dict(best_params_so_far)
    else:
        theta_star = {
            "CONFIDENCE_THRESHOLD":   0.40,
            "NORM_QUALITY_THRESHOLD": NORM_QUALITY_THRESHOLD,
            "EMA_MOMENTUM":           0.85,
            "FRAMES_PER_VOTE":        3,
            "SOFTMAX_TAU":            SOFTMAX_TAU,
            "MIN_BOX_SIZE":           60.0,
            "MIN_TOTAL_FRAMES":       MIN_TOTAL_FRAMES,
            "MIN_TOTAL_SAMPLES":      20,
            "MIN_SAMPLE_RATIO":       0.40,
            "MIN_WIN_RATIO":          0.45,
        }

    current_steps = {k: v["step"] for k, v in ACTIVE_VARS.items()}
    t_opt_start = time.perf_counter()

    console.print(f"\n[bold white]Starting Epoch-Based Coordinate Ascent Optimizer (MAX_EPOCHS={MAX_EPOCHS}):[/bold white]")
    console.print(f"  * GPU: CUDA:0  |  Batch Size: {GPU_BATCH_SIZE} combos/batch")
    console.print(f"  * Cached Videos on GPU: {len(gpu_videos)}  ({total_cached_frames:,} total frames)\n")

    # Outer Epoch Loop
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        console.print(f"\n[bold cyan]=== Epoch {epoch} / {MAX_EPOCHS} ===[/bold cyan]")
        console.print(f"[dim]Current center theta*: {theta_star}[/dim]")
        console.print(f"[dim]Current steps: {current_steps}[/dim]")

        # 1. Generate candidate neighborhood around theta*
        candidates = []
        candidates.append(dict(theta_star))

        # 1D coordinate steps (theta_i +- Delta_i, theta_i +- 2*Delta_i)
        for k, cfg in ACTIVE_VARS.items():
            delta = current_steps[k]
            for mult in [-2, -1, 1, 2]:
                val = theta_star[k] + mult * delta
                val = max(cfg["min"], min(cfg["max"], val))
                cand = dict(theta_star)
                cand[k] = int(round(val)) if cfg["is_int"] else round(val, 4)
                candidates.append(cand)

        # 2D pairwise interaction steps for TOP4_VARS
        for k1, k2 in itertools.combinations(TOP4_VARS, 2):
            cfg1, cfg2 = ACTIVE_VARS[k1], ACTIVE_VARS[k2]
            d1, d2 = current_steps[k1], current_steps[k2]
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v1 = max(cfg1["min"], min(cfg1["max"], theta_star[k1] + s1 * d1))
                    v2 = max(cfg2["min"], min(cfg2["max"], theta_star[k2] + s2 * d2))
                    cand = dict(theta_star)
                    cand[k1] = int(round(v1)) if cfg1["is_int"] else round(v1, 4)
                    cand[k2] = int(round(v2)) if cfg2["is_int"] else round(v2, 4)
                    candidates.append(cand)

        # Deduplicate candidates
        unique_epoch_candidates = []
        seen_keys = set()
        for c in candidates:
            pk = param_key(c)
            if pk not in seen_keys:
                seen_keys.add(pk)
                unique_epoch_candidates.append(c)

        # Filter against evaluated_cache
        pending_candidates = []
        for c in unique_epoch_candidates:
            pk = param_key(c)
            if pk in evaluated_cache:
                row = evaluated_cache[pk]
                c_acc = float(row.get("Overall_Accuracy", 0.0))
                c_tp  = int(row.get("TP", 0))
                c_fp  = int(row.get("FP", 0))
                c_id  = int(row["Combo_ID"])
                if (c_acc > best_acc_so_far) or (c_acc == best_acc_so_far and c_tp > best_tp_so_far) or (c_acc == best_acc_so_far and c_tp == best_tp_so_far and c_fp < best_fp_so_far):
                    best_acc_so_far = c_acc
                    best_tp_so_far  = c_tp
                    best_fp_so_far  = c_fp
                    best_cid_so_far = c_id
                    best_params_so_far = {k: row[k] for k in ALL_PARAM_KEYS}
            else:
                pending_candidates.append(c)

        total_epoch_combos = len(unique_epoch_candidates)
        already_evaluated_in_epoch = total_epoch_combos - len(pending_candidates)

        console.print(
            f"Epoch {epoch}: {total_epoch_combos} candidate combinations "
            f"({already_evaluated_in_epoch} cached, {len(pending_candidates)} pending evaluation)"
        )

        improved_this_epoch = False

        if pending_candidates:
            # Set up Rich Progress Bar
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=32),
                MofNCompleteColumn(),
                TextColumn("[bold yellow]{task.percentage:>5.1f}%"),
                TimeRemainingColumn(),
                TextColumn("[bold green]{task.fields[speed]}"),
                TextColumn("[bold magenta]{task.fields[best]}"),
                console=console,
                refresh_per_second=5
            )

            task_id = progress.add_task(
                f"Epoch {epoch}/{MAX_EPOCHS}",
                total=total_epoch_combos,
                completed=already_evaluated_in_epoch,
                speed="Starting GPU batch...",
                best=f"Best: {best_acc_so_far:.2f}%" if best_cid_so_far else "Best: --"
            )

            # Assign sequential combo IDs
            pending_with_ids = []
            for cand in pending_candidates:
                max_combo_id += 1
                pending_with_ids.append((max_combo_id, cand))

            num_batches = (len(pending_with_ids) + GPU_BATCH_SIZE - 1) // GPU_BATCH_SIZE
            completed_in_epoch = already_evaluated_in_epoch
            epoch_eval_count = 0

            with progress:
                for b_i in range(num_batches):
                    b_start = b_i * GPU_BATCH_SIZE
                    b_end   = min(b_start + GPU_BATCH_SIZE, len(pending_with_ids))
                    combo_batch = pending_with_ids[b_start:b_end]

                    b_t0 = time.perf_counter()
                    try:
                        batch_results = evaluate_combo_batch_gpu(
                            combo_batch, gpu_videos, ref_embs_t, target_names,
                            y_real, roll_names, true_dict, short_to_roll, device
                        )
                    except torch.cuda.OutOfMemoryError:
                        console.print("[bold red]OOM encountered. Retrying in halves...[/bold red]")
                        batch_results = []
                        half = max(1, len(combo_batch) // 2)
                        for s_start in range(0, len(combo_batch), half):
                            sub = combo_batch[s_start:s_start + half]
                            torch.cuda.empty_cache()
                            sub_res = evaluate_combo_batch_gpu(
                                sub, gpu_videos, ref_embs_t, target_names,
                                y_real, roll_names, true_dict, short_to_roll, device
                            )
                            batch_results.extend(sub_res)

                    b_t1 = time.perf_counter()

                    # Record results and update CSV
                    buffer_rows = []
                    for res in batch_results:
                        res["Epoch"] = epoch
                        buffer_rows.append(res)
                        evaluated_cache[param_key(res)] = res
                        completed_in_epoch += 1
                        epoch_eval_count += 1

                        acc = res["Overall_Accuracy"]
                        tp  = res["TP"]
                        fp  = res["FP"]

                        if (acc > best_acc_so_far) or (acc == best_acc_so_far and tp > best_tp_so_far) or (acc == best_acc_so_far and tp == best_tp_so_far and fp < best_fp_so_far):
                            best_acc_so_far    = acc
                            best_tp_so_far     = tp
                            best_fp_so_far     = fp
                            best_cid_so_far    = res["Combo_ID"]
                            best_params_so_far = {k: res[k] for k in ALL_PARAM_KEYS}
                            improved_this_epoch = True

                    df_chunk = pd.DataFrame(buffer_rows)
                    write_header = not os.path.exists(final_accuracy_csv)
                    df_chunk.to_csv(final_accuracy_csv, mode='a', header=write_header, index=False)

                    # Telemetry calculations
                    elapsed_total = time.perf_counter() - t_opt_start
                    elapsed_batch = b_t1 - b_t0
                    combos_sec = len(combo_batch) / elapsed_batch if elapsed_batch > 0 else 0.0
                    eff_fps = combos_sec * total_cached_frames
                    vids_sec = combos_sec * len(gpu_videos)

                    remaining_combos_epoch = total_epoch_combos - completed_in_epoch
                    eta_sec = remaining_combos_epoch / combos_sec if combos_sec > 0 else 0.0

                    progress.update(
                        task_id,
                        completed=completed_in_epoch,
                        speed=f"{combos_sec:.1f} c/s | {eff_fps:,.0f} FPS",
                        best=f"Best: {best_acc_so_far:.2f}% (#{best_cid_so_far})"
                    )

                    write_live_progress(
                        SWEEP_OUTPUT_DIR, main_pid, epoch, MAX_EPOCHS,
                        completed_in_epoch, total_epoch_combos, len(evaluated_cache),
                        combos_sec, eff_fps, elapsed_total, eta_sec,
                        best_acc_so_far, best_cid_so_far, best_params_so_far
                    )

        # Update center point theta* to the best combination found
        if best_params_so_far is not None:
            theta_star = dict(best_params_so_far)

        # Step size decay: Delta <- 0.5 * Delta
        for k, cfg in ACTIVE_VARS.items():
            if cfg["is_int"]:
                current_steps[k] = max(1, int(round(current_steps[k] * 0.5)))
            else:
                current_steps[k] = max(0.01, round(current_steps[k] * 0.5, 4))

        # Check early stopping convergence
        all_at_minimum = all(
            (current_steps[k] <= 1 if cfg["is_int"] else current_steps[k] <= 0.01)
            for k, cfg in ACTIVE_VARS.items()
        )
        if all_at_minimum and not improved_this_epoch:
            console.print("[bold green]Local convergence reached at minimum step precision. Finishing optimizer early.[/bold green]")
            break

    # 6. Generate Final Visualization & Rich Leaderboard Report
    console.print("\n[bold yellow]Generating final report and visualization...[/bold yellow]")
    generate_visualizations_and_report(final_accuracy_csv, SWEEP_OUTPUT_DIR, ALL_PARAM_KEYS)
    console.print("\n[bold green]Optimizer run completed successfully![/bold green]")

# pyright: reportMissingImports=false
# sweep_yolo_sota.py – DGX A100 GPU-Accelerated Hyperparameter Optimizer (Stage 2)
# ==============================================================================
# Upgrades:
#   1. Multi-GPU acceleration across ACTIVE_GPUS with clean single-line Rich progress.
#      - No multi-line terminal flooding: GPU % done and ETA update inline on the epoch bar.
#   2. Daily Detection Count Objective (matching process_daily_attendance.py):
#      - Combines all protocols/videos of the same day by summing confirmed track detections
#        per (Date, Student) and dynamically finding the optimal daily detection count
#        threshold (1..20) that maximizes Daily Attendance Accuracy.
#      - Retains Session_Accuracy and per-protocol accuracies (5m, 10m, 15m, 20m, 30m)
#        as secondary diagnostics.
#   3. Broader 10-Variable Search Space & Extended 20-Epoch Schedule:
#      - All 10 hyperparameters unlocked and swept across wider bounds.
#      - Smooth 0.65x coordinate step decay across 20 epochs with 1D & 2D steps.
# ==============================================================================

import os
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
# All 10 parameters unlocked with broader search ranges
ACTIVE_VARS = {
    "CONFIDENCE_THRESHOLD":   {"center": 0.38, "min": 0.22, "max": 0.65, "step": 0.05, "is_int": False},
    "NORM_QUALITY_THRESHOLD": {"center": 7.0,  "min": 3.0,  "max": 14.0, "step": 1.5,  "is_int": False},
    "EMA_MOMENTUM":           {"center": 0.84, "min": 0.35, "max": 0.98, "step": 0.08, "is_int": False},
    "FRAMES_PER_VOTE":        {"center": 2,    "min": 1,    "max": 12,   "step": 2,    "is_int": True},
    "SOFTMAX_TAU":            {"center": 5.0,  "min": 1.5,  "max": 12.0, "step": 1.5,  "is_int": False},
    "MIN_BOX_SIZE":           {"center": 50.0, "min": 45.0, "max": 95.0, "step": 8.0,  "is_int": False},
    "MIN_TOTAL_FRAMES":       {"center": 30,   "min": 10,   "max": 80,   "step": 10,   "is_int": True},
    "MIN_TOTAL_SAMPLES":      {"center": 16,   "min": 4,    "max": 55,   "step": 6,    "is_int": True},
    "MIN_SAMPLE_RATIO":       {"center": 0.40, "min": 0.10, "max": 0.75, "step": 0.08, "is_int": False},
    "MIN_WIN_RATIO":          {"center": 0.39, "min": 0.20, "max": 0.80, "step": 0.07, "is_int": False},
}

TOP6_VARS = [
    "CONFIDENCE_THRESHOLD",
    "MIN_WIN_RATIO",
    "MIN_SAMPLE_RATIO",
    "MIN_TOTAL_SAMPLES",
    "FRAMES_PER_VOTE",
    "EMA_MOMENTUM",
]

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

MAX_EPOCHS = 20
STEP_DECAY = 0.65
MAX_DAILY_THRESHOLD = 20
MIN_FRAMES_FLOOR = int(ACTIVE_VARS["MIN_TOTAL_FRAMES"]["min"])
BASE_MIN_BOX_SIZE = 45.0
FRAME_SKIP = 1
MAX_STAGE1_WORKERS = 4
GPU_BATCH_SIZE = 512
AT_BUDGET = 600_000
SIM_ROW_CHUNK_ELEMS = 1 << 27
ACTIVE_GPUS = [6, 7]

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
def prepare_subset_true_attendance(target_videos, results_base_dir, sweep_dir, base_dir=None):
    """
    Extracts ground truth attendance for all unique video dates across all discovered videos.
    Saves true_attendance_all_videos.csv in ATTENDENCE RESULTS/sweep.
    Also loads protocol_mapping.csv and returns base_to_protocol dictionary.
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

    protocol_path = os.path.join(results_base_dir, "protocol_mapping.csv")
    if not os.path.exists(protocol_path):
        protocol_path = os.path.join(base_dir or BASE_DIR, "protocol_mapping.csv")

    base_to_protocol = {}
    if os.path.exists(protocol_path):
        proto_df = pd.read_csv(protocol_path).dropna(subset=['Base_Timestamp', 'Protocol_Length'])
        base_to_protocol = dict(zip(
            proto_df['Base_Timestamp'].astype(str).str.strip(),
            proto_df['Protocol_Length'].astype(str).str.strip()
        ))
        console.print(f"[bold green]OK Loaded Protocol Mapping:[/bold green] {len(base_to_protocol)} sessions from {protocol_path}")
    else:
        console.print(f"[bold yellow]Warning:[/bold yellow] protocol_mapping.csv not found at {protocol_path}")

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

    return true_df, short_to_roll, base_to_protocol


def write_live_progress(sweep_dir, pid, epoch, max_epochs, completed_epoch, total_epoch, total_tested,
                        combos_per_sec, effective_fps, elapsed_sec, eta_sec, best_acc, best_cid,
                        best_thresh, best_params):
    """Writes real-time JSON and human-readable text status files."""
    pct = (completed_epoch / total_epoch * 100.0) if total_epoch > 0 else 100.0
    status_data = {
        "pid": pid,
        "epoch": epoch,
        "max_epochs": max_epochs,
        "active_gpus": ACTIVE_GPUS,
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
        "best_daily_threshold": int(best_thresh) if best_thresh is not None else 1,
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
        f"DGX A100 MULTI-GPU HYPERPARAMETER OPTIMIZER LIVE MONITOR\n"
        f"Updated  : {status_data['updated_at']}\n"
        f"=================================================================\n"
        f"PID      : {pid}  |  GPUs: {ACTIVE_GPUS}  |  Batch: {GPU_BATCH_SIZE} combos/batch per GPU\n"
        f"Epoch    : {epoch} / {max_epochs}  [{bar_str}] {pct:6.2f}%\n"
        f"Combos   : {completed_epoch:,} / {total_epoch:,} in Epoch  ({total_tested:,} total tested)\n"
        f"Speed    : {combos_per_sec:.2f} combos/sec\n"
        f"Throughput: {effective_fps:,.0f} Effective Video Frames/sec\n"
        f"Elapsed  : {elapsed_sec/60.0:.1f} min  |  ETA: {eta_sec/60.0:.1f} min\n"
        f"Best Daily Acc: {best_acc:.2f}% (Combo #{best_cid}, Daily Count >= {status_data['best_daily_threshold']})\n"
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
# 6. STAGE 2: GPU-ACCELERATED BATCH EVALUATION (DAILY + SESSION)
# ==========================================
def _load_gpu_sweep_data(target_videos, cache_dir, base_dir, short_to_roll, true_dict, base_to_protocol=None):
    """
    Loads FAISS gallery and all valid cached video track tensors into CPU memory.
    Active GPU workers will load/transfer tensors directly to their respective GPU device.
    """
    if base_to_protocol is None:
        base_to_protocol = {}

    faiss_index_path = os.path.join(base_dir, 'face_attendance_sota_faiss.bin')
    meta_path        = os.path.join(base_dir, 'face_attendance_sota_meta.pkl')

    if not os.path.exists(faiss_index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Run 'python train_yolo_sota.py' first to build FAISS gallery and metadata!")

    console.print(f"[cyan]  Loading FAISS gallery -> CPU memory...[/cyan]")
    index = faiss.read_index(faiss_index_path)
    index.nprobe = 20
    try:
        ref_np = index.reconstruct_n(0, index.ntotal)
    except AttributeError:
        index.make_direct_map()
        ref_np = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    ref_embs = F.normalize(torch.from_numpy(ref_np).float(), p=2, dim=1).half()

    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)
    target_names = meta_data['target_names']
    y_real       = meta_data['y_real']
    roll_names   = [short_to_roll.get(n.strip(), n.strip()) for n in target_names]

    console.print(f"[cyan]  Loading feature caches for {len(target_videos)} videos -> CPU memory...[/cyan]")
    cpu_videos = []
    total_cached_frames = 0

    for v in target_videos:
        parts = v.split('_')
        date_str = parts[0]
        base_ts = f"{parts[0]}_{parts[1]}" if len(parts) > 1 else parts[0]
        protocol = base_to_protocol.get(base_ts, 'Unknown')

        stem = os.path.splitext(v)[0]
        cpath = get_cache_path(cache_dir, stem)
        if not (os.path.exists(cpath) and os.path.getsize(cpath) > 1024):
            continue

        raw_tracks = torch.load(cpath, map_location='cpu', weights_only=False)
        video_tracks = []
        max_vid_frame = 0

        for t_id in sorted(raw_tracks.keys()):
            td = raw_tracks[t_id]
            embs  = td['embs'].half()
            norms = td['norms'].float()
            if embs.numel() == 0:
                continue

            frames_alive = int(td.get('frames_alive', 0))
            max_vid_frame = max(max_vid_frame, frames_alive)

            video_tracks.append({
                'frames_alive': frames_alive,
                'box_w': td['box_w'].float(),
                'box_h': td['box_h'].float(),
                'embs' : embs,
                'norms': norms,
            })

        if video_tracks:
            total_cached_frames += max_vid_frame
            cpu_videos.append({
                'base_ts': base_ts,
                'date_str': date_str,
                'protocol': protocol,
                'tracks': video_tracks,
            })

    unique_sessions = len({v['base_ts'] for v in cpu_videos})
    unique_dates    = len({v['date_str'] for v in cpu_videos if v['date_str'] in true_dict})
    console.print(
        f"[bold green]OK Loaded {len(cpu_videos)} video caches across {unique_sessions} sessions "
        f"({unique_dates} evaluated dates, {total_cached_frames:,} total frames)[/bold green]"
    )
    return cpu_videos, ref_embs, target_names, y_real, roll_names, total_cached_frames


@torch.no_grad()
def _fpv_info(bank, fpv):
    """Per-FRAMES_PER_VOTE track ordering (cached). Tracks are sorted by window count (desc) so that at
    window step w the still-active tracks are always a *prefix* -> no masking / no Python per-track loop."""
    cache = bank['fpv_cache']
    if fpv in cache:
        return cache[fpv]
    lens = bank['lens_np']
    ok = np.nonzero(lens >= fpv)[0]
    if ok.size == 0:
        cache[fpv] = None
        return None
    W = lens[ok] // fpv
    order = np.argsort(-W, kind='stable')
    sel = ok[order]
    W_s = W[order]
    Wmax = int(W_s[0])
    asc = np.ascontiguousarray(W_s[::-1])
    n_gt = len(W_s) - np.searchsorted(asc, np.arange(Wmax), side='right')
    dev = bank['device']
    tidx = torch.from_numpy(sel).to(dev)
    info = {
        'starts': bank['starts_t'][tidx],
        'W_t': torch.from_numpy(W_s).to(dev),
        'W_np': W_s,
        'alive': bank['alive_t'][tidx],
        'sess': bank['sess_t'][tidx],
        'date': bank['date_t'][tidx],
        'n_gt': n_gt,
        'Wmax': Wmax,
    }
    cache[fpv] = info
    return info


def _chunked_max_sim(q, ref_t):
    """q: (R,512), ref_t: (512,G)  ->  (max_sim (R,), argmax (R,)) without materialising a giant (R,G)."""
    R = q.shape[0]
    G = ref_t.shape[1]
    rows_per = max(1, SIM_ROW_CHUNK_ELEMS // max(G, 1))
    if R <= rows_per:
        return torch.mm(q, ref_t).max(dim=1)
    ms, ix = [], []
    for r0 in range(0, R, rows_per):
        m, i = torch.mm(q[r0:r0 + rows_per], ref_t).max(dim=1)
        ms.append(m)
        ix.append(i)
    return torch.cat(ms), torch.cat(ix)


@torch.no_grad()
def _eval_fpv_group(bank, ref_embs_t, fpv, grp, P, presence, daily_counts, n_classes, at_budget, live):
    """
    Evaluates every combo in `grp` (all sharing FRAMES_PER_VOTE=fpv) against ALL tracks at once.
    Accumulates:
      - `presence` (B, n_sessions, n_classes): binary session presence for per-protocol diagnostics
      - `daily_counts` (B, n_dates, n_classes): total confirmed track detections summed across all
        protocols & cameras of the same date (matching process_daily_attendance.py).
    """
    info = _fpv_info(bank, fpv)
    if info is None:
        return
    dev = bank['device']
    cdt = bank['dtype']

    min_samples = int(P['MIN_TOTAL_SAMPLES'][grp].min().item())
    Te = int((info['W_np'] >= min_samples).sum())
    if Te == 0:
        return

    starts = info['starts'][:Te]
    n_gt = info['n_gt']
    Wmax = info['Wmax']
    embs_flat, norms_flat = bank['embs'], bank['norms']
    bw_flat, bh_flat = bank['box_w'], bank['box_h']
    y_real_t = bank['y_real_t']
    ar_f = torch.arange(fpv, device=dev)

    n_dates = daily_counts.shape[1]
    daily_counts_flat = daily_counts.view(-1)

    A_chunk = max(1, at_budget // Te)
    for a0 in range(0, grp.numel(), A_chunk):
        sub = grp[a0:a0 + A_chunk]
        A = sub.numel()

        mbs   = P['MIN_BOX_SIZE'][sub].view(1, A, 1)
        nth   = P['NORM_QUALITY_THRESHOLD'][sub].view(1, A, 1)
        tau   = P['SOFTMAX_TAU'][sub].view(1, A, 1)
        mom   = P['EMA_MOMENTUM'][sub].to(cdt).view(1, A, 1)
        inv   = (1.0 - P['EMA_MOMENTUM'][sub]).to(cdt).view(1, A, 1)
        conf  = P['CONFIDENCE_THRESHOLD'][sub].to(cdt).view(1, A)

        ema    = torch.zeros(Te, A, 512, dtype=cdt, device=dev)
        inited = torch.zeros(Te, A, dtype=torch.bool, device=dev)
        votes  = torch.zeros(Te, A, n_classes, dtype=torch.int32, device=dev)
        votes_flat = votes.view(Te * A, n_classes)
        row_ids = torch.arange(Te * A, device=dev)

        for w in range(Wmax):
            n = int(min(n_gt[w], Te))
            if n == 0:
                break
            live['w'] = w
            live['tracks'] = n
            live['rows_done'] += n * A

            idx = starts[:n].unsqueeze(1) + (w * fpv) + ar_f.unsqueeze(0)       # (n, fpv)
            w_embs = embs_flat[idx]                                              # (n, fpv, 512)
            nrm    = norms_flat[idx]                                             # (n, fpv)
            bw     = bw_flat[idx].unsqueeze(1)                                   # (n, 1, fpv)
            bh     = bh_flat[idx].unsqueeze(1)

            box_ok = (bw >= mbs) & (bh >= mbs)                                   # (n, A, fpv)
            masked = nrm.unsqueeze(1) * box_ok                                   # (n, A, fpv)
            valid  = (masked >= nth).any(dim=2)                                  # (n, A)

            weights = torch.softmax(masked / tau, dim=2).to(cdt)                 # (n, A, fpv)
            e_win = F.normalize(torch.bmm(weights, w_embs), p=2, dim=2)          # (n, A, 512)

            prev = ema[:n]
            blended = F.normalize(mom * prev + inv * e_win, p=2, dim=2)
            new = torch.where(inited[:n].unsqueeze(2), blended, e_win)
            ema[:n] = torch.where(valid.unsqueeze(2), new, prev)
            inited[:n] |= valid

            flat = ema[:n].reshape(n * A, 512)
            max_sim, max_idx = _chunked_max_sim(flat, ref_embs_t)
            cls = y_real_t[max_idx]
            is_conf = (max_sim.view(n, A) >= conf) & valid
            votes_flat[row_ids[:n * A], cls] += is_conf.reshape(-1).to(torch.int32)

        # ---- confirm tracks ----
        best_votes, best_cls = votes.max(dim=2)                                  # (Te, A)
        total_valid = votes.sum(dim=2)                                           # (Te, A)
        W_f = info['W_t'][:Te].unsqueeze(1)                                      # (Te, 1)
        alive = info['alive'][:Te].unsqueeze(1)

        alive_ok   = alive >= P['MIN_TOTAL_FRAMES'][sub].view(1, A)
        samples_ok = W_f >= P['MIN_TOTAL_SAMPLES'][sub].view(1, A)
        sample_ratio = best_votes.float() / W_f.float().clamp(min=1)
        win_ratio    = best_votes.float() / total_valid.float().clamp(min=1)
        confirmed = (alive_ok & samples_ok & (total_valid > 0)
                     & (sample_ratio >= P['MIN_SAMPLE_RATIO'][sub].view(1, A))
                     & (win_ratio >= P['MIN_WIN_RATIO'][sub].view(1, A)))
        t_i, a_i = confirmed.nonzero(as_tuple=True)
        if t_i.numel() > 0:
            combo_idx = sub[a_i]
            sess_idx  = info['sess'][:Te][t_i]
            date_idx  = info['date'][:Te][t_i]
            cls_idx   = best_cls[t_i, a_i]

            # 1) Session-level binary presence (for per-protocol breakdown)
            presence[combo_idx, sess_idx, cls_idx] = True

            # 2) Daily total detection count across all protocols & cameras of the same day
            flat_daily_idx = (combo_idx * n_dates + date_idx) * n_classes + cls_idx
            ones = torch.ones(flat_daily_idx.numel(), dtype=torch.int32, device=dev)
            daily_counts_flat.index_add_(0, flat_daily_idx, ones)

        del ema, inited, votes, votes_flat, row_ids, best_votes, best_cls, total_valid, confirmed


@torch.no_grad()
def build_track_bank(cpu_videos, true_dict, roll_names, y_real, device, min_frames_floor, dtype=torch.float16):
    """
    Flattens EVERY track of EVERY video into contiguous GPU tensors (once at startup)
    and pre-builds both the Daily ground-truth masks and Session/Protocol ground-truth masks.
    """
    all_sessions = sorted({v['base_ts'] for v in cpu_videos if v['date_str'] in true_dict})
    session_to_idx = {s: i for i, s in enumerate(all_sessions)}

    all_dates = sorted({v['date_str'] for v in cpu_videos if v['date_str'] in true_dict})
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    n_sessions = len(all_sessions)
    n_dates    = len(all_dates)
    n_persons  = len(roll_names)

    session_protocol = {}
    embs_l, norms_l, bw_l, bh_l = [], [], [], []
    starts, lens, alive, sess, dates = [], [], [], [], []
    cursor = 0
    for v in cpu_videos:
        s_idx = session_to_idx.get(v['base_ts'], -1)
        d_idx = date_to_idx.get(v['date_str'], -1)
        if s_idx < 0 or d_idx < 0:
            continue
        session_protocol.setdefault(v['base_ts'], v.get('protocol', 'Unknown'))
        for td in v['tracks']:
            S = int(td['embs'].shape[0])
            fa = int(td['frames_alive'])
            if S == 0 or fa < min_frames_floor:
                continue
            embs_l.append(td['embs']); norms_l.append(td['norms'])
            bw_l.append(td['box_w']);  bh_l.append(td['box_h'])
            starts.append(cursor); lens.append(S); alive.append(fa)
            sess.append(s_idx); dates.append(d_idx)
            cursor += S

    if embs_l:
        embs  = torch.cat(embs_l).to(dtype).to(device)
        norms = torch.cat(norms_l).float().to(device)
        bw    = torch.cat(bw_l).float().to(device)
        bh    = torch.cat(bh_l).float().to(device)
    else:
        embs  = torch.zeros(0, 512, dtype=dtype, device=device)
        norms = bw = bh = torch.zeros(0, dtype=torch.float32, device=device)

    # 1) Session-level ground truth masks
    gt_present = np.zeros((n_sessions, n_persons), dtype=bool)
    gt_valid   = np.zeros((n_sessions, n_persons), dtype=bool)
    for s_idx, base_ts in enumerate(all_sessions):
        date_map = true_dict.get(base_ts.split('_')[0], {})
        for cls_i, roll in enumerate(roll_names):
            tv = date_map.get(roll, '')
            if tv in ('unknown', 'nan', 'none', ''):
                continue
            gt_valid[s_idx, cls_i] = True
            if tv == 'present':
                gt_present[s_idx, cls_i] = True

    # 2) Daily ground truth masks (all protocols of the same day combined)
    gt_daily_present = np.zeros((n_dates, n_persons), dtype=bool)
    gt_daily_valid   = np.zeros((n_dates, n_persons), dtype=bool)
    for d_idx, date_str in enumerate(all_dates):
        date_map = true_dict.get(date_str, {})
        for cls_i, roll in enumerate(roll_names):
            tv = date_map.get(roll, '')
            if tv in ('unknown', 'nan', 'none', ''):
                continue
            gt_daily_valid[d_idx, cls_i] = True
            if tv == 'present':
                gt_daily_present[d_idx, cls_i] = True

    std_protocols = ['5 min', '10 min', '15 min', '20 min', '30 min']
    active = sorted({session_protocol[s] for s in all_sessions if s in session_protocol})
    eval_protocols = [p for p in std_protocols if p in active]
    eval_protocols += [p for p in active if p not in eval_protocols and p != 'Unknown']
    if 'Unknown' in active:
        eval_protocols.append('Unknown')
    proto_sidx = {}
    for p in eval_protocols:
        ids = [i for i, s in enumerate(all_sessions) if session_protocol.get(s) == p]
        proto_sidx[f"Acc_{p.replace(' ', '')}"] = torch.tensor(ids, dtype=torch.int64, device=device)

    return {
        'device': device, 'dtype': dtype,
        'T': len(lens), 'N': cursor,
        'embs': embs, 'norms': norms, 'box_w': bw, 'box_h': bh,
        'lens_np': np.asarray(lens, dtype=np.int64),
        'starts_t': torch.tensor(starts, dtype=torch.int64, device=device),
        'alive_t':  torch.tensor(alive,  dtype=torch.int64, device=device),
        'sess_t':   torch.tensor(sess,   dtype=torch.int64, device=device),
        'date_t':   torch.tensor(dates,  dtype=torch.int64, device=device),
        'y_real_t': torch.tensor(y_real, dtype=torch.int64, device=device),
        'min_frames_floor': int(min_frames_floor),
        'n_sessions': n_sessions, 'n_dates': n_dates, 'n_persons': n_persons,
        'all_sessions': all_sessions, 'all_dates': all_dates,
        'gt_present':       torch.from_numpy(gt_present).to(device),
        'gt_valid':         torch.from_numpy(gt_valid).to(device),
        'gt_daily_present': torch.from_numpy(gt_daily_present).to(device),
        'gt_daily_valid':   torch.from_numpy(gt_daily_valid).to(device),
        'proto_sidx': proto_sidx,
        'fpv_cache': {},
    }


@torch.no_grad()
def evaluate_combo_batch_gpu(combo_batch, bank, ref_embs_t, device, at_budget=None, tag=0):
    """
    Evaluates a batch of parameter combinations against all cached tracks.
    Primary Objective (`Overall_Accuracy`):
      - Combines all protocols/videos on the same date into daily detection counts per student
        and selects the optimal threshold t in 1..20 (matching process_daily_attendance.py).
    Secondary Diagnostics:
      - `Session_Accuracy` and per-protocol accuracies (`Acc_5min`, etc.).
    """
    at_budget = at_budget or AT_BUDGET
    B = len(combo_batch)
    if B == 0:
        return []

    n_classes = bank['n_persons']

    def _pt(key, dtype=torch.float32):
        return torch.tensor([p[key] for _, p in combo_batch], dtype=dtype, device=device)

    P = {k: _pt(k) for k in ["CONFIDENCE_THRESHOLD", "NORM_QUALITY_THRESHOLD", "EMA_MOMENTUM", "SOFTMAX_TAU",
                             "MIN_BOX_SIZE", "MIN_SAMPLE_RATIO", "MIN_WIN_RATIO"]}
    for k in ["MIN_TOTAL_FRAMES", "MIN_TOTAL_SAMPLES"]:
        P[k] = _pt(k, torch.int64)

    if int(P['MIN_TOTAL_FRAMES'].min().item()) < bank['min_frames_floor']:
        raise ValueError("MIN_TOTAL_FRAMES below the floor used when the track bank was built.")

    n_sessions = bank['n_sessions']
    n_dates    = bank['n_dates']
    presence     = torch.zeros(B, max(n_sessions, 1), n_classes, dtype=torch.bool, device=device)
    daily_counts = torch.zeros(B, max(n_dates, 1), n_classes, dtype=torch.int32, device=device)

    if n_sessions > 0 and bank['T'] > 0:
        fpv_list = [int(p['FRAMES_PER_VOTE']) for _, p in combo_batch]

        groups = []
        for fpv in sorted(set(fpv_list)):
            grp = torch.tensor([i for i, f in enumerate(fpv_list) if f == fpv], dtype=torch.int64, device=device)
            info = _fpv_info(bank, fpv)
            if info is None:
                continue
            min_s = int(P['MIN_TOTAL_SAMPLES'][grp].min().item())
            Te = int((info['W_np'] >= min_s).sum())
            if Te == 0:
                continue
            work = int(grp.numel()) * int(np.minimum(info['n_gt'], Te).sum())
            groups.append((fpv, grp, work, info['Wmax']))

        live = {
            'phase': 'running',
            'rows_done': 0,
            'rows_total': max(1, sum(g[2] for g in groups)),
            't0': time.perf_counter(),
        }
        LIVE_STATUS[tag] = live
        for gi, (fpv, grp, work, Wmax) in enumerate(groups, start=1):
            _eval_fpv_group(bank, ref_embs_t, fpv, grp, P, presence, daily_counts, n_classes, at_budget, live)
        live['phase'] = 'finalising'

    # ---- 1. Daily Combined Evaluation (Threshold 1..20 Sweep per Combo) ----
    gt_dp = bank['gt_daily_present'].unsqueeze(0).unsqueeze(0)  # (1, 1, n_dates, n_persons)
    gt_dv = bank['gt_daily_valid'].unsqueeze(0).unsqueeze(0)    # (1, 1, n_dates, n_persons)
    daily_counts = daily_counts[:, :n_dates, :]

    thresh_t = torch.arange(1, MAX_DAILY_THRESHOLD + 1, device=device, dtype=torch.int32).view(1, -1, 1, 1)
    est_daily = daily_counts.unsqueeze(1) >= thresh_t           # (B, 20, n_dates, n_persons)

    tp_d = (est_daily & gt_dp & gt_dv).sum(dim=(2, 3))          # (B, 20)
    tn_d = ((~est_daily) & (~gt_dp) & gt_dv).sum(dim=(2, 3))
    fp_d = (est_daily & (~gt_dp) & gt_dv).sum(dim=(2, 3))
    fn_d = ((~est_daily) & gt_dp & gt_dv).sum(dim=(2, 3))
    matches_d = tp_d + tn_d                                     # (B, 20)

    # Tie-breaker across thresholds: maximize matches, then maximize TP, then minimize FP
    score_d = matches_d.to(torch.int64) * 1_000_000_000 + tp_d.to(torch.int64) * 1_000 - fp_d.to(torch.int64)
    best_t_idx = score_d.argmax(dim=1)                          # (B,)
    b_arange = torch.arange(B, device=device)

    best_thresh_list = (best_t_idx + 1).cpu().tolist()
    tp = tp_d[b_arange, best_t_idx].cpu().tolist()
    tn = tn_d[b_arange, best_t_idx].cpu().tolist()
    fp = fp_d[b_arange, best_t_idx].cpu().tolist()
    fn = fn_d[b_arange, best_t_idx].cpu().tolist()
    matches = matches_d[b_arange, best_t_idx].cpu().tolist()
    total_daily_valid = int(bank['gt_daily_valid'].sum().item())

    # ---- 2. Session & Protocol Breakdown Diagnostics ----
    gt_present, gt_valid = bank['gt_present'], bank['gt_valid']
    presence = presence[:, :n_sessions, :]
    sess_match_t = ((presence & gt_present) | ((~presence) & (~gt_present))) & gt_valid
    sess_matches = sess_match_t.sum(dim=(1, 2)).cpu().tolist()
    total_sess_valid = int(gt_valid.sum().item())

    proto_accs = {}
    for col, s_idx_t in bank['proto_sidx'].items():
        ptv = int(gt_valid[s_idx_t].sum().item())
        if ptv > 0:
            pm = sess_match_t[:, s_idx_t, :].sum(dim=(1, 2)).float() / ptv * 100.0
            proto_accs[col] = [round(a, 4) for a in pm.cpu().tolist()]
        else:
            proto_accs[col] = [0.0] * B

    results = []
    for b_i, (combo_id, params) in enumerate(combo_batch):
        daily_acc = (matches[b_i] / total_daily_valid * 100.0) if total_daily_valid > 0 else 0.0
        sess_acc  = (sess_matches[b_i] / total_sess_valid * 100.0) if total_sess_valid > 0 else 0.0
        res = {
            "Combo_ID": combo_id,
            **params,
            "Best_Daily_Threshold": int(best_thresh_list[b_i]),
            "Overall_Accuracy": round(daily_acc, 4),
            "Daily_Accuracy": round(daily_acc, 4),
            "Session_Accuracy": round(sess_acc, 4),
        }
        for col in proto_accs:
            res[col] = proto_accs[col][b_i]
        res.update({
            "Matches": matches[b_i],
            "Total_Valid": total_daily_valid,
            "TP": tp[b_i],
            "TN": tn[b_i],
            "FP": fp[b_i],
            "FN": fn[b_i],
        })
        results.append(res)
    return results


# ==========================================
# MULTI-GPU STAGE 2 WORKER EVALUATION POOL
# ==========================================
GPU_DATA_CACHE = {}        # gpu_id -> (track_bank, ref_embs_t)
LIVE_STATUS    = {}        # gpu_id -> concise telemetry dict
RESULT_Q       = queue.Queue()
target_names   = None
y_real         = None
roll_names     = None
true_dict      = None
short_to_roll  = None


def get_compact_gpu_summary():
    """Returns a compact inline string with overall GPU batch % done and time left for the single-line progress bar."""
    done_sum = 0
    total_sum = 0
    max_eta_sec = 0.0
    any_running = False

    for g in ACTIVE_GPUS:
        st = LIVE_STATUS.get(g)
        if not st:
            continue
        phase = st.get('phase')
        if phase == 'running':
            any_running = True
            d = st.get('rows_done', 0)
            t = max(1, st.get('rows_total', 1))
            done_sum += d
            total_sum += t
            el = time.perf_counter() - st.get('t0', time.perf_counter())
            if d > 0 and el > 1.0:
                eta = el * (t - d) / d
                if eta > max_eta_sec:
                    max_eta_sec = eta
        elif phase in ('finalising', 'idle'):
            t = max(1, st.get('rows_total', 1))
            done_sum += t
            total_sum += t

    if total_sum == 0:
        return "GPU: 0.0% | ETA: --"

    pct = min(100.0, 100.0 * done_sum / total_sum)
    if any_running and max_eta_sec > 0:
        if max_eta_sec >= 60.0:
            eta_str = f"~{max_eta_sec / 60.0:.1f}m left"
        else:
            eta_str = f"~{max_eta_sec:.0f}s left"
    elif pct >= 99.9:
        eta_str = "done"
    else:
        eta_str = "estimating..."

    return f"GPU: {pct:5.1f}% ({eta_str})"


def chunk_combinations(items, num_chunks):
    """Chunks a list of (combo_id, params) into equal parts based on num_chunks."""
    if not items or num_chunks <= 0:
        return []
    k, m = divmod(len(items), num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        if start < end:
            chunks.append(items[start:end])
    return chunks


def worker_evaluate_chunk(chunk, gpu_id):
    """Runs this GPU's share of combos; each finished sub-batch is pushed to RESULT_Q for the main thread."""
    device = torch.device(f'cuda:{gpu_id}')
    bank, ref_embs_t = GPU_DATA_CACHE[gpu_id]
    n_done = 0
    LIVE_STATUS[gpu_id] = {'phase': 'starting', 'rows_done': 0, 'rows_total': 1, 't0': time.perf_counter()}
    with torch.cuda.device(device):
        for b_start in range(0, len(chunk), GPU_BATCH_SIZE):
            combo_batch = chunk[b_start:b_start + GPU_BATCH_SIZE]
            budget = AT_BUDGET
            while True:
                try:
                    batch_results = evaluate_combo_batch_gpu(combo_batch, bank, ref_embs_t, device, budget, gpu_id)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if budget <= 2000:
                        raise
                    budget //= 2
            RESULT_Q.put(batch_results)
            n_done += len(batch_results)
    st = LIVE_STATUS.get(gpu_id, {})
    LIVE_STATUS[gpu_id] = {'phase': 'idle', 'rows_done': st.get('rows_total', 1), 'rows_total': st.get('rows_total', 1)}
    return n_done


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

    proto_cols = [c for c in ["Acc_5min", "Acc_10min", "Acc_15min", "Acc_20min", "Acc_30min"] if c in df.columns]
    other_proto_cols = [c for c in df.columns if c.startswith("Acc_") and c not in proto_cols]
    all_proto_cols = proto_cols + other_proto_cols

    leader_table = Table(title=f"[bold green]TOP 15 OPTIMAL HYPERPARAMETER COMBINATIONS (OUT OF {len(df):,} TESTED)[/bold green]")
    leader_table.add_column("Rank", justify="center", style="bold yellow")
    leader_table.add_column("Combo ID", justify="center", style="cyan")
    for k in varied_keys:
        short_h = k.replace("_THRESHOLD", "").replace("MIN_", "")
        leader_table.add_column(short_h, justify="right")
    if "Best_Daily_Threshold" in df.columns:
        leader_table.add_column("DayThr", justify="center", style="bold cyan")
    leader_table.add_column("Daily TP/TN/FP/FN", justify="center", style="magenta")
    if "Session_Accuracy" in df.columns:
        leader_table.add_column("SessAcc", justify="right", style="cyan")
    leader_table.add_column("Daily Accuracy", justify="right", style="bold green")

    for rank, (_, row) in enumerate(df_sorted.head(15).iterrows(), start=1):
        row_vals = [str(rank), f"#{int(row['Combo_ID'])}"]
        for k in varied_keys:
            val = row[k]
            row_vals.append(f"{int(val)}" if float(val).is_integer() else f"{val:.3f}")
        if "Best_Daily_Threshold" in df.columns:
            thr_val = row.get("Best_Daily_Threshold", 1)
            row_vals.append(f">={int(thr_val)}" if pd.notna(thr_val) else ">=1")
        row_vals.append(f"{int(row['TP'])}/{int(row['TN'])}/{int(row['FP'])}/{int(row['FN'])}")
        if "Session_Accuracy" in df.columns:
            s_val = row.get("Session_Accuracy")
            row_vals.append(f"{float(s_val):.2f}%" if pd.notna(s_val) else "--")
        row_vals.append(f"{row['Overall_Accuracy']:.2f}%")
        leader_table.add_row(*row_vals)

    console.print("\n")
    console.print(leader_table)

    best_params_str = "\n".join([f"  * [bold cyan]{k}[/bold cyan]: {best_row[k]}" for k in param_keys if k in best_row])
    best_thr_val = int(best_row["Best_Daily_Threshold"]) if "Best_Daily_Threshold" in best_row and pd.notna(best_row["Best_Daily_Threshold"]) else 1
    sess_acc_val = float(best_row["Session_Accuracy"]) if "Session_Accuracy" in best_row and pd.notna(best_row["Session_Accuracy"]) else 0.0
    proto_breakdown = "  ".join([f"[cyan]{pc.replace('Acc_', '')}:[/cyan] {best_row[pc]:.2f}%" for pc in all_proto_cols if pc in best_row and pd.notna(best_row[pc])])

    console.print(Panel(
        f"[bold green]Max Daily Combined Accuracy: {best_row['Overall_Accuracy']:.2f}% (Combo #{int(best_row['Combo_ID'])}, Optimal Daily Detection Count >= {best_thr_val})[/bold green]\n"
        + (f"[bold white]Individual Session Accuracy:[/bold white] {sess_acc_val:.2f}%   |   {proto_breakdown}\n" if proto_breakdown else "")
        + f"[bold magenta]Daily Confusion Breakdown:[/bold magenta] TP={int(best_row['TP'])}, TN={int(best_row['TN'])}, FP={int(best_row['FP'])}, FN={int(best_row['FN'])}\n\n"
        f"[bold white]Optimal Generalization Parameters for test_yolo_sota.py:[/bold white]\n{best_params_str}",
        title="Stage 2 Daily Attendance Optimizer Conclusion",
        expand=False
    ))

    fig = plt.figure(figsize=(18, 12), dpi=150)
    fig.suptitle(
        f"Stage 2 Daily Attendance Optimizer ({len(df):,} Combos) -- Best Daily Accuracy: {best_row['Overall_Accuracy']:.2f}% (Count >= {best_thr_val})",
        fontsize=16, fontweight='bold', y=0.98
    )

    # 1. Top 12 Combos
    ax1 = plt.subplot(2, 2, 1)
    top_n = df_sorted.head(12).iloc[::-1]
    labels = [
        f"#{int(r['Combo_ID'])} (>={int(r.get('Best_Daily_Threshold', 1))})"
        if 'Best_Daily_Threshold' in r and pd.notna(r['Best_Daily_Threshold'])
        else f"#{int(r['Combo_ID'])}"
        for _, r in top_n.iterrows()
    ]
    bars = ax1.barh(labels, top_n["Overall_Accuracy"], color="#2ecc71", edgecolor="#27ae60")
    ax1.set_xlabel("Daily Combined Accuracy (%)")
    ax1.set_title("Top 12 Parameter Combinations (with Best Daily Count Threshold)", fontweight="bold")
    if len(top_n) > 0:
        min_x = max(0.0, top_n["Overall_Accuracy"].min() - 3.0)
        max_x = min(100.0, top_n["Overall_Accuracy"].max() + 1.5)
        ax1.set_xlim(min_x, max_x)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va='center', fontsize=9, fontweight='bold')

    # 2. Epoch Progression
    ax2 = plt.subplot(2, 2, 2)
    if "Epoch" in df.columns and df["Epoch"].nunique() > 1:
        epoch_agg = df.groupby("Epoch")["Overall_Accuracy"].agg(["mean", "max"]).reset_index()
        ax2.plot(epoch_agg["Epoch"], epoch_agg["max"], marker="o", color="#e74c3c", linewidth=2.5, label="Max Daily Accuracy (%)")
        ax2.plot(epoch_agg["Epoch"], epoch_agg["mean"], marker="s", color="#3498db", linewidth=2.0, linestyle="--", label="Mean Daily Accuracy (%)")
        ax2.set_xlabel("Epoch", fontweight="bold")
        ax2.set_ylabel("Daily Accuracy (%)", fontweight="bold")
        ax2.set_title("Daily Accuracy Progression Across Epochs", fontweight="bold")
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
        ax2.set_title(f"Max Daily Accuracy Heatmap: {p1} vs {p2}", fontweight="bold")
        plt.colorbar(im, ax=ax2, label="Max Daily Accuracy (%)")
    else:
        ax2.text(0.5, 0.5, "Single Parameter Swept", ha="center", va="center")

    # 3. Sensitivity across active variables
    ax3 = plt.subplot(2, 1, 2)
    if varied_keys:
        x_positions, x_labels, mean_vals, max_vals, group_boundaries = [], [], [], [], []
        pos = 0
        for k in varied_keys:
            grouped = df.groupby(k)["Overall_Accuracy"].agg(["mean", "max"]).reset_index()
            # Keep chart readable when many slices exist by showing up to 8 quantile slices per variable
            if len(grouped) > 8:
                idx_sel = np.linspace(0, len(grouped) - 1, 8).round().astype(int)
                grouped = grouped.iloc[np.unique(idx_sel)]
            for _, grow in grouped.iterrows():
                x_positions.append(pos)
                val_str = f"{int(grow[k])}" if float(grow[k]).is_integer() else f"{grow[k]:.2f}"
                x_labels.append(f"{k.replace('_THRESHOLD', '').replace('MIN_', '')}\n={val_str}")
                mean_vals.append(grow["mean"])
                max_vals.append(grow["max"])
                pos += 1
            group_boundaries.append(pos - 0.5)

        width = 0.38
        x_arr = np.array(x_positions)
        ax3.bar(x_arr - width/2, mean_vals, width=width, label="Mean Daily Accuracy (%)", color="#3498db")
        ax3.bar(x_arr + width/2, max_vals, width=width, label="Max Daily Accuracy (%)", color="#e67e22")
        for b in group_boundaries[:-1]:
            ax3.axvline(b, color="gray", linestyle="--", alpha=0.5)
        ax3.set_xticks(x_arr)
        ax3.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
        ax3.set_ylabel("Daily Accuracy (%)")
        ax3.set_title("Active Hyperparameter Sensitivity (Mean vs. Max Daily Accuracy across tested slices)", fontweight="bold")
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
    gpu_desc = ", ".join([f"cuda:{g}" for g in ACTIVE_GPUS])

    console.print(Panel(
        f"[bold yellow]Main Process ID (PID) for Excel Sheet:[/bold yellow] [bold white]{main_pid}[/bold white]\n"
        f"[bold cyan]Multi-GPU Acceleration:[/bold cyan] Active GPUs: {ACTIVE_GPUS} ({gpu_desc})\n"
        f"[bold green]Shared Feature Cache:[/bold green] {FEATURE_CACHE_DIR}\n"
        f"[bold magenta]Sweep Output Directory:[/bold magenta] {SWEEP_OUTPUT_DIR}\n"
        f"[bold white]Optimization Target:[/bold white] Daily Combined Attendance Accuracy (Protocols per day summed, Threshold 1..{MAX_DAILY_THRESHOLD})",
        title="DGX A100 GPU Hyperparameter Optimizer (Copy PID to SPCLAB-DGX_Utilization.xlsx)",
        expand=False
    ))

    # 1. Discover all videos from VIDEOS/ or fallback to FEATURE_CACHE_DIR
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov')
    try:
        target_videos = sorted([
            f for f in os.listdir(INPUT_VIDEOS_DIR)
            if f.lower().endswith(video_extensions) and not f.startswith('.')
        ])
        console.print(f"[bold cyan]Discovered {len(target_videos)} videos in '{INPUT_VIDEOS_DIR}'.[/bold cyan]\n")
    except FileNotFoundError:
        console.print(f"[bold yellow]'{INPUT_VIDEOS_DIR}' not found. Falling back to feature cache: '{FEATURE_CACHE_DIR}'...[/bold yellow]")
        fallback_videos = []
        for f in os.listdir(FEATURE_CACHE_DIR):
            if f.startswith('.'):
                continue
            if f.endswith('_raw_features.pt'):
                fallback_videos.append(f[:-len('_raw_features.pt')] + '.mp4')
            elif f.endswith('_features.pt'):
                fallback_videos.append(f[:-len('_features.pt')] + '.mp4')
        target_videos = sorted(list(set(fallback_videos)))
        console.print(f"[bold cyan]Discovered {len(target_videos)} cached pseudo-videos in '{FEATURE_CACHE_DIR}'.[/bold cyan]\n")

    if not target_videos:
        console.print("[bold red]No videos found in VIDEOS folder or feature cache![/bold red]")
        sys.exit(0)

    # 2. Extract or load ground truth for all videos
    true_df, short_to_roll, base_to_protocol = prepare_subset_true_attendance(
        target_videos, RESULTS_BASE_DIR, SWEEP_OUTPUT_DIR, BASE_DIR
    )
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

    # 4. Load all cached video features and FAISS gallery directly into CPU shared memory
    console.print("\n[bold yellow]Loading video caches and gallery into CPU shared memory...[/bold yellow]")
    cpu_videos, ref_embs_cpu, target_names, y_real, roll_names, total_cached_frames = _load_gpu_sweep_data(
        target_videos, FEATURE_CACHE_DIR, BASE_DIR, short_to_roll, true_dict, base_to_protocol
    )

    if not cpu_videos:
        console.print("[bold red]No valid video caches loaded! Exiting.[/bold red]")
        sys.exit(0)

    num_cached_videos = len(cpu_videos)

    # Pre-stage ONE flat "track bank" per GPU (with MIN_FRAMES_FLOOR = 10 to allow sweeping MIN_TOTAL_FRAMES down to 10)
    console.print(f"\n[bold cyan]Pre-staging flattened track bank directly onto GPUs {ACTIVE_GPUS} (MIN_FRAMES_FLOOR={MIN_FRAMES_FLOOR})...[/bold cyan]")
    GPU_DATA_CACHE = {}
    for gpu_id in ACTIVE_GPUS:
        dev = torch.device(f"cuda:{gpu_id}")
        ref_embs_t = ref_embs_cpu.to(dev).t().contiguous()
        bank = build_track_bank(cpu_videos, true_dict, roll_names, y_real, dev, MIN_FRAMES_FLOOR)
        GPU_DATA_CACHE[gpu_id] = (bank, ref_embs_t)
        console.print(
            f"  [bold green]OK cuda:{gpu_id}: {bank['T']:,} tracks, {bank['N']:,} embeddings, "
            f"{bank['n_dates']} dates ({bank['n_sessions']} sessions), gallery={ref_embs_t.shape[1]:,}[/bold green]"
        )

    del cpu_videos
    del ref_embs_cpu
    gc.collect()

    # 5. Initialize Progressive Epoch Optimizer
    # Uses a dedicated CSV for daily-combined optimization so old session-only rows don't pollute the cache
    final_accuracy_csv = os.path.join(SWEEP_OUTPUT_DIR, "sweep_final_accuracy.csv")

    evaluated_cache = {}
    max_combo_id = 0
    best_acc_so_far = -1.0
    best_tp_so_far  = -1
    best_fp_so_far  = 999999
    best_cid_so_far = None
    best_thr_so_far = 1
    best_params_so_far = None

    if os.path.exists(final_accuracy_csv):
        try:
            existing_df = pd.read_csv(final_accuracy_csv)
            # Only resume from CSV if it already contains the new Daily_Accuracy / Best_Daily_Threshold schema
            if not existing_df.empty and "Combo_ID" in existing_df.columns and "Best_Daily_Threshold" in existing_df.columns:
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
                        thr = int(row.get("Best_Daily_Threshold", 1))
                        if (acc > best_acc_so_far) or (acc == best_acc_so_far and tp > best_tp_so_far) or (acc == best_acc_so_far and tp == best_tp_so_far and fp < best_fp_so_far):
                            best_acc_so_far = acc
                            best_tp_so_far  = tp
                            best_fp_so_far  = fp
                            best_cid_so_far = cid
                            best_thr_so_far = thr
                            best_params_so_far = p_dict
                console.print(f"[bold cyan]Found {len(evaluated_cache)} already-evaluated daily combinations in {final_accuracy_csv}. Resuming...[/bold cyan]")
            elif not existing_df.empty and "Best_Daily_Threshold" not in existing_df.columns:
                backup_csv = os.path.join(SWEEP_OUTPUT_DIR, f"sweep_session_legacy_{int(time.time())}.csv")
                os.replace(final_accuracy_csv, backup_csv)
                console.print(f"[bold yellow]Archived legacy session-based CSV to {backup_csv} to start fresh daily-protocol optimization.[/bold yellow]")
        except Exception as e:
            console.print(f"[bold yellow]Warning reading existing sweep CSV: {e}[/bold yellow]")

    # Initial baseline center point (seeded from your previous best combo #430)
    if best_params_so_far is not None:
        theta_star = dict(best_params_so_far)
    else:
        theta_star = {k: cfg["center"] for k, cfg in ACTIVE_VARS.items()}

    current_steps = {k: v["step"] for k, v in ACTIVE_VARS.items()}
    t_opt_start = time.perf_counter()

    num_sessions = GPU_DATA_CACHE[ACTIVE_GPUS[0]][0]['n_sessions'] if ACTIVE_GPUS else 0
    num_dates    = GPU_DATA_CACHE[ACTIVE_GPUS[0]][0]['n_dates'] if ACTIVE_GPUS else 0
    console.print(f"\n[bold white]Starting 10-Variable Daily Attendance Optimizer (MAX_EPOCHS={MAX_EPOCHS}):[/bold white]")
    console.print(f"  * Parallel GPUs: {ACTIVE_GPUS}  |  Batch Size: {GPU_BATCH_SIZE} combos/batch per GPU")
    console.print(f"  * Cached Videos: {num_cached_videos} across {num_dates} dates / {num_sessions} sessions ({total_cached_frames:,} total frames)\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ACTIVE_GPUS)) as executor:
        for epoch in range(1, MAX_EPOCHS + 1):
            # 1. Generate broader candidate neighborhood around theta*
            candidates = [dict(theta_star)]

            # 1D coordinate steps: +-1*Delta, +-2*Delta, +-3*Delta across all 10 active variables
            for k, cfg in ACTIVE_VARS.items():
                delta = current_steps[k]
                for mult in [-3, -2, -1, 1, 2, 3]:
                    val = theta_star[k] + mult * delta
                    val = max(cfg["min"], min(cfg["max"], val))
                    cand = dict(theta_star)
                    cand[k] = int(round(val)) if cfg["is_int"] else round(val, 4)
                    candidates.append(cand)

            # 2D pairwise interaction steps for TOP6_VARS
            for k1, k2 in itertools.combinations(TOP6_VARS, 2):
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
                    c_thr = int(row.get("Best_Daily_Threshold", 1))
                    if (c_acc > best_acc_so_far) or (c_acc == best_acc_so_far and c_tp > best_tp_so_far) or (c_acc == best_acc_so_far and c_tp == best_tp_so_far and c_fp < best_fp_so_far):
                        best_acc_so_far = c_acc
                        best_tp_so_far  = c_tp
                        best_fp_so_far  = c_fp
                        best_cid_so_far = c_id
                        best_thr_so_far = c_thr
                        best_params_so_far = {k: row[k] for k in ALL_PARAM_KEYS}
                else:
                    pending_candidates.append(c)

            total_epoch_combos = len(unique_epoch_candidates)
            already_evaluated_in_epoch = total_epoch_combos - len(pending_candidates)
            improved_this_epoch = False

            if pending_candidates:
                # Single-line Rich Progress Bar (updates in-place with GPU % and time left)
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]{task.description}"),
                    BarColumn(bar_width=28),
                    MofNCompleteColumn(),
                    TextColumn("[bold yellow]{task.percentage:>5.1f}%"),
                    TimeRemainingColumn(),
                    TextColumn("[bold green]{task.fields[gpu_info]}"),
                    TextColumn("[bold magenta]{task.fields[best]}"),
                    console=console,
                    transient=False,
                    refresh_per_second=4
                )

                best_label = (
                    f"Best: {best_acc_so_far:.2f}% (#{best_cid_so_far}, >={best_thr_so_far})"
                    if best_cid_so_far is not None else "Best: --"
                )
                task_id = progress.add_task(
                    f"Epoch {epoch:02d}/{MAX_EPOCHS:02d}",
                    total=total_epoch_combos,
                    completed=already_evaluated_in_epoch,
                    gpu_info="GPU: 0.0% (starting)",
                    best=best_label
                )

                pending_with_ids = []
                for cand in pending_candidates:
                    max_combo_id += 1
                    pending_with_ids.append((max_combo_id, cand))

                completed_in_epoch = already_evaluated_in_epoch
                epoch_eval_count = 0
                epoch_eval_start = time.perf_counter()

                chunks = chunk_combinations(pending_with_ids, len(ACTIVE_GPUS))
                active_assignments = list(zip(chunks, ACTIVE_GPUS[:len(chunks)]))

                with progress:
                    futures = [
                        executor.submit(worker_evaluate_chunk, chunk, gpu_id)
                        for chunk, gpu_id in active_assignments
                    ]
                    not_done = set(futures)

                    while not_done or not RESULT_Q.empty():
                        done, not_done = concurrent.futures.wait(
                            not_done, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for fut in done:
                            fut.result()

                        new_batches = []
                        while True:
                            try:
                                new_batches.append(RESULT_Q.get_nowait())
                            except queue.Empty:
                                break

                        buffer_rows = []
                        for chunk_results in new_batches:
                            for res in chunk_results:
                                res["Epoch"] = epoch
                                buffer_rows.append(res)
                                evaluated_cache[param_key(res)] = res
                                completed_in_epoch += 1
                                epoch_eval_count += 1

                                acc = res["Overall_Accuracy"]
                                tp  = res["TP"]
                                fp  = res["FP"]
                                thr = res.get("Best_Daily_Threshold", 1)

                                if (acc > best_acc_so_far) or (acc == best_acc_so_far and tp > best_tp_so_far) or (acc == best_acc_so_far and tp == best_tp_so_far and fp < best_fp_so_far):
                                    best_acc_so_far    = acc
                                    best_tp_so_far     = tp
                                    best_fp_so_far     = fp
                                    best_cid_so_far    = res["Combo_ID"]
                                    best_thr_so_far    = thr
                                    best_params_so_far = {k: res[k] for k in ALL_PARAM_KEYS}
                                    improved_this_epoch = True

                        if buffer_rows:
                            df_chunk = pd.DataFrame(buffer_rows)
                            write_header = not os.path.exists(final_accuracy_csv)
                            df_chunk.to_csv(final_accuracy_csv, mode='a', header=write_header, index=False)

                        elapsed_total = time.perf_counter() - t_opt_start
                        elapsed_eval = time.perf_counter() - epoch_eval_start
                        combos_sec = epoch_eval_count / elapsed_eval if elapsed_eval > 0 else 0.0
                        eff_fps = combos_sec * total_cached_frames
                        remaining_combos_epoch = total_epoch_combos - completed_in_epoch
                        eta_sec = remaining_combos_epoch / combos_sec if combos_sec > 0 else 0.0

                        progress.update(
                            task_id,
                            completed=completed_in_epoch,
                            gpu_info=get_compact_gpu_summary(),
                            best=(
                                f"Best: {best_acc_so_far:.2f}% (#{best_cid_so_far}, >={best_thr_so_far})"
                                if best_cid_so_far is not None else "Best: --"
                            )
                        )

                        if buffer_rows:
                            write_live_progress(
                                SWEEP_OUTPUT_DIR, main_pid, epoch, MAX_EPOCHS,
                                completed_in_epoch, total_epoch_combos, len(evaluated_cache),
                                combos_sec, eff_fps, elapsed_total, eta_sec,
                                best_acc_so_far, best_cid_so_far, best_thr_so_far, best_params_so_far
                            )
            else:
                console.print(
                    f"[bold cyan]Epoch {epoch:02d}/{MAX_EPOCHS:02d}:[/bold cyan] All {total_epoch_combos} combinations "
                    f"already cached | [bold magenta]Best: {best_acc_so_far:.2f}% (#{best_cid_so_far}, >={best_thr_so_far})[/bold magenta]"
                )

            # Update center point theta* to the best combination found
            if best_params_so_far is not None:
                theta_star = dict(best_params_so_far)

            # Smooth step size decay (0.65x per epoch) across all 10 variables
            for k, cfg in ACTIVE_VARS.items():
                if cfg["is_int"]:
                    current_steps[k] = max(1, int(round(current_steps[k] * STEP_DECAY)))
                else:
                    current_steps[k] = max(0.005, round(current_steps[k] * STEP_DECAY, 4))

            all_at_minimum = all(
                (current_steps[k] <= 1 if cfg["is_int"] else current_steps[k] <= 0.005)
                for k, cfg in ACTIVE_VARS.items()
            )
            if all_at_minimum and not improved_this_epoch:
                console.print("[bold green]Local convergence reached at minimum step precision. Finishing optimizer early.[/bold green]")
                break

    # 6. Generate Final Visualization & Rich Leaderboard Report
    console.print("\n[bold yellow]Generating final report and visualization...[/bold yellow]")
    generate_visualizations_and_report(final_accuracy_csv, SWEEP_OUTPUT_DIR, ALL_PARAM_KEYS)
    console.print("\n[bold green]Optimizer run completed successfully![/bold green]")
# pyright: reportMissingImports=false
# sweep_yolo_sota.py – DGX A100 GPU-Accelerated Hyperparameter Sweep (Stage 2)
# ==============================================================================
# Workflow:
#   1. Locks GPU visibility strictly to GPU 0 (leaves GPU 7 & busy GPUs untouched).
#   2. Prints Main Process ID (PID) at startup for SPCLAB-DGX_Utilization.xlsx.
#   3. Stage 1 (YOLO+AdaFace): unchanged — uses GPU 0 via ProcessPoolExecutor.
#   4. Stage 2 (Parameter Sweep): 100% GPU-native batched evaluation.
#      - Loads ALL cached .pt track tensors & FAISS gallery onto CUDA:0 once.
#      - Processes GPU_BATCH_SIZE param combos at a time with fully vectorized ops.
#      - Keeps live progress bar, live_status.txt, sweep_progress.json.
#      - Maintains resume support via sweep_final_accuracy.csv.
# ==============================================================================

import os
# Strictly use GPU 0 only (avoids GPU 1, GPU 4, and reserved GPU 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
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
import sys
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
# 1. HARDCODED 4 VIDEOS & SWEEP GRID
# ==========================================
SWEEP_VIDEOS = [
    "20260821_111106_cam1_raw.mp4",
    "20260821_111106_cam2_raw.mp4",
    "20260910_104345_cam1_raw.mp4",
    "20260910_104345_cam2_raw.mp4",
]

# Total combinations = 4 x 3 x 3 x 3 x 1 x 2 x 3 x 3 x 3 x 3 = 5,832 combinations
SWEEP_GRID = {
    "CONFIDENCE_THRESHOLD":   [0.34, 0.37, 0.40, 0.43],
    "NORM_QUALITY_THRESHOLD": [5.5, 7.0, 8.5],
    "EMA_MOMENTUM":           [0.55, 0.70, 0.85],
    "FRAMES_PER_VOTE":        [3, 5, 8],
    "SOFTMAX_TAU":            [5.0],
    "MIN_BOX_SIZE":           [50, 60],
    "MIN_TOTAL_FRAMES":       [30, 45, 60],
    "MIN_TOTAL_SAMPLES":      [20, 30, 40],
    "MIN_SAMPLE_RATIO":       [0.25, 0.33, 0.40],
    "MIN_WIN_RATIO":          [0.45, 0.52, 0.60],
}

BASE_MIN_BOX_SIZE = min(45, min(SWEEP_GRID["MIN_BOX_SIZE"]))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEOS_DIR = os.path.join(BASE_DIR, "VIDEOS")

if os.path.exists(os.path.join(BASE_DIR, "ATTENDENCE_RESULTS")):
    ATTENDANCE_ROOT = os.path.join(BASE_DIR, "ATTENDENCE_RESULTS")
else:
    ATTENDANCE_ROOT = os.path.join(BASE_DIR, "ATTENDENCE RESULTS")

RESULTS_BASE_DIR = os.path.join(ATTENDANCE_ROOT, "Results")
MINE_OUTPUT_DIR = os.path.join(ATTENDANCE_ROOT, "MINE")
FEATURE_CACHE_DIR = os.path.join(MINE_OUTPUT_DIR, "feature_cache")

# Stage 1 extraction workers (still CPU+GPU via spawn)
CPU_COUNT = os.cpu_count() or 8
MAX_VIDEO_WORKERS = 4
MAX_SWEEP_WORKERS = min(96, max(8, CPU_COUNT - 32)) if CPU_COUNT >= 64 else max(8, CPU_COUNT - 2)
FRAME_SKIP = 1

# GPU Sweep Tuning:
# How many parameter combos to evaluate in one GPU batch.
# 256 is safe for an A100-80GB; lower to 128 if OOM occurs.
GPU_BATCH_SIZE = 256

# ==========================================
# 2. TRUE ATTENDANCE CACHING & LIVE PROGRESS WRITER
# ==========================================
def prepare_subset_true_attendance(sweep_videos, results_base_dir, mine_dir):
    os.makedirs(mine_dir, exist_ok=True)
    subset_true_path = os.path.join(mine_dir, "true_attendance_4videos.csv")
    full_true_path = os.path.join(results_base_dir, "filtered_true_attendance.csv")
    name_mapping_path = os.path.join(results_base_dir, "name_mapping.csv")

    if not os.path.exists(name_mapping_path):
        raise FileNotFoundError(f"Could not find name_mapping.csv at {name_mapping_path}")

    name_map_df = pd.read_csv(name_mapping_path).dropna(subset=['Short_Name', 'Roll_Name'])
    name_map_df = name_map_df[name_map_df['Short_Name'].astype(str).str.strip() != '']
    short_to_roll = dict(zip(
        name_map_df['Short_Name'].astype(str).str.strip(),
        name_map_df['Roll_Name'].astype(str).str.strip()
    ))

    required_dates = sorted(list({v.split('_')[0] for v in sweep_videos}))

    need_create = True
    if os.path.exists(subset_true_path):
        existing_df = pd.read_csv(subset_true_path)
        existing_cols = [str(c).strip().replace('-', '') for c in existing_df.columns]
        if all(d in existing_cols for d in required_dates):
            console.print(f"[bold green]OK Subset True Attendance already exists:[/bold green] {subset_true_path} (Skipping creation)")
            need_create = False

    if need_create:
        if not os.path.exists(full_true_path):
            raise FileNotFoundError(f"Could not find filtered_true_attendance.csv at {full_true_path}")

        console.print(f"[bold yellow]Creating Subset True Attendance for dates {required_dates} in MINE folder...[/bold yellow]")
        full_df = pd.read_csv(full_true_path)
        full_df.columns = [str(c).strip().replace('-', '') for c in full_df.columns]

        base_cols = list(full_df.columns[:2])
        date_cols = [d for d in required_dates if d in full_df.columns]
        missing_dates = [d for d in required_dates if d not in full_df.columns]
        if missing_dates:
            console.print(f"[bold red]Warning:[/bold red] Dates {missing_dates} not found in filtered_true_attendance.csv!")

        subset_df = full_df[base_cols + date_cols].copy()
        subset_df.to_csv(subset_true_path, index=False)
        console.print(f"[bold green]OK Saved Subset True Attendance to:[/bold green] {subset_true_path}")

    true_df = pd.read_csv(subset_true_path)
    roll_col = true_df.iloc[:, 0].astype(str).str.strip()
    true_df = true_df.set_index(roll_col)
    true_df.columns = true_df.columns.astype(str).str.strip().str.replace('-', '')
    for col in true_df.columns:
        true_df[col] = true_df[col].astype(str).str.strip()

    return true_df, short_to_roll


def write_live_progress(mine_dir, pid, completed, total, rate, elapsed_sec, eta_sec, best_acc, best_cid, best_params):
    pct = (completed / total * 100.0) if total > 0 else 100.0
    status_data = {
        "pid": pid,
        "gpu_device": 0,
        "gpu_batch_size": GPU_BATCH_SIZE,
        "completed_combos": completed,
        "total_combos": total,
        "percent": round(pct, 2),
        "combos_per_sec": round(rate, 2),
        "video_evals_per_sec": round(rate * len(SWEEP_VIDEOS), 2),
        "elapsed_min": round(elapsed_sec / 60.0, 2),
        "eta_min": round(eta_sec / 60.0, 2),
        "best_accuracy": round(best_acc, 4),
        "best_combo_id": best_cid,
        "best_params": best_params,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    json_path = os.path.join(mine_dir, "sweep_progress.json")
    txt_path = os.path.join(mine_dir, "live_status.txt")

    bar_len = 30
    filled = int(bar_len * pct / 100.0)
    bar_str = "#" * filled + "." * (bar_len - filled)

    txt_content = (
        f"=================================================================\n"
        f"DGX A100 GPU HYPERPARAMETER SWEEP LIVE MONITOR (Updated: {status_data['updated_at']})\n"
        f"=================================================================\n"
        f"PID      : {pid}  |  GPU: 0 (CUDA)  |  Batch Size: {GPU_BATCH_SIZE} combos/batch\n"
        f"Progress : [{bar_str}] {pct:6.2f}%\n"
        f"Combos   : {completed:,} / {total:,} ({completed * len(SWEEP_VIDEOS):,} / {total * len(SWEEP_VIDEOS):,} Video Runs)\n"
        f"Speed    : {rate:.1f} combos/sec  ({rate * len(SWEEP_VIDEOS):.1f} videos/sec)\n"
        f"Elapsed  : {elapsed_sec/60.0:.1f} min  |  ETA Remaining: {eta_sec/60.0:.1f} min\n"
        f"Best Acc : {best_acc:.2f}% (Combo #{best_cid})\n"
        f"=================================================================\n"
    )
    try:
        with open(json_path, "w") as f:
            json.dump(status_data, f, indent=2)
        with open(txt_path, "w") as f:
            f.write(txt_content)
    except OSError:
        pass


# ==========================================
# 3. MODELS, ALIGNMENT & VIDEO READER (STAGE 1)
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


# ==========================================
# 4. STAGE 1: ONE-TIME YOLO + ADAFACE FEATURE CACHE
# ==========================================
ext_yolo_model = None
ext_resnet = None
ext_device = None
ext_use_half = False


def init_extractor_worker():
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

                for r_id in raw_ids:
                    if r_id not in raw_to_local_id:
                        local_id_counter += 1
                        raw_to_local_id[r_id] = local_id_counter

                    t_id = raw_to_local_id[r_id]
                    local_ids_list.append(t_id)

                    if t_id not in active_tracks:
                        active_tracks[t_id] = {
                            'frames_alive': 0,
                            'missing_frames': 0,
                            'box_w': [],
                            'box_h': [],
                            'embs': [],
                            'norms': []
                        }
                    active_tracks[t_id]['frames_alive'] += 1

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
                    'box_w': torch.tensor(data['box_w'], dtype=torch.float32),
                    'box_h': torch.tensor(data['box_h'], dtype=torch.float32),
                    'embs': torch.cat(data['embs'], dim=0).half(),
                    'norms': torch.cat(data['norms'], dim=0).float()
                }
            else:
                serialized_tracks[t_id] = {
                    'frames_alive': data['frames_alive'],
                    'box_w': torch.empty(0, dtype=torch.float32),
                    'box_h': torch.empty(0, dtype=torch.float32),
                    'embs': torch.empty((0, 512), dtype=torch.float16),
                    'norms': torch.empty(0, dtype=torch.float32)
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


def run_extraction_dashboard(progress_q, total_videos, stop_event):
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
                table.add_row("[italic dim]Initializing / Finishing workers...[/italic dim]", "-", "-", "-")

            live.update(table, refresh=True)
            time.sleep(0.2)


def ensure_all_features_cached(sweep_videos, input_dir, cache_dir, ctx):
    os.makedirs(cache_dir, exist_ok=True)
    missing_cache = []
    for v in sweep_videos:
        stem = os.path.splitext(v)[0]
        cpath = os.path.join(cache_dir, f"{stem}_features.pt")
        if not (os.path.exists(cpath) and os.path.getsize(cpath) > 1024):
            missing_cache.append(v)

    if not missing_cache:
        console.print(f"[bold green]OK All 4 YOLO+AdaFace feature cache files already exist in:[/bold green] {cache_dir}")
        console.print("[bold green]  -> Skipping video decoding & GPU extraction! Jumping straight to Stage 2 parameter sweep.[/bold green]")
        return

    missing_vids = [v for v in missing_cache if not os.path.exists(os.path.join(input_dir, v))]
    if missing_vids:
        raise FileNotFoundError(
            f"Missing cache files for {missing_cache} in '{cache_dir}', and raw videos {missing_vids} not found in '{input_dir}'!"
        )

    ensure_ocsort_tracker()
    console.print(f"[bold yellow]Extracting YOLO+AdaFace embeddings for {len(missing_cache)} video(s) (One-Time Process)...[/bold yellow]")
    manager = ctx.Manager()
    progress_q = manager.Queue()
    stop_event = threading.Event()

    dash_t = threading.Thread(
        target=run_extraction_dashboard,
        args=(progress_q, len(missing_cache), stop_event),
        daemon=True
    )
    dash_t.start()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(MAX_VIDEO_WORKERS, len(missing_cache)),
        mp_context=ctx,
        initializer=init_extractor_worker
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
    console.print(f"[bold green]OK One-time YOLO+AdaFace feature extraction complete![/bold green]")


# ==========================================
# 5. STAGE 2: GPU-ACCELERATED BATCH SWEEP ENGINE
# ==========================================

def _load_gpu_sweep_data(sweep_videos, cache_dir, base_dir, short_to_roll, true_dict, device):
    """
    Load ALL data needed for Stage 2 onto the GPU once.

    Returns:
        gpu_videos   : list of dicts, one per video, each containing:
                         date_str, tracks (list of track-dicts with GPU tensors)
        ref_embs     : (G, 512) float32 GPU tensor -- normalized gallery embeddings
        target_names : list[str]
        y_real       : list[int]
        roll_names   : list[str] -- roll-mapped target names
        true_dict    : passed through
        short_to_roll: passed through
    """
    # Gallery
    faiss_index_path = os.path.join(base_dir, 'face_attendance_sota_faiss.bin')
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"Missing '{faiss_index_path}'!")

    console.print(f"[cyan]  Loading FAISS gallery -> GPU...[/cyan]")
    index = faiss.read_index(faiss_index_path)
    index.nprobe = 20
    try:
        ref_np = index.reconstruct_n(0, index.ntotal)
    except AttributeError:
        index.make_direct_map()
        ref_np = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    ref_embs = F.normalize(
        torch.from_numpy(ref_np).float().to(device), p=2, dim=1
    )  # (G, 512)

    meta_path = os.path.join(base_dir, 'face_attendance_sota_meta.pkl')
    with open(meta_path, 'rb') as f:
        saved_data = pickle.load(f)
    target_names = saved_data['target_names']
    y_real       = saved_data['y_real']

    roll_names = [short_to_roll.get(n.strip(), n.strip()) for n in target_names]

    # Feature caches
    console.print(f"[cyan]  Loading {len(sweep_videos)} feature caches -> GPU...[/cyan]")
    gpu_videos = []
    for v in sweep_videos:
        date_str = v.split('_')[0]
        stem = os.path.splitext(v)[0]
        cpath = os.path.join(cache_dir, f"{stem}_features.pt")
        raw_tracks = torch.load(cpath, map_location='cpu', weights_only=False)

        gpu_tracks = []
        for t_id in sorted(raw_tracks.keys()):
            td = raw_tracks[t_id]
            embs  = td['embs'].float()
            norms = td['norms'].float()
            if embs.numel() == 0:
                continue
            gpu_tracks.append({
                'frames_alive': int(td['frames_alive']),
                'box_w': td['box_w'].to(device),
                'box_h': td['box_h'].to(device),
                'embs' : embs.to(device),
                'norms': norms.to(device),
            })
        gpu_videos.append({'date_str': date_str, 'tracks': gpu_tracks})
        console.print(f"[dim]    {v}: {len(gpu_tracks)} valid tracks on GPU[/dim]")

    return gpu_videos, ref_embs, target_names, y_real, roll_names, true_dict, short_to_roll


@torch.no_grad()
def _eval_track_batch(
    embs, norms, frames_alive,
    box_w, box_h,
    ref_embs,
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
    For a single track, accumulate per-class vote counts for all B param combos.
    Iterates over unique FRAMES_PER_VOTE values (at most 3) to stay vectorized.
    vote_counts  : (B, n_classes) float32 -- modified in-place
    total_preds  : (B,) int32             -- modified in-place
    """
    B = b_norm_thresh.shape[0]
    S = embs.shape[0]

    # Per-combo box-size mask: box_ok[b, s] = True if both dims pass threshold b
    box_ok = (box_w.unsqueeze(0) >= b_min_box_size.unsqueeze(1)) & \
             (box_h.unsqueeze(0) >= b_min_box_size.unsqueeze(1))  # (B, S)

    # Frames alive gate per combo
    alive_ok = (frames_alive >= b_min_total_frames)  # (B,) bool

    for fpv_val in b_frames_per_vote.unique():
        fpv = fpv_val.item()
        combo_mask = (b_frames_per_vote == fpv_val)
        active = combo_mask & alive_ok  # (B,)
        if not active.any():
            continue

        act_idx = active.nonzero(as_tuple=True)[0]  # (A,)
        A = act_idx.shape[0]

        a_norm_thresh  = b_norm_thresh[act_idx]    # (A,)
        a_ema_momentum = b_ema_momentum[act_idx]   # (A,)
        a_softmax_tau  = b_softmax_tau[act_idx]    # (A,)
        a_box_ok       = box_ok[act_idx]           # (A, S)

        # Zero out norm for samples that fail box filter
        masked_norms = norms.unsqueeze(0) * a_box_ok.float()  # (A, S)

        num_full_windows = S // fpv
        if num_full_windows == 0:
            continue

        used_S  = num_full_windows * fpv
        embs_w  = embs[:used_S].view(num_full_windows, fpv, 512)          # (W, fpv, 512)
        norms_w = masked_norms[:, :used_S].view(A, num_full_windows, fpv) # (A, W, fpv)

        ema_emb = None  # (A, 512)
        a_votes  = torch.zeros(A, n_classes, device=embs.device, dtype=torch.float32)
        a_totals = torch.zeros(A, dtype=torch.int32, device=embs.device)

        for w in range(num_full_windows):
            w_embs  = embs_w[w]        # (fpv, 512)
            w_norms = norms_w[:, w, :] # (A, fpv)

            q_mask_aw = (w_norms >= a_norm_thresh.unsqueeze(1))  # (A, fpv)
            any_valid = q_mask_aw.any(dim=1)                      # (A,)
            if not any_valid.any():
                continue

            weights = torch.softmax(
                w_norms / a_softmax_tau.unsqueeze(1), dim=1
            )  # (A, fpv)

            # Weighted average embedding for each combo: (A, 512)
            e_win = F.normalize(
                torch.bmm(
                    weights.unsqueeze(1),
                    w_embs.unsqueeze(0).expand(A, -1, -1)
                ).squeeze(1),
                p=2, dim=1
            )  # (A, 512)

            if ema_emb is None:
                ema_emb = e_win
            else:
                ema_emb = F.normalize(
                    a_ema_momentum.unsqueeze(1) * ema_emb +
                    (1.0 - a_ema_momentum).unsqueeze(1) * e_win,
                    p=2, dim=1
                )

            # Similarity to gallery: (A, G)
            sim = torch.mm(ema_emb, ref_embs.t())
            max_sim, max_idx = sim.max(dim=1)    # (A,), (A,)
            pred_class = y_real_t[max_idx]       # (A,) class indices

            a_conf_thresh_active = b_conf_thresh[act_idx]
            is_conf = (max_sim >= a_conf_thresh_active) & any_valid  # (A,)

            # Vectorized vote accumulation (replaces slow Python for-loop)
            a_totals += 1
            if is_conf.any():
                conf_indices = is_conf.nonzero(as_tuple=True)[0]  # indices where confident
                conf_classes = pred_class[conf_indices]            # their predicted classes
                a_votes.scatter_add_(
                    1,
                    conf_classes.unsqueeze(1),
                    torch.ones(conf_indices.shape[0], 1, device=embs.device, dtype=torch.float32)
                )

        # Vectorized write-back to global accumulators
        vote_counts[act_idx] += a_votes
        total_preds[act_idx] += a_totals

    return vote_counts, total_preds


@torch.no_grad()
def evaluate_combo_batch_gpu(
    combo_batch,
    gpu_videos,
    ref_embs,
    target_names,
    y_real,
    roll_names,
    true_dict,
    short_to_roll,
    device,
):
    """
    Evaluate a batch of parameter combinations fully on the GPU.
    Returns a list of result dicts (same schema as old CPU worker output).
    """
    B = len(combo_batch)
    n_classes = len(target_names)
    y_real_t = torch.tensor(y_real, dtype=torch.int64, device=device)  # (G,)

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

    # presence[b, date_idx, person_idx] = True if that person was identified
    presence = torch.zeros(B, n_dates, n_persons, dtype=torch.bool, device=device)

    for vid_info in gpu_videos:
        date_str = vid_info['date_str']
        tracks   = vid_info['tracks']
        d_idx    = date_to_idx.get(date_str, -1)
        if d_idx < 0:
            continue

        for td in tracks:
            embs         = td['embs']          # (S, 512) GPU
            norms        = td['norms']          # (S,) GPU
            frames_alive = td['frames_alive']   # int
            box_w        = td['box_w']          # (S,) GPU
            box_h        = td['box_h']          # (S,) GPU
            S = embs.shape[0]
            if S == 0:
                continue

            vote_counts = torch.zeros(B, n_classes, device=device, dtype=torch.float32)
            total_preds = torch.zeros(B, dtype=torch.int32, device=device)

            _eval_track_batch(
                embs, norms, frames_alive,
                box_w, box_h,
                ref_embs, y_real_t,
                b_norm_thresh, b_frames_per_vote, b_ema_momentum,
                b_softmax_tau, b_min_box_size, b_min_total_frames,
                b_min_total_samples, b_min_sample_ratio,
                b_conf_thresh, b_min_win_ratio,
                vote_counts, total_preds,
                n_classes,
            )

            # Apply decision thresholds
            best_class_votes, best_class_idx = vote_counts.max(dim=1)  # (B,), (B,)
            total_valid_votes = vote_counts.sum(dim=1)                  # (B,)

            alive_ok     = (frames_alive >= b_min_total_frames)
            samples_ok   = (total_preds.long() >= b_min_total_samples)
            sample_ratio = best_class_votes / (total_preds.float().clamp(min=1))
            sample_ok_r  = (sample_ratio >= b_min_sample_ratio)
            win_ratio    = best_class_votes / (total_valid_votes.clamp(min=1))
            win_ok       = (win_ratio >= b_min_win_ratio)
            has_votes    = (total_valid_votes > 0)

            confirmed = alive_ok & samples_ok & sample_ok_r & win_ok & has_votes  # (B,)

            # Scatter confirmed predictions into presence tensor
            conf_idx = confirmed.nonzero(as_tuple=True)[0]
            if conf_idx.numel() > 0:
                cls_idx = best_class_idx[conf_idx]
                presence[conf_idx, d_idx, cls_idx] = True

    # Score each combo
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
# 6. VISUALIZATION & LEADERBOARD REPORTING
# ==========================================
def generate_visualizations_and_report(final_csv_path, output_dir, param_keys):
    if not os.path.exists(final_csv_path):
        console.print("[bold red]No sweep results found to visualize![/bold red]")
        return

    df = pd.read_csv(final_csv_path)
    if df.empty:
        return

    df_sorted = df.sort_values(by=["Overall_Accuracy", "TP"], ascending=[False, False]).reset_index(drop=True)
    best_row = df_sorted.iloc[0]
    varied_keys = [k for k in param_keys if k in df.columns and df[k].nunique() > 1]

    leader_table = Table(title="[bold green]TOP 15 HYPERPARAMETER COMBINATIONS (OUT OF " + str(len(df)) + " TESTED)[/bold green]")
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
            row_vals.append(f"{int(val)}" if float(val).is_integer() else f"{val}")
        row_vals.append(f"{int(row['TP'])} / {int(row['TN'])} / {int(row['FP'])} / {int(row['FN'])}")
        row_vals.append(f"{row['Overall_Accuracy']:.2f}%")
        leader_table.add_row(*row_vals)

    console.print("\n")
    console.print(leader_table)

    best_params_str = "\n".join([f"  * [bold cyan]{k}[/bold cyan]: {best_row[k]}" for k in param_keys if k in best_row])
    console.print(Panel(
        f"[bold green]Max Overall Accuracy Achieved: {best_row['Overall_Accuracy']:.2f}% (Combo #{int(best_row['Combo_ID'])})[/bold green]\n"
        f"[bold magenta]Confusion Breakdown:[/bold magenta] TP={int(best_row['TP'])}, TN={int(best_row['TN'])}, FP={int(best_row['FP'])}, FN={int(best_row['FN'])}\n\n"
        f"[bold white]Recommended Parameters to Generalize Across All Videos:[/bold white]\n{best_params_str}",
        title="Optimal Hyperparameter Sweep Conclusion",
        expand=False
    ))

    fig = plt.figure(figsize=(18, 12), dpi=150)
    fig.suptitle(
        f"Hyperparameter Sweep Analysis ({len(df)} Combos across 4 Videos) -- Best Accuracy: {best_row['Overall_Accuracy']:.2f}%",
        fontsize=16, fontweight='bold', y=0.98
    )

    ax1 = plt.subplot(2, 2, 1)
    top_n = df_sorted.head(12).iloc[::-1]
    labels = [f"Combo #{int(r['Combo_ID'])}" for _, r in top_n.iterrows()]
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

    ax2 = plt.subplot(2, 2, 2)
    if len(varied_keys) >= 2:
        p1, p2 = varied_keys[0], varied_keys[1]
        pivot = df.pivot_table(index=p1, columns=p2, values="Overall_Accuracy", aggfunc="max")
        im = ax2.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax2.set_xticks(range(len(pivot.columns)))
        ax2.set_xticklabels([str(c) for c in pivot.columns])
        ax2.set_yticks(range(len(pivot.index)))
        ax2.set_yticklabels([str(i) for i in pivot.index])
        ax2.set_xlabel(p2, fontweight="bold")
        ax2.set_ylabel(p1, fontweight="bold")
        ax2.set_title(f"Max Accuracy Heatmap: {p1} vs {p2}", fontweight="bold")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax2.text(j, i, f"{val:.2f}%", ha="center", va="center", color="white", fontweight="bold")
        plt.colorbar(im, ax=ax2, label="Max Accuracy (%)")
    else:
        ax2.text(0.5, 0.5, "Need >= 2 varied parameters for 2D Heatmap", ha="center", va="center")

    ax3 = plt.subplot(2, 1, 2)
    if varied_keys:
        x_positions, x_labels, mean_vals, max_vals, group_boundaries = [], [], [], [], []
        pos = 0
        for k in varied_keys:
            grouped = df.groupby(k)["Overall_Accuracy"].agg(["mean", "max"]).reset_index()
            for _, grow in grouped.iterrows():
                x_positions.append(pos)
                val_str = f"{int(grow[k])}" if float(grow[k]).is_integer() else f"{grow[k]}"
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
        ax3.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8.5)
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_title("Hyperparameter Sensitivity Across All Tested Values (Mean vs. Max Accuracy)", fontweight="bold")
        if mean_vals:
            ax3.set_ylim(max(0.0, min(mean_vals) - 4.0), min(100.0, max(max_vals) + 1.5))
        ax3.legend(loc="lower right")
        ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plot_path = os.path.join(output_dir, "sweep_visualization.png")
    plt.savefig(plot_path)
    plt.close(fig)
    console.print(f"[bold green]Saved Sweep Visualization Chart to:[/bold green] {plot_path}")


# ==========================================
# 7. MAIN ENTRY POINT
# ==========================================
if __name__ == '__main__':
    mp.freeze_support()
    ctx = mp.get_context('spawn')
    main_pid = os.getpid()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    console.print(Panel(
        f"[bold yellow]Main Process ID (PID) for Excel Sheet:[/bold yellow] [bold white]{main_pid}[/bold white]\n"
        f"[bold cyan]Assigned GPU Device:[/bold cyan] {device}  (CUDA_VISIBLE_DEVICES='0')\n"
        f"[bold green]Stage 2 Engine:[/bold green] GPU-Vectorized Batch Sweep  (batch={GPU_BATCH_SIZE} combos)\n"
        f"[bold magenta]Working Directory:[/bold magenta] {BASE_DIR}",
        title="DGX A100 Utilization Info (Copy to SPCLAB-DGX_Utilization.xlsx)",
        expand=False
    ))

    # Step 1: Prepare or load cached True Attendance
    true_df, short_to_roll = prepare_subset_true_attendance(SWEEP_VIDEOS, RESULTS_BASE_DIR, MINE_OUTPUT_DIR)

    true_dict = {}
    for col in true_df.columns:
        col_clean = str(col).strip()
        if col_clean.isdigit():
            true_dict[col_clean] = {
                str(idx).strip(): str(val).strip().lower()
                for idx, val in true_df[col].items()
            }

    # Step 2: Build or verify one-time YOLO + AdaFace feature cache
    ensure_all_features_cached(SWEEP_VIDEOS, INPUT_VIDEOS_DIR, FEATURE_CACHE_DIR, ctx)

    # Step 3: Build parameter combinations
    param_keys = list(SWEEP_GRID.keys())
    param_values = [SWEEP_GRID[k] for k in param_keys]
    all_combos = [
        dict(zip(param_keys, prod))
        for prod in itertools.product(*param_values)
    ]

    final_accuracy_csv = os.path.join(MINE_OUTPUT_DIR, "sweep_final_accuracy.csv")

    completed_combo_ids = set()
    best_acc_so_far = 0.0
    best_cid_so_far = None
    best_params_so_far = {}

    if os.path.exists(final_accuracy_csv):
        try:
            existing_res = pd.read_csv(final_accuracy_csv)
            if "Combo_ID" in existing_res.columns and not existing_res.empty:
                completed_combo_ids = set(existing_res["Combo_ID"].astype(int).tolist())
                best_idx = existing_res["Overall_Accuracy"].idxmax()
                best_row_init = existing_res.loc[best_idx]
                best_acc_so_far = float(best_row_init["Overall_Accuracy"])
                best_cid_so_far = int(best_row_init["Combo_ID"])
                best_params_so_far = {k: best_row_init[k] for k in param_keys if k in best_row_init}
                console.print(
                    f"[bold cyan]Found {len(completed_combo_ids)} already-evaluated combinations in "
                    f"{final_accuracy_csv}. Resuming remaining...[/bold cyan]"
                )
        except Exception:
            pass

    pending_combos = [
        (idx + 1, params)
        for idx, params in enumerate(all_combos)
        if (idx + 1) not in completed_combo_ids
    ]

    total_all_combos = len(all_combos)
    already_done_count = len(completed_combo_ids)
    total_pending = len(pending_combos)

    if not pending_combos:
        console.print("[bold green]OK All combos already evaluated. Skipping to visualization.[/bold green]")
    else:
        console.print(
            f"\n[bold white]Stage 2: GPU-Accelerated Parameter Sweep on {device}:[/bold white]\n"
            f"   {total_pending:,} pending combos x {len(SWEEP_VIDEOS)} videos"
            f"  |  GPU batch = {GPU_BATCH_SIZE} combos at a time\n"
            f"   -> {total_pending * len(SWEEP_VIDEOS):,} total video evaluations"
        )

        # Load ALL GPU data once
        console.print("\n[bold yellow]Loading feature caches & gallery onto GPU...[/bold yellow]")
        gpu_videos, ref_embs, target_names, y_real, roll_names, _, _ = _load_gpu_sweep_data(
            SWEEP_VIDEOS, FEATURE_CACHE_DIR, BASE_DIR, short_to_roll, true_dict, device
        )
        console.print(
            f"[bold green]OK GPU data ready: {ref_embs.shape[0]} gallery embeddings, "
            f"{sum(len(v['tracks']) for v in gpu_videos)} total tracks[/bold green]\n"
        )

        # Progress bar setup
        t_sweep_start = time.perf_counter()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=38),
            MofNCompleteColumn(),
            TextColumn("[bold yellow]{task.percentage:>5.1f}%"),
            TimeRemainingColumn(),
            TextColumn("[bold green]{task.fields[speed]}"),
            TextColumn("[bold magenta]{task.fields[best]}"),
            console=console,
            refresh_per_second=5
        )

        task_id = progress.add_task(
            "GPU Sweep",
            total=total_all_combos,
            completed=already_done_count,
            speed=f"Starting (batch={GPU_BATCH_SIZE})...",
            best=f"Best: {best_acc_so_far:.2f}%" if best_cid_so_far else "Best: --"
        )

        write_live_progress(
            MINE_OUTPUT_DIR, main_pid, already_done_count, total_all_combos,
            0.0, 0.0, 0.0, best_acc_so_far, best_cid_so_far, best_params_so_far
        )

        buffer_rows = []
        completed_session = 0
        num_batches = (total_pending + GPU_BATCH_SIZE - 1) // GPU_BATCH_SIZE

        with progress:
            for batch_num in range(num_batches):
                batch_start = batch_num * GPU_BATCH_SIZE
                batch_end   = min(batch_start + GPU_BATCH_SIZE, total_pending)
                combo_batch = pending_combos[batch_start:batch_end]

                try:
                    batch_results = evaluate_combo_batch_gpu(
                        combo_batch, gpu_videos, ref_embs, target_names,
                        y_real, roll_names, true_dict, short_to_roll, device,
                    )
                except torch.cuda.OutOfMemoryError:
                    console.print(
                        f"[bold red]OOM on batch {batch_num+1} (size={len(combo_batch)}). "
                        f"Retrying with half-size sub-batches...[/bold red]"
                    )
                    batch_results = []
                    half = max(1, len(combo_batch) // 2)
                    for sub_start in range(0, len(combo_batch), half):
                        sub = combo_batch[sub_start:sub_start + half]
                        torch.cuda.empty_cache()
                        sub_res = evaluate_combo_batch_gpu(
                            sub, gpu_videos, ref_embs, target_names,
                            y_real, roll_names, true_dict, short_to_roll, device,
                        )
                        batch_results.extend(sub_res)

                for row_result in batch_results:
                    buffer_rows.append(row_result)
                    completed_session += 1

                    acc = row_result["Overall_Accuracy"]
                    if acc > best_acc_so_far:
                        best_acc_so_far    = acc
                        best_cid_so_far    = row_result["Combo_ID"]
                        best_params_so_far = {k: row_result[k] for k in param_keys}

                # Flush buffer every 50 results or at the end
                if len(buffer_rows) >= 50 or completed_session == total_pending:
                    df_chunk = pd.DataFrame(buffer_rows)
                    write_header = not os.path.exists(final_accuracy_csv)
                    df_chunk.to_csv(final_accuracy_csv, mode='a', header=write_header, index=False)
                    buffer_rows.clear()

                    elapsed = time.perf_counter() - t_sweep_start
                    rate = completed_session / elapsed if elapsed > 0 else 0.0
                    current_total_done = already_done_count + completed_session
                    remaining_combos = total_all_combos - current_total_done
                    eta_sec = remaining_combos / rate if rate > 0 else 0.0

                    progress.update(
                        task_id,
                        completed=current_total_done,
                        speed=f"{rate:.1f} combos/s ({rate*len(SWEEP_VIDEOS):.0f} vids/s)",
                        best=f"Best: {best_acc_so_far:.2f}% (#{best_cid_so_far})"
                    )
                    write_live_progress(
                        MINE_OUTPUT_DIR, main_pid, current_total_done, total_all_combos,
                        rate, elapsed, eta_sec, best_acc_so_far, best_cid_so_far, best_params_so_far
                    )

                # Periodic GPU cache cleanup to prevent fragmentation
                if (batch_num + 1) % 20 == 0:
                    torch.cuda.empty_cache()

    # Step 4: Output Best Parameters, Leaderboard & Visualization Graphs
    generate_visualizations_and_report(final_accuracy_csv, MINE_OUTPUT_DIR, param_keys)

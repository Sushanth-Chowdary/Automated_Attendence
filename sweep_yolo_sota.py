# sweep_yolo_sota.py – Fast Feature-Cached Asynchronous Multi-Variable Sweep
# ==============================================================================
# Workflow:
#   1. True Attendance Cache: Creates 'true_attendance_4videos.csv' in MINE/ once
#      (skips automatically if already present).
#   2. YOLO + AdaFace Feature Cache: Extracts per-track (frames_alive, box_w, box_h,
#      512-D AdaFace embedding, feature_norm) once per video and saves to
#      'ATTENDENCE RESULTS/MINE/feature_cache/<stem>_features.pt'.
#      If the .pt cache files already exist, skips video reading & GPU inference!
#   3. Asynchronous Multi-Variable Sweep (1,000+ combos):
#      Asynchronously simulates each video per parameter combo from the cached
#      embeddings -> writes temp_cXXXX_<video>_output.csv -> combines all 4 CSVs
#      as soon as a combo's 4 videos finish -> records Overall Accuracy in
#      'sweep_final_accuracy.csv' -> deletes the 4 temporary output.csv files.
#   4. Visualization & Leaderboard: Generates 'sweep_visualization.png' and prints
#      the top parameter combinations and optimal values.
# ==============================================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
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
import faiss
import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from ultralytics import YOLO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel

warnings.filterwarnings("ignore", category=FutureWarning)
console = Console()

# ==========================================
# 1. HARDCODED 4 VIDEOS & 1,944-COMBO SWEEP GRID
# ==========================================
SWEEP_VIDEOS = [
    "20260821_111106_cam1_raw.mp4",
    "20260821_111106_cam2_raw.mp4",
    "20260910_104345_cam1_raw.mp4",
    "20260910_104345_cam2_raw.mp4",
]

# Total combinations = 4 x 3 x 3 x 3 x 1 x 2 x 1 x 3 x 3 x 1 = 1,944 combinations
# (7,776 video evaluations). Feel free to add/modify any values!
SWEEP_GRID = {
    "CONFIDENCE_THRESHOLD":   [0.34, 0.37, 0.40, 0.43],
    "NORM_QUALITY_THRESHOLD": [5.5, 7.0, 8.5],
    "EMA_MOMENTUM":           [0.55, 0.70, 0.85],
    "FRAMES_PER_VOTE":        [3, 5, 8],
    "SOFTMAX_TAU":            [5.0],
    "MIN_BOX_SIZE":           [50, 60],
    "MIN_TOTAL_FRAMES":       [45],
    "MIN_TOTAL_SAMPLES":      [20, 30, 40],
    "MIN_SAMPLE_RATIO":       [0.25, 0.33, 0.40],
    "MIN_WIN_RATIO":          [0.45, 0.52, 0.60],
}

# The feature extractor captures all boxes >= BASE_MIN_BOX_SIZE so any MIN_BOX_SIZE >= 45
# in SWEEP_GRID can be filtered dynamically without re-running the videos.
BASE_MIN_BOX_SIZE = min(45, min(SWEEP_GRID["MIN_BOX_SIZE"]))

INPUT_VIDEOS_DIR = "VIDEOS"
RESULTS_BASE_DIR = "ATTENDENCE RESULTS/Results"
MINE_OUTPUT_DIR = os.path.abspath("ATTENDENCE RESULTS/MINE")
FEATURE_CACHE_DIR = os.path.join(MINE_OUTPUT_DIR, "feature_cache")

MAX_VIDEO_WORKERS = 4   # Parallel GPU workers for one-time feature extraction
MAX_SWEEP_WORKERS = 8   # Parallel asynchronous workers for cached parameter sweep
FRAME_SKIP = 1


# ==========================================
# 2. TRUE ATTENDANCE CACHING & EVALUATION
# ==========================================
def prepare_subset_true_attendance(sweep_videos, results_base_dir, mine_dir):
    """
    Creates (or loads if already present) true_attendance_4videos.csv inside MINE/
    for the 4 hardcoded videos, plus the Short_Name -> Roll_Name dictionary.
    """
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
            console.print(f"[bold green]✔ Subset True Attendance already exists:[/bold green] {subset_true_path} (Skipping creation)")
            need_create = False

    if need_create:
        if not os.path.exists(full_true_path):
            raise FileNotFoundError(f"Could not find filtered_true_attendance.csv at {full_true_path}")

        console.print(f"[bold yellow]⚙ Creating Subset True Attendance for dates {required_dates} in MINE folder...[/bold yellow]")
        full_df = pd.read_csv(full_true_path)
        full_df.columns = [str(c).strip().replace('-', '') for c in full_df.columns]

        base_cols = list(full_df.columns[:2])
        date_cols = [d for d in required_dates if d in full_df.columns]
        missing_dates = [d for d in required_dates if d not in full_df.columns]
        if missing_dates:
            console.print(f"[bold red]Warning:[/bold red] Dates {missing_dates} not found in filtered_true_attendance.csv!")

        subset_df = full_df[base_cols + date_cols].copy()
        subset_df.to_csv(subset_true_path, index=False)
        console.print(f"[bold green]✔ Saved Subset True Attendance to:[/bold green] {subset_true_path}")

    true_df = pd.read_csv(subset_true_path)
    roll_col = true_df.iloc[:, 0].astype(str).str.strip()
    true_df = true_df.set_index(roll_col)
    true_df.columns = true_df.columns.astype(str).str.strip().str.replace('-', '')
    for col in true_df.columns:
        true_df[col] = true_df[col].astype(str).str.strip()

    return true_df, short_to_roll


def evaluate_combo_and_cleanup(combo_id, params, csv_file_map, true_df, short_to_roll):
    """
    Combines the 4 temporary output CSVs for a completed parameter combination,
    calculates the exact union accuracy against true_df[cite: 1], and deletes the 4 CSVs.
    """
    sessions = {}
    for video_name, csv_path in csv_file_map.items():
        parts = video_name.split('_')
        base_ts = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else video_name
        sessions.setdefault(base_ts, []).append(csv_path)

    total_matches = 0
    total_valid = 0
    tp = fp = tn = fn = 0

    for ts, session_files in sorted(sessions.items()):
        date_str = ts.split('_')[0]
        if date_str not in true_df.columns:
            continue

        cam_dfs = [pd.read_csv(f) for f in session_files if os.path.exists(f)]
        if not cam_dfs:
            continue

        merged_cams = pd.concat(cam_dfs, ignore_index=True)
        merged_cams['Name'] = merged_cams['Name'].astype(str).str.strip()
        merged_cams['Roll_Name'] = merged_cams['Name'].map(short_to_roll).fillna(merged_cams['Name'])
        merged_cams['Is_Present'] = merged_cams['Status'].astype(str).str.strip().str.lower() == 'present'

        grouped = merged_cams.groupby('Roll_Name').agg(
            Is_Present=('Is_Present', 'any')
        ).reset_index()
        grouped['Estimated'] = grouped['Is_Present'].apply(lambda x: 'present' if x else 'absent')
        grouped = grouped.set_index('Roll_Name')

        grouped['True'] = true_df[date_str].reindex(grouped.index).fillna('')

        est_col = grouped['Estimated'].astype(str).str.strip().str.lower()
        true_col = grouped['True'].astype(str).str.strip().str.lower()

        valid_mask = (~true_col.isin(['unknown', 'nan', 'none', ''])) & (true_col != '')
        matches = (est_col == true_col) & valid_mask

        total_matches += int(matches.sum())
        total_valid += int(valid_mask.sum())

        tp += int(((est_col == 'present') & (true_col == 'present') & valid_mask).sum())
        tn += int(((est_col == 'absent') & (true_col == 'absent') & valid_mask).sum())
        fp += int(((est_col == 'present') & (true_col == 'absent') & valid_mask).sum())
        fn += int(((est_col == 'absent') & (true_col == 'present') & valid_mask).sum())

    # Immediately delete the 4 temporary output.csv files for this combo
    for csv_path in csv_file_map.values():
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
            except OSError:
                pass

    overall_acc = (total_matches / total_valid * 100.0) if total_valid > 0 else 0.0
    return {
        "Combo_ID": combo_id,
        **params,
        "Overall_Accuracy": round(overall_acc, 4),
        "Matches": total_matches,
        "Total_Valid": total_valid,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


# ==========================================
# 3. MODELS, ALIGNMENT & VIDEO READER
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
        if os.path.exists(wf):
            ckpt = torch.load(wf, map_location='cpu', weights_only=False)
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
        if os.path.exists(model_path):
            model = YOLO(model_path, task='detect')
            model.to(device)
            return model
    raise FileNotFoundError("No YOLO face model found!")


OCSORT_TRACKER_PATH = "custom_ocsort.yaml"


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
    """
    Runs YOLO tracking + 5-point crop alignment + AdaFace embedding extraction ONCE
    for a video and saves all per-track embeddings & box sizes to a .pt file[cite: 2].
    """
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

                    # Extract all candidate faces down to BASE_MIN_BOX_SIZE
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
                    'embs': torch.cat(data['embs'], dim=0).half(),   # (M, 512) float16
                    'norms': torch.cat(data['norms'], dim=0).float() # (M,) float32
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
        console.print(f"[bold green]✔ All 4 YOLO+AdaFace feature cache files already exist in:[/bold green] {cache_dir}")
        console.print("[bold green]  -> Skipping video decoding & GPU extraction! Jumping straight to 1,000+ parameter sweep.[/bold green]")
        return

    console.print(f"[bold yellow]⚙ Extracting YOLO+AdaFace embeddings for {len(missing_cache)} video(s) (One-Time Process)...[/bold yellow]")
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
    console.print(f"[bold green]✔ One-time YOLO+AdaFace feature extraction complete![/bold green]")


# ==========================================
# 5. STAGE 2: ULTRA-FAST ASYNC SWEEP WORKER
# ==========================================
sweep_cached_videos = {}
sweep_ref_embeddings = None
sweep_target_names = []
sweep_y_real = []
sweep_device = None


def init_sweep_worker(sweep_videos, cache_dir):
    """
    Loads the 4 cached .pt feature files and the FAISS gallery into worker memory
    once, so every (combo_id, video) task executes in milliseconds!
    """
    global sweep_cached_videos, sweep_ref_embeddings, sweep_target_names, sweep_y_real, sweep_device

    sweep_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    faiss_index_path = './face_attendance_sota_faiss.bin'
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError("Missing './face_attendance_sota_faiss.bin'!")

    index = faiss.read_index(faiss_index_path)
    index.nprobe = 20
    try:
        ref_embeddings = index.reconstruct_n(0, index.ntotal)
    except AttributeError:
        index.make_direct_map()
        ref_embeddings = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    sweep_ref_embeddings = torch.from_numpy(ref_embeddings).to(sweep_device).float()
    sweep_ref_embeddings = F.normalize(sweep_ref_embeddings, p=2, dim=1)

    meta_path = './face_attendance_sota_meta.pkl'
    with open(meta_path, 'rb') as f:
        saved_data = pickle.load(f)
    sweep_target_names, sweep_y_real = saved_data['target_names'], saved_data['y_real']

    sweep_cached_videos = {}
    for v in sweep_videos:
        stem = os.path.splitext(v)[0]
        cpath = os.path.join(cache_dir, f"{stem}_features.pt")
        tracks_data = torch.load(cpath, map_location='cpu', weights_only=False)
        # Keep tensors in float32 on CPU/GPU for ultra-fast slicing
        for t_id, t_info in tracks_data.items():
            t_info['embs'] = t_info['embs'].float()
        sweep_cached_videos[v] = tracks_data


def evaluate_cached_video_task(combo_id, params, video_filename, output_dir):
    """
    Replays the exact mathematical logic of test_yolo_sota.py for one video using
    the cached embeddings, writes temp_cXXXX_<video>_output.csv, and returns[cite: 2].
    """
    global sweep_cached_videos, sweep_ref_embeddings, sweep_target_names, sweep_y_real, sweep_device

    video_stem = os.path.splitext(video_filename)[0]
    temp_output_csv = os.path.join(output_dir, f"temp_c{combo_id:04d}_{video_stem}_output.csv")

    conf_thresh = params["CONFIDENCE_THRESHOLD"]
    norm_thresh = params["NORM_QUALITY_THRESHOLD"]
    ema_momentum = params["EMA_MOMENTUM"]
    frames_per_vote = int(params["FRAMES_PER_VOTE"])
    softmax_tau = params["SOFTMAX_TAU"]
    min_box_size = params["MIN_BOX_SIZE"]
    min_total_frames = params["MIN_TOTAL_FRAMES"]
    min_total_samples = params["MIN_TOTAL_SAMPLES"]
    min_sample_ratio = params["MIN_SAMPLE_RATIO"]
    min_win_ratio = params["MIN_WIN_RATIO"]

    tracks_data = sweep_cached_videos[video_filename]
    student_presence = {name: False for name in sweep_target_names}
    student_detection_count = {name: 0 for name in sweep_target_names}

    with torch.inference_mode():
        for t_id, data in tracks_data.items():
            total_frames = data['frames_alive']
            if total_frames < min_total_frames or data['embs'].numel() == 0:
                continue

            # 1. Filter by MIN_BOX_SIZE
            box_mask = (data['box_w'] >= min_box_size) & (data['box_h'] >= min_box_size)
            if not box_mask.any():
                continue

            embs = data['embs'][box_mask]
            norms = data['norms'][box_mask]
            num_crops = embs.size(0)

            # 2. Process in exact chunks of FRAMES_PER_VOTE (matching crop_buffer >= FRAMES_PER_VOTE)
            num_full_windows = num_crops // frames_per_vote
            if num_full_windows == 0:
                continue

            ema_emb = None
            all_preds = []

            for w_idx in range(num_full_windows):
                start_i = w_idx * frames_per_vote
                end_i = start_i + frames_per_vote

                w_embs = embs[start_i:end_i]
                w_norms = norms[start_i:end_i]

                # 3. Quality gate: feature norm >= NORM_QUALITY_THRESHOLD
                q_mask = w_norms >= norm_thresh
                if not q_mask.any():
                    continue

                e_track = w_embs[q_mask].to(sweep_device)
                n_track = w_norms[q_mask].to(sweep_device).unsqueeze(1)
                K = e_track.size(0)

                # 4. Softmax quality weights & window embedding
                weights = torch.softmax(n_track / softmax_tau, dim=0)
                e_win = F.normalize(torch.sum(weights * e_track, dim=0, keepdim=True), p=2, dim=1)

                # 5. EMA tracklet smoothing
                if ema_emb is None:
                    ema_emb = e_win
                else:
                    ema_emb = F.normalize(ema_momentum * ema_emb + (1.0 - ema_momentum) * e_win, p=2, dim=1)

                # 6. Cosine similarity against FAISS gallery
                sim_matrix = torch.mm(ema_emb, sweep_ref_embeddings.t())
                max_sim, max_idx = torch.max(sim_matrix, dim=1)
                conf_score = max_sim.item()
                winner = sweep_target_names[sweep_y_real[max_idx.item()]] if conf_score > conf_thresh else "Unknown"

                all_preds.extend([winner] * K)

            # 7. Track-level voting gate
            valid_preds = [p for p in all_preds if p != "Unknown"]
            valid_votes_count = len(valid_preds)
            if valid_votes_count > 0:
                counts = Counter(valid_preds)
                winner = counts.most_common(1)[0][0]
                win_ratio = counts.get(winner, 0) / valid_votes_count
                total_samples = len(all_preds)
                sample_ratio = counts.get(winner, 0) / total_samples if total_samples > 0 else 0.0

                if (total_frames >= min_total_frames and
                    total_samples >= min_total_samples and
                    sample_ratio >= min_sample_ratio and
                    win_ratio >= min_win_ratio):
                    if winner != "Unknown":
                        student_presence[winner] = True
                        student_detection_count[winner] += counts.get(winner, 0)

    # Save temporary output.csv for this video & combo
    output_data = [
        {
            'Name': s,
            'Status': 'Present' if student_presence[s] else 'Absent',
            'Detection Count': student_detection_count[s]
        }
        for s in sweep_target_names
    ]
    pd.DataFrame(output_data).to_csv(temp_output_csv, index=False)
    return combo_id, video_filename, temp_output_csv, "OK"


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
    varied_keys = [k for k in param_keys if df[k].nunique() > 1]

    # 1. Print Top 15 Leaderboard in Terminal
    leader_table = Table(title="[bold green]🏆 TOP 15 HYPERPARAMETER COMBINATIONS (OUT OF " + str(len(df)) + " TESTED)[/bold green]")
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

    # Best Parameters Summary Panel
    best_params_str = "\n".join([f"  • [bold cyan]{k}[/bold cyan]: {best_row[k]}" for k in param_keys])
    console.print(Panel(
        f"[bold green]Max Overall Accuracy Achieved: {best_row['Overall_Accuracy']:.2f}% (Combo #{int(best_row['Combo_ID'])})[/bold green]\n"
        f"[bold magenta]Confusion Breakdown:[/bold magenta] TP={int(best_row['TP'])}, TN={int(best_row['TN'])}, FP={int(best_row['FP'])}, FN={int(best_row['FN'])}\n\n"
        f"[bold white]Recommended Parameters to Generalize Across All Videos:[/bold white]\n{best_params_str}",
        title="✨ Optimal Hyperparameter Sweep Conclusion",
        expand=False
    ))

    # 2. Generate Multi-Panel Visualization Chart
    fig = plt.figure(figsize=(18, 12), dpi=150)
    fig.suptitle(
        f"Hyperparameter Sweep Analysis ({len(df)} Combos across 4 Videos) — Best Accuracy: {best_row['Overall_Accuracy']:.2f}%",
        fontsize=16, fontweight='bold', y=0.98
    )

    # Panel A: Top 12 Combinations Bar Chart
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

    # Panel B: 2D Heatmap of CONFIDENCE_THRESHOLD vs NORM_QUALITY_THRESHOLD
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

    # Panel C: Parameter Sensitivity Curves (Mean & Max Accuracy per Value)
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
    console.print(f"[bold green]📊 Saved Sweep Visualization Chart to:[/bold green] {plot_path}")


# ==========================================
# 7. MAIN ENTRY POINT
# ==========================================
if __name__ == '__main__':
    mp.freeze_support()
    ensure_ocsort_tracker()
    ctx = mp.get_context('spawn')

    # Verify the 4 hardcoded videos exist if feature cache isn't already built
    missing_vids = [
        v for v in SWEEP_VIDEOS
        if not os.path.exists(os.path.join(FEATURE_CACHE_DIR, f"{os.path.splitext(v)[0]}_features.pt"))
        and not os.path.exists(os.path.join(INPUT_VIDEOS_DIR, v))
    ]
    if missing_vids:
        raise FileNotFoundError(f"Missing hardcoded videos in '{INPUT_VIDEOS_DIR}': {missing_vids}")

    # Step 1: Prepare or load cached True Attendance for the 4 videos in MINE/
    true_df, short_to_roll = prepare_subset_true_attendance(SWEEP_VIDEOS, RESULTS_BASE_DIR, MINE_OUTPUT_DIR)

    # Step 2: Build or verify one-time YOLO + AdaFace feature cache for the 4 videos
    ensure_all_features_cached(SWEEP_VIDEOS, INPUT_VIDEOS_DIR, FEATURE_CACHE_DIR, ctx)

    # Step 3: Build parameter combinations
    param_keys = list(SWEEP_GRID.keys())
    param_values = [SWEEP_GRID[k] for k in param_keys]
    all_combos = [
        dict(zip(param_keys, prod))
        for prod in itertools.product(*param_values)
    ]

    final_accuracy_csv = os.path.join(MINE_OUTPUT_DIR, "sweep_final_accuracy.csv")

    # Resume support: skip combinations already saved in sweep_final_accuracy.csv
    completed_combo_ids = set()
    if os.path.exists(final_accuracy_csv):
        try:
            existing_res = pd.read_csv(final_accuracy_csv)
            if "Combo_ID" in existing_res.columns:
                completed_combo_ids = set(existing_res["Combo_ID"].astype(int).tolist())
                if completed_combo_ids:
                    console.print(
                        f"[bold cyan]↻ Found {len(completed_combo_ids)} already-evaluated combinations in "
                        f"{final_accuracy_csv}. Resuming remaining...[/bold cyan]"
                    )
        except Exception:
            pass

    pending_combos = [
        (idx + 1, params)
        for idx, params in enumerate(all_combos)
        if (idx + 1) not in completed_combo_ids
    ]

    total_combos = len(pending_combos)
    total_video_runs = total_combos * len(SWEEP_VIDEOS)

    if pending_combos:
        console.print(
            f"\n[bold white]🚀 Stage 2: Launching Asynchronous Parameter Sweep:[/bold white] "
            f"{total_combos} parameter combos × {len(SWEEP_VIDEOS)} videos = {total_video_runs} async video evaluations."
        )
        t_sweep_start = time.perf_counter()

        combo_params_lookup = {cid: p for cid, p in pending_combos}
        combo_completed_csvs = {cid: {} for cid, _ in pending_combos}
        completed_combos_count = 0
        best_acc_so_far = 0.0
        best_cid_so_far = None

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_SWEEP_WORKERS,
            mp_context=ctx,
            initializer=init_sweep_worker,
            initargs=(SWEEP_VIDEOS, FEATURE_CACHE_DIR)
        ) as executor:
            futures = {}
            for combo_id, params in pending_combos:
                for video_name in SWEEP_VIDEOS:
                    fut = executor.submit(
                        evaluate_cached_video_task,
                        combo_id, params, video_name, MINE_OUTPUT_DIR
                    )
                    futures[fut] = (combo_id, video_name)

            with Live(auto_refresh=False, refresh_per_second=8) as live:
                for fut in concurrent.futures.as_completed(futures):
                    combo_id, video_name = futures[fut]
                    try:
                        cid, vname, csv_path, status = fut.result()
                        if status == "OK" and csv_path:
                            combo_completed_csvs[cid][vname] = csv_path

                            # As soon as all 4 videos for this combo finish:
                            if len(combo_completed_csvs[cid]) == len(SWEEP_VIDEOS):
                                row_result = evaluate_combo_and_cleanup(
                                    cid,
                                    combo_params_lookup[cid],
                                    combo_completed_csvs[cid],
                                    true_df,
                                    short_to_roll
                                )
                                row_df = pd.DataFrame([row_result])
                                write_header = not os.path.exists(final_accuracy_csv)
                                row_df.to_csv(final_accuracy_csv, mode='a', header=write_header, index=False)

                                completed_combos_count += 1
                                acc = row_result["Overall_Accuracy"]
                                if acc > best_acc_so_far:
                                    best_acc_so_far = acc
                                    best_cid_so_far = cid

                                del combo_completed_csvs[cid]

                                if completed_combos_count % 10 == 0 or completed_combos_count == total_combos:
                                    elapsed = time.perf_counter() - t_sweep_start
                                    rate = completed_combos_count / elapsed if elapsed > 0 else 0.0
                                    status_tbl = Table(box=None)
                                    status_tbl.add_column("Sweep Progress", style="bold cyan")
                                    status_tbl.add_column("Speed", style="green")
                                    status_tbl.add_column("Best Accuracy So Far", style="bold yellow")
                                    status_tbl.add_row(
                                        f"{completed_combos_count} / {total_combos} Combos ({completed_combos_count/total_combos*100:.1f}%)",
                                        f"{rate:.1f} combos/sec",
                                        f"{best_acc_so_far:.2f}% (Combo #{best_cid_so_far})"
                                    )
                                    live.update(status_tbl, refresh=True)
                    except Exception as exc:
                        console.print(f"[bold red]Error on Combo #{combo_id} ({video_name}): {exc}[/bold red]")

    # Step 4: Output Best Parameters, Leaderboard & Visualization Graphs
    generate_visualizations_and_report(final_accuracy_csv, MINE_OUTPUT_DIR, param_keys)
# sweep_yolo_sota.py – Asynchronous Multi-Variable Hyperparameter Sweep
# ==============================================================================
# Features:
#   1. Extracts & caches True Attendance for the 4 hardcoded videos in MINE/
#   2. No video encoding / bounding box drawing / DEBUG CSV creation (fast & zero bloat)
#   3. Asynchronous task queue across (parameter_combo x video) — no blocking on slow videos
#   4. Instant evaluation + temp output.csv deletion + live append to sweep_final_accuracy.csv
#   5. Generates multi-panel matplotlib visualizations & Rich leaderboard at completion
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
# 1. HARDCODED 4 VIDEOS & SWEEP CONFIGURATION
# ==========================================
# Best practice: Pick 2 sessions (cam1 + cam2 for 2 different dates = 4 videos)
# so the cam1+cam2 union logic matches process_attendance.py 100%.
SWEEP_VIDEOS = [
    "20260807_110016_cam1_raw.mp4",
    "20260807_110016_cam2_raw.mp4",
    "20260910_100327_cam1_raw.mp4",
    "20260910_100327_cam2_raw.mp4",
]

# Define the multi-variable grid to sweep.
# (Adjust lists below to add/remove values; total combos = product of list lengths)
SWEEP_GRID = {
    "CONFIDENCE_THRESHOLD":   [0.35, 0.39, 0.43],
    "NORM_QUALITY_THRESHOLD": [6.0, 7.0, 8.5],
    "EMA_MOMENTUM":           [0.6, 0.75],
    "FRAMES_PER_VOTE":        [5],
    "SOFTMAX_TAU":            [5.0],
    "MIN_BOX_SIZE":           [60],
    "MIN_TOTAL_FRAMES":       [45],
    "MIN_TOTAL_SAMPLES":      [25, 35],
    "MIN_SAMPLE_RATIO":       [0.28, 0.35],
    "MIN_WIN_RATIO":          [0.48, 0.55],
}

INPUT_VIDEOS_DIR = "VIDEOS"
RESULTS_BASE_DIR = "ATTENDENCE RESULTS/Results"
MINE_OUTPUT_DIR = os.path.abspath("ATTENDENCE RESULTS/MINE")
MAX_WORKERS = 4
FRAME_SKIP = 1


# ==========================================
# 2. TRUE ATTENDANCE CACHING & EVALUATION
# ==========================================
def prepare_subset_true_attendance(sweep_videos, results_base_dir, mine_dir):
    """
    Creates (or loads if already existing) the true attendance CSV for the 4 hardcoded
    videos inside the MINE folder, along with the short_name -> Roll_Name dictionary.
    """
    os.makedirs(mine_dir, exist_ok=True)
    subset_true_path = os.path.join(mine_dir, "true_attendance_4videos.csv")
    full_true_path = os.path.join(results_base_dir, "filtered_true_attendance.csv")
    name_mapping_path = os.path.join(results_base_dir, "name_mapping.csv")

    if not os.path.exists(name_mapping_path):
        raise FileNotFoundError(f"Could not find name_mapping.csv at {name_mapping_path}")

    # Load short_to_roll mapping
    name_map_df = pd.read_csv(name_mapping_path).dropna(subset=['Short_Name', 'Roll_Name'])
    name_map_df = name_map_df[name_map_df['Short_Name'].astype(str).str.strip() != '']
    short_to_roll = dict(zip(
        name_map_df['Short_Name'].astype(str).str.strip(),
        name_map_df['Roll_Name'].astype(str).str.strip()
    ))

    # Extract unique dates (YYYYMMDD) from the 4 hardcoded videos
    required_dates = sorted(list({v.split('_')[0] for v in sweep_videos}))

    need_create = True
    if os.path.exists(subset_true_path):
        existing_df = pd.read_csv(subset_true_path)
        existing_cols = [c.strip().replace('-', '') for c in existing_df.columns]
        if all(d in existing_cols for d in required_dates):
            console.print(f"[bold green]✔ Subset True Attendance already exists at:[/bold green] {subset_true_path} (Skipping creation)")
            need_create = False

    if need_create:
        if not os.path.exists(full_true_path):
            raise FileNotFoundError(f"Could not find filtered_true_attendance.csv at {full_true_path}")

        console.print(f"[bold yellow]⚙ Creating Subset True Attendance for dates {required_dates} in MINE folder...[/bold yellow]")
        full_df = pd.read_csv(full_true_path)
        clean_cols = [str(c).strip().replace('-', '') for c in full_df.columns]
        full_df.columns = clean_cols

        # Keep first two columns (Roll No, Name) + columns matching our 4 videos' dates
        base_cols = list(full_df.columns[:2])
        date_cols = [d for d in required_dates if d in full_df.columns]
        missing_dates = [d for d in required_dates if d not in full_df.columns]
        if missing_dates:
            console.print(f"[bold red]Warning:[/bold red] Dates {missing_dates} not found in filtered_true_attendance.csv!")

        subset_df = full_df[base_cols + date_cols].copy()
        subset_df.to_csv(subset_true_path, index=False)
        console.print(f"[bold green]✔ Saved Subset True Attendance to:[/bold green] {subset_true_path}")

    # Load and format subset_true_df for fast in-memory accuracy lookup
    true_df = pd.read_csv(subset_true_path)
    roll_col = true_df.iloc[:, 0].astype(str).str.strip()
    true_df = true_df.set_index(roll_col)
    true_df.columns = true_df.columns.astype(str).str.strip().str.replace('-', '')
    for col in true_df.columns:
        true_df[col] = true_df[col].astype(str).str.strip()

    return true_df, short_to_roll


def evaluate_combo_and_cleanup(combo_id, params, csv_file_map, true_df, short_to_roll):
    """
    Combines the 4 output CSVs for a completed parameter combination, computes the
    overall average accuracy against true_df, and immediately deletes the 4 output CSVs.
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
# 4. WORKER INITIALIZATION & FAST INFERENCE
# ==========================================
yolo_model = None
resnet = None
ref_embeddings_tensor = None
target_names = []
y_real = []
device = None
use_half = False


def init_worker():
    global yolo_model, resnet, ref_embeddings_tensor, target_names, y_real, device, use_half

    cv2.setNumThreads(1)
    time.sleep(random.uniform(0.1, 1.5))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_half = torch.cuda.is_available()

    yolo_model = load_yolo_face(device)
    resnet = load_adaface_model(device, use_half=use_half)

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

    ref_embeddings_tensor = torch.from_numpy(ref_embeddings).to(device).float()
    ref_embeddings_tensor = F.normalize(ref_embeddings_tensor, p=2, dim=1)
    if use_half:
        ref_embeddings_tensor = ref_embeddings_tensor.half()

    meta_path = './face_attendance_sota_meta.pkl'
    with open(meta_path, 'rb') as f:
        saved_data = pickle.load(f)
    target_names, y_real = saved_data['target_names'], saved_data['y_real']


def save_temp_output_csv(output_csv_path, archived_tracks, active_track_memory, target_names, params):
    """Saves ONLY the lightweight output.csv (no debug CSV, no video file)."""
    final_mem = {**archived_tracks, **active_track_memory}
    student_presence = {name: False for name in target_names}
    student_detection_count = {name: 0 for name in target_names}

    min_total_frames = params["MIN_TOTAL_FRAMES"]
    min_total_samples = params["MIN_TOTAL_SAMPLES"]
    min_sample_ratio = params["MIN_SAMPLE_RATIO"]
    min_win_ratio = params["MIN_WIN_RATIO"]

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
            sample_ratio = counts.get(winner, 0) / total_samples if total_samples > 0 else 0.0

            if (total_frames >= min_total_frames and
                total_samples >= min_total_samples and
                sample_ratio >= min_sample_ratio and
                win_ratio >= min_win_ratio):
                if winner != "Unknown":
                    student_presence[winner] = True
                    student_detection_count[winner] += counts.get(winner, 0)

    output_data = [
        {
            'Name': s,
            'Status': 'Present' if student_presence[s] else 'Absent',
            'Detection Count': student_detection_count[s]
        }
        for s in target_names
    ]
    pd.DataFrame(output_data).to_csv(output_csv_path, index=False)


def process_sweep_task(combo_id, params, video_filename, input_dir, output_dir, progress_q):
    """Runs a single video for a given parameter combination without saving video or debug files."""
    global yolo_model, resnet, ref_embeddings_tensor, target_names, y_real, device, use_half

    video_path = os.path.join(input_dir, video_filename)
    video_stem = os.path.splitext(video_filename)[0]
    temp_output_csv = os.path.join(output_dir, f"temp_c{combo_id:04d}_{video_stem}_output.csv")
    task_key = f"[C{combo_id:03d}] {video_filename}"

    # Reset YOLO tracker state cleanly
    if hasattr(yolo_model, 'predictor') and yolo_model.predictor is not None:
        if hasattr(yolo_model.predictor, 'trackers') and yolo_model.predictor.trackers:
            for tracker in yolo_model.predictor.trackers:
                tracker.reset()

    video_stream = ThreadedVideoReader(video_path).start()
    total_frames = max(video_stream.total_frames, 1)

    progress_q.put({"type": "start", "task": task_key, "total_frames": total_frames})

    conf_thresh = params["CONFIDENCE_THRESHOLD"]
    norm_thresh = params["NORM_QUALITY_THRESHOLD"]
    ema_momentum = params["EMA_MOMENTUM"]
    frames_per_vote = params["FRAMES_PER_VOTE"]
    softmax_tau = params["SOFTMAX_TAU"]
    min_box_size = params["MIN_BOX_SIZE"]

    local_id_counter = 0
    raw_to_local_id = {}
    active_track_memory, archived_tracks = {}, {}
    frame_count = 0
    t_start = time.perf_counter()

    try:
        while video_stream.more():
            frame = video_stream.read()
            if frame is None:
                break

            results = yolo_model.track(
                frame,
                persist=True,
                tracker=OCSORT_TRACKER_PATH,
                verbose=False,
                half=use_half,
                imgsz=640
            )
            has_detections = results[0].boxes.id is not None
            local_ids_list = []

            if has_detections and ref_embeddings_tensor is not None:
                boxes = results[0].boxes.xyxy.to(device)
                raw_ids = results[0].boxes.id.to(device).int().cpu().tolist()

                for r_id in raw_ids:
                    if r_id not in raw_to_local_id:
                        local_id_counter += 1
                        raw_to_local_id[r_id] = local_id_counter

                    t_id = raw_to_local_id[r_id]
                    local_ids_list.append(t_id)

                    if t_id not in active_track_memory:
                        active_track_memory[t_id] = {
                            'frames_alive': 0,
                            'all_preds': [],
                            'missing_frames': 0,
                            'crop_buffer': [],
                            'ema_emb': None
                        }
                    active_track_memory[t_id]['frames_alive'] += 1

                if frame_count % FRAME_SKIP == 0:
                    batch_tensors, batch_track_ids = [], []

                    box_w = boxes[:, 2] - boxes[:, 0]
                    box_h = boxes[:, 3] - boxes[:, 1]
                    aspect_ratios = box_w / (box_h + 1e-6)

                    valid_mask = (
                        (box_w >= min_box_size) &
                        (box_h >= min_box_size) &
                        (aspect_ratios >= 0.7) &
                        (aspect_ratios <= 1.7)
                    )

                    if valid_mask.any():
                        valid_boxes = boxes[valid_mask].clone()
                        valid_mask_cpu = valid_mask.cpu().numpy()
                        valid_local_ids = [local_ids_list[k] for k in range(len(local_ids_list)) if valid_mask_cpu[k]]

                        frame_tensor_chw = (
                            torch.from_numpy(frame)
                            .to(device, non_blocking=True)
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .contiguous()
                            .float()
                        )
                        crops = extract_faces_112(
                            frame_tensor_chw, results, valid_mask, valid_boxes,
                            ARCFACE_PINV, frame.shape[0], frame.shape[1], device
                        )
                        del frame_tensor_chw

                        for i, t_id in enumerate(valid_local_ids):
                            active_track_memory[t_id]['crop_buffer'].append(crops[i:i+1])
                            if len(active_track_memory[t_id]['crop_buffer']) >= frames_per_vote:
                                batch_tensors.extend(active_track_memory[t_id]['crop_buffer'])
                                batch_track_ids.extend([t_id] * len(active_track_memory[t_id]['crop_buffer']))
                                active_track_memory[t_id]['crop_buffer'] = []

                    if batch_tensors:
                        with torch.inference_mode():
                            batch_tensor = torch.cat(batch_tensors, dim=0)
                            batch_tensor = (batch_tensor / 127.5) - 1.0
                            if use_half:
                                batch_tensor = batch_tensor.half()

                            normed_embeddings, feature_norms = resnet(batch_tensor)
                            norms_flat = feature_norms.squeeze(1)
                            quality_mask_list = (norms_flat >= norm_thresh).cpu().tolist()

                            track_emb_groups = {}
                            for k in range(len(batch_track_ids)):
                                if quality_mask_list[k]:
                                    t_id = batch_track_ids[k]
                                    if t_id not in track_emb_groups:
                                        track_emb_groups[t_id] = {'embs': [], 'norms': []}
                                    track_emb_groups[t_id]['embs'].append(normed_embeddings[k:k+1])
                                    track_emb_groups[t_id]['norms'].append(norms_flat[k:k+1])

                            for t_id, group in track_emb_groups.items():
                                e_track = torch.cat(group['embs'], dim=0)
                                n_track = torch.cat(group['norms'], dim=0).unsqueeze(1)
                                K = e_track.size(0)

                                weights = torch.softmax(n_track / softmax_tau, dim=0)
                                e_win = F.normalize(torch.sum(weights * e_track, dim=0, keepdim=True), p=2, dim=1)

                                if active_track_memory[t_id]['ema_emb'] is None:
                                    ema_emb = e_win
                                else:
                                    prev_ema = active_track_memory[t_id]['ema_emb']
                                    ema_emb = F.normalize(
                                        ema_momentum * prev_ema + (1.0 - ema_momentum) * e_win,
                                        p=2, dim=1
                                    )
                                active_track_memory[t_id]['ema_emb'] = ema_emb

                                sim_matrix = torch.mm(ema_emb.to(ref_embeddings_tensor.dtype), ref_embeddings_tensor.t())
                                max_sim, max_idx = torch.max(sim_matrix, dim=1)
                                conf_score = max_sim.item()
                                winner = target_names[y_real[max_idx.item()]] if conf_score > conf_thresh else "Unknown"

                                active_track_memory[t_id]['all_preds'].extend([winner] * K)

            alive_ids = set(local_ids_list) if has_detections else set()
            for t_id in list(active_track_memory.keys()):
                if t_id not in alive_ids:
                    active_track_memory[t_id]['missing_frames'] += 1
                    if active_track_memory[t_id]['missing_frames'] > 50:
                        archived_tracks[t_id] = active_track_memory.pop(t_id)
                else:
                    active_track_memory[t_id]['missing_frames'] = 0

            frame_count += 1
            if frame_count % 20 == 0:
                elapsed = time.perf_counter() - t_start
                curr_fps = frame_count / elapsed if elapsed > 0 else 0.0
                progress_q.put({
                    "type": "progress",
                    "task": task_key,
                    "frame": frame_count,
                    "total": total_frames,
                    "fps": curr_fps
                })

        save_temp_output_csv(temp_output_csv, archived_tracks, active_track_memory, target_names, params)
        progress_q.put({"type": "finish_video", "task": task_key})
        return combo_id, video_filename, temp_output_csv, "OK"

    except Exception as e:
        if os.path.exists(temp_output_csv):
            try:
                os.remove(temp_output_csv)
            except OSError:
                pass
        progress_q.put({"type": "error", "task": task_key, "err": str(e)})
        return combo_id, video_filename, "", f"Error: {str(e)}"
    finally:
        video_stream.stop()
        del active_track_memory, archived_tracks, raw_to_local_id
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==========================================
# 5. LIVE TERMINAL DASHBOARD
# ==========================================
def run_sweep_dashboard(progress_q, total_videos, total_combos, stop_event):
    active_tasks = {}
    completed_videos = 0
    completed_combos = 0
    best_acc = 0.0
    best_combo_id = None
    last_acc_str = "N/A"

    with Live(auto_refresh=False, refresh_per_second=4) as live:
        while not stop_event.is_set() or not progress_q.empty():
            while True:
                try:
                    msg = progress_q.get_nowait()
                    mtype = msg.get("type")
                    tkey = msg.get("task")

                    if mtype == "start":
                        active_tasks[tkey] = {"frame": 0, "total": msg.get("total_frames", 1), "fps": 0.0}
                    elif mtype == "progress":
                        if tkey in active_tasks:
                            active_tasks[tkey]["frame"] = msg["frame"]
                            active_tasks[tkey]["total"] = msg["total"]
                            active_tasks[tkey]["fps"] = msg["fps"]
                    elif mtype == "finish_video":
                        active_tasks.pop(tkey, None)
                        completed_videos += 1
                    elif mtype == "combo_done":
                        completed_combos += 1
                        acc = msg["acc"]
                        cid = msg["combo_id"]
                        last_acc_str = f"Combo #{cid}: {acc:.2f}%"
                        if acc > best_acc:
                            best_acc = acc
                            best_combo_id = cid
                    elif mtype == "error":
                        active_tasks.pop(tkey, None)
                except queue.Empty:
                    break

            best_str = f"Best So Far: [bold yellow]{best_acc:.2f}% (Combo #{best_combo_id})[/bold yellow]" if best_combo_id is not None else "Best So Far: -"
            title_str = (
                f"[bold green]Async Sweep Progress[/bold green] | "
                f"Videos: {completed_videos}/{total_videos} | "
                f"Combos Evaluated: {completed_combos}/{total_combos} | "
                f"Latest: {last_acc_str} | {best_str}"
            )

            table = Table(title=title_str, box=None)
            table.add_column("Active Task ([Combo] Video)", style="cyan", width=42, no_wrap=True)
            table.add_column("Frames", style="magenta", width=18, justify="right")
            table.add_column("Completion", style="yellow", width=12, justify="right")
            table.add_column("FPS", style="green", width=10, justify="right")

            if active_tasks:
                for tname, stat in sorted(active_tasks.items()):
                    pct = min(100.0, (stat["frame"] / stat["total"]) * 100.0) if stat["total"] > 0 else 0.0
                    table.add_row(tname, f"{stat['frame']}/{stat['total']}", f"{pct:5.1f}%", f"{stat['fps']:5.1f}")
            else:
                table.add_row("[italic dim]Waiting for workers...[/italic dim]", "-", "-", "-")

            live.update(table, refresh=True)
            time.sleep(0.2)


# ==========================================
# 6. VISUALIZATION & REPORTING
# ==========================================
def generate_visualizations_and_report(final_csv_path, output_dir, param_keys):
    if not os.path.exists(final_csv_path):
        console.print("[bold red]No sweep results found to visualize![/bold red]")
        return

    df = pd.read_csv(final_csv_path)
    if df.empty:
        return

    df_sorted = df.sort_values(by="Overall_Accuracy", ascending=False).reset_index(drop=True)
    best_row = df_sorted.iloc[0]

    # 1. Print Top 10 Leaderboard in Terminal
    leader_table = Table(title="[bold green]🏆 TOP 10 HYPERPARAMETER COMBINATIONS[/bold green]")
    leader_table.add_column("Rank", justify="center", style="bold yellow")
    leader_table.add_column("Combo ID", justify="center", style="cyan")
    for k in param_keys:
        if df[k].nunique() > 1:
            leader_table.add_column(k, justify="right")
    leader_table.add_column("TP / TN / FP / FN", justify="center", style="magenta")
    leader_table.add_column("Overall Accuracy", justify="right", style="bold green")

    varied_keys = [k for k in param_keys if df[k].nunique() > 1]
    for rank, (_, row) in enumerate(df_sorted.head(10).iterrows(), start=1):
        row_vals = [str(rank), f"#{int(row['Combo_ID'])}"]
        for k in varied_keys:
            row_vals.append(str(row[k]))
        row_vals.append(f"{int(row['TP'])} / {int(row['TN'])} / {int(row['FP'])} / {int(row['FN'])}")
        row_vals.append(f"{row['Overall_Accuracy']:.2f}%")
        leader_table.add_row(*row_vals)

    console.print("\n")
    console.print(leader_table)

    # Print Best Parameters Summary Panel
    best_params_str = "\n".join([f"  • [bold cyan]{k}[/bold cyan]: {best_row[k]}" for k in param_keys])
    console.print(Panel(
        f"[bold green]Best Overall Accuracy: {best_row['Overall_Accuracy']:.2f}% (Combo #{int(best_row['Combo_ID'])})[/bold green]\n\n"
        f"[bold white]Optimal Parameters to generalize across all videos:[/bold white]\n{best_params_str}",
        title="✨ Optimal Sweep Conclusion",
        expand=False
    ))

    # 2. Create Multi-Panel Visualization Figure
    fig = plt.figure(figsize=(18, 12), dpi=150)
    fig.suptitle(
        f"Hyperparameter Sweep Analysis (4-Video Benchmark) — Best Accuracy: {best_row['Overall_Accuracy']:.2f}%",
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
        min_x = max(0.0, top_n["Overall_Accuracy"].min() - 5.0)
        max_x = min(100.0, top_n["Overall_Accuracy"].max() + 2.0)
        ax1.set_xlim(min_x, max_x)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.15, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va='center', fontsize=9, fontweight='bold')

    # Panel B: 2D Heatmap of Top 2 Varied Parameters
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
                    ax2.text(j, i, f"{val:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        plt.colorbar(im, ax=ax2, label="Max Accuracy (%)")
    else:
        ax2.text(0.5, 0.5, "Need >= 2 varied parameters for 2D Heatmap", ha="center", va="center")

    # Panel C: Parameter Sensitivity Curves (Mean & Max per Parameter Value)
    ax3 = plt.subplot(2, 1, 2)
    if varied_keys:
        x_positions = []
        x_labels = []
        mean_vals = []
        max_vals = []
        group_boundaries = []
        pos = 0
        for k in varied_keys:
            grouped = df.groupby(k)["Overall_Accuracy"].agg(["mean", "max"]).reset_index()
            for _, grow in grouped.iterrows():
                x_positions.append(pos)
                x_labels.append(f"{k.replace('_THRESHOLD', '').replace('MIN_', '')}\n={grow[k]}")
                mean_vals.append(grow["mean"])
                max_vals.append(grow["max"])
                pos += 1
            group_boundaries.append(pos - 0.5)

        width = 0.35
        x_arr = np.array(x_positions)
        ax3.bar(x_arr - width/2, mean_vals, width=width, label="Mean Accuracy (%)", color="#3498db")
        ax3.bar(x_arr + width/2, max_vals, width=width, label="Max Accuracy (%)", color="#e67e22")
        for b in group_boundaries[:-1]:
            ax3.axvline(b, color="gray", linestyle="--", alpha=0.5)
        ax3.set_xticks(x_arr)
        ax3.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=9)
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_title("Hyperparameter Sensitivity (Mean vs. Max Accuracy across Sweep)", fontweight="bold")
        if mean_vals:
            ax3.set_ylim(max(0.0, min(mean_vals) - 5.0), min(100.0, max(max_vals) + 2.0))
        ax3.legend(loc="lower right")
        ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plot_path = os.path.join(output_dir, "sweep_visualization.png")
    plt.savefig(plot_path)
    plt.close(fig)
    console.print(f"[bold green]📊 Saved Sweep Visualization Chart to:[/bold green] {plot_path}")


# ==========================================
# 7. MAIN ASYNCHRONOUS ORCHESTRATOR
# ==========================================
if __name__ == '__main__':
    mp.freeze_support()
    ensure_ocsort_tracker()
    ctx = mp.get_context('spawn')

    # Verify the 4 hardcoded videos exist
    missing_vids = [v for v in SWEEP_VIDEOS if not os.path.exists(os.path.join(INPUT_VIDEOS_DIR, v))]
    if missing_vids:
        raise FileNotFoundError(f"Missing hardcoded videos in '{INPUT_VIDEOS_DIR}': {missing_vids}")

    # Step 1: Prepare or load cached True Attendance for the 4 videos in MINE/
    true_df, short_to_roll = prepare_subset_true_attendance(SWEEP_VIDEOS, RESULTS_BASE_DIR, MINE_OUTPUT_DIR)

    # Step 2: Build parameter combinations
    param_keys = list(SWEEP_GRID.keys())
    param_values = [SWEEP_GRID[k] for k in param_keys]
    all_combos = [
        dict(zip(param_keys, prod))
        for prod in itertools.product(*param_values)
    ]

    final_accuracy_csv = os.path.join(MINE_OUTPUT_DIR, "sweep_final_accuracy.csv")

    # Optional resume check: skip parameter combinations already recorded in sweep_final_accuracy.csv
    completed_combo_ids = set()
    if os.path.exists(final_accuracy_csv):
        try:
            existing_res = pd.read_csv(final_accuracy_csv)
            if "Combo_ID" in existing_res.columns:
                completed_combo_ids = set(existing_res["Combo_ID"].astype(int).tolist())
                if completed_combo_ids:
                    console.print(f"[bold cyan]↻ Found {len(completed_combo_ids)} already-evaluated combinations in {final_accuracy_csv}. Resuming remaining...[/bold cyan]")
        except Exception:
            pass

    pending_combos = [
        (idx + 1, params)
        for idx, params in enumerate(all_combos)
        if (idx + 1) not in completed_combo_ids
    ]

    total_combos = len(pending_combos)
    total_video_runs = total_combos * len(SWEEP_VIDEOS)
    console.print(f"[bold white]🚀 Launching Asynchronous Sweep:[/bold white] {total_combos} parameter combos × {len(SWEEP_VIDEOS)} videos = {total_video_runs} video runs.")

    if pending_combos:
        manager = ctx.Manager()
        progress_q = manager.Queue()
        stop_event = threading.Event()

        dash_thread = threading.Thread(
            target=run_sweep_dashboard,
            args=(progress_q, total_video_runs, total_combos, stop_event),
            daemon=True
        )
        dash_thread.start()

        # Track completed videos per combo_id asynchronously
        combo_params_lookup = {cid: p for cid, p in pending_combos}
        combo_completed_csvs = {cid: {} for cid, _ in pending_combos}

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            mp_context=ctx,
            initializer=init_worker
        ) as executor:
            # Submit all (combo_id, video) pairs so workers never wait on slow videos
            futures = {}
            for combo_id, params in pending_combos:
                for video_name in SWEEP_VIDEOS:
                    fut = executor.submit(
                        process_sweep_task,
                        combo_id, params, video_name,
                        INPUT_VIDEOS_DIR, MINE_OUTPUT_DIR, progress_q
                    )
                    futures[fut] = (combo_id, video_name)

            # Asynchronously process each video completion as soon as it finishes
            for fut in concurrent.futures.as_completed(futures):
                combo_id, video_name = futures[fut]
                try:
                    cid, vname, csv_path, status = fut.result()
                    if status == "OK" and csv_path:
                        combo_completed_csvs[cid][vname] = csv_path

                        # The instant all 4 videos for this parameter combo finish:
                        if len(combo_completed_csvs[cid]) == len(SWEEP_VIDEOS):
                            row_result = evaluate_combo_and_cleanup(
                                cid,
                                combo_params_lookup[cid],
                                combo_completed_csvs[cid],
                                true_df,
                                short_to_roll
                            )
                            # Append immediately to sweep_final_accuracy.csv
                            row_df = pd.DataFrame([row_result])
                            write_header = not os.path.exists(final_accuracy_csv)
                            row_df.to_csv(final_accuracy_csv, mode='a', header=write_header, index=False)

                            progress_q.put({
                                "type": "combo_done",
                                "combo_id": cid,
                                "acc": row_result["Overall_Accuracy"]
                            })
                            del combo_completed_csvs[cid]
                except Exception as exc:
                    pass

        stop_event.set()
        dash_thread.join(timeout=1.0)

    # Step 3: Output Best Parameters, Leaderboard & Visualization Graphs
    generate_visualizations_and_report(final_accuracy_csv, MINE_OUTPUT_DIR, param_keys)
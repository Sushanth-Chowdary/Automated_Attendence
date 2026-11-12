# test_yolo_sota.py  –  SOTA Multi-Process Batch Video Evaluation & Rich Dashboard
# ==============================================================================
# Upgrades & Synchronizations with sweep_yolo_sota.py:
#   1. Exact Window-Vote Synchronization: 1 window (FRAMES_PER_VOTE) = 1 vote
#      (fixes the K-vote inflation bug & pre-window norm filtering drift).
#   2. Fixed Softmax Quality Masking: frames failing NORM_QUALITY_THRESHOLD are
#      masked with -1e4 before Softmax so blurry frames get 0.0 weight.
#   3. Exact Per-Class Top-2 Margin Filtering (scatter_reduce amax over classes).
#   4. Track-Level Quality-Weighted Centroid Verification (Post-Confirmation Gate).
#   5. Continuous Confidence-Weighted Track Scoring & High-Certainty Single-Track
#      Rescue (awards +1 bonus Detection Count for rock-solid single tracks, and
#      logs Continuous_Score + Centroid_Sim in *_DEBUG_Tracks.csv & *_output.csv).
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
from collections import Counter
import concurrent.futures
import multiprocessing as mp

import cv2
import faiss
import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from ultralytics import YOLO

from rich.live import Live
from rich.table import Table

warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================
# 1. HYPERPARAMETERS (Synced with sweep_yolo_sota.py)
# ==========================================
CONFIDENCE_THRESHOLD   = 0.4125
NORM_QUALITY_THRESHOLD = 7.0
EMA_MOMENTUM           = 0.708
FRAMES_PER_VOTE        = 4
SOFTMAX_TAU            = 5.0
MIN_BOX_SIZE           = 50.0
MIN_TOTAL_FRAMES       = 30
MIN_TOTAL_SAMPLES      = 16
MIN_SAMPLE_RATIO       = 0.3222
MIN_WIN_RATIO          = 0.3604
MARGIN_THRESHOLD       = 0.015
CENTROID_THRESHOLD     = 0.380
FRAME_SKIP             = 1


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


@torch.no_grad()
def compute_per_class_similarities(query_emb, ref_embs_t, y_real_t, n_classes):
    """
    Computes exact max cosine similarity per distinct student class for query_emb (1, 512).
    Returns a 1D tensor of shape (n_classes,).
    """
    sim = torch.mm(query_emb.to(ref_embs_t.dtype), ref_embs_t)  # (1, G)
    class_sims = torch.full((1, n_classes), -2.0, dtype=sim.dtype, device=sim.device)
    class_sims.scatter_reduce_(1, y_real_t.unsqueeze(0), sim, reduce='amax', include_self=True)
    return class_sims.squeeze(0)


@torch.no_grad()
def save_attendance_results(video_filename, archived_tracks, active_track_memory, target_names,
                            ref_embs_t, y_real_t, name_to_cls_idx, output_dir):
    final_mem = {**archived_tracks, **active_track_memory}
    debug_data = []
    student_presence = {name: False for name in target_names}
    student_detection_count = {name: 0 for name in target_names}
    student_continuous_score = {name: 0.0 for name in target_names}
    n_classes = len(target_names)

    for t_id, data in sorted(final_mem.items(), key=lambda x: x[0]):
        total_frames = data.get('frames_alive', 0)
        all_preds = data['all_preds']
        valid_preds = [p for p in all_preds if p != "Unknown"]
        valid_votes_count = len(valid_preds)
        total_samples = len(all_preds)

        centroid_sim = 0.0
        continuous_score = 0.0
        track_weight = 0

        if valid_votes_count > 0:
            counts = Counter(valid_preds)
            winner, best_votes = counts.most_common(1)[0]
            win_ratio = best_votes / valid_votes_count
            sample_ratio = best_votes / total_samples if total_samples > 0 else 0.0

            # Upgrade 3: Track-Level Centroid Matching (Post-Confirmation Gate)
            if data.get('centroid_accum') is not None and winner in name_to_cls_idx:
                win_cls_idx = name_to_cls_idx[winner]
                cent_emb = F.normalize(data['centroid_accum'], p=2, dim=1)
                class_sims = compute_per_class_similarities(cent_emb, ref_embs_t, y_real_t, n_classes)
                centroid_sim = float(class_sims[win_cls_idx].item())

            passed_stage1 = (
                total_frames >= MIN_TOTAL_FRAMES and
                total_samples >= MIN_TOTAL_SAMPLES and
                sample_ratio >= MIN_SAMPLE_RATIO and
                win_ratio >= MIN_WIN_RATIO
            )
            passed_centroid = centroid_sim >= CENTROID_THRESHOLD

            if passed_stage1 and passed_centroid:
                status = "Passed"
                continuous_score = round(sample_ratio * win_ratio * max(0.0, centroid_sim), 4)
                # Upgrade 2: High-Certainty Single-Track Rescue (+1 bonus detection count)
                is_high_cert = (
                    best_votes >= 2.5 * MIN_TOTAL_SAMPLES and
                    win_ratio >= 0.75 and
                    centroid_sim >= (CONFIDENCE_THRESHOLD + 0.04)
                )
                track_weight = 2 if is_high_cert else 1
            elif passed_stage1 and not passed_centroid:
                status = f"Failed (Centroid {centroid_sim:.3f} < {CENTROID_THRESHOLD})"
            else:
                status = "Failed"
        else:
            winner = "Unknown"
            status = "Failed"
            win_ratio = 0.0
            sample_ratio = 0.0

        if status == "Passed" and winner != "Unknown":
            student_presence[winner] = True
            student_detection_count[winner] += track_weight
            student_continuous_score[winner] = round(student_continuous_score[winner] + continuous_score, 4)

        debug_data.append({
            'Track ID': t_id,
            'Start Time': data.get('start_time', ''),
            'Total Frames': total_frames,
            'Valid Votes': valid_votes_count,
            'Total Windows (Samples)': total_samples,
            'Sample Ratio': round(sample_ratio, 4),
            'Win Ratio': round(win_ratio, 4),
            'Centroid Sim': round(centroid_sim, 4),
            'Continuous Score': continuous_score,
            'Detection Weight': track_weight,
            'Predicted Identity': winner,
            'Gate Status': status,
            'Breakdown': dict(Counter(all_preds))
        })

    stem = os.path.splitext(video_filename)[0]
    pd.DataFrame(debug_data).to_csv(os.path.join(output_dir, f"{stem}_DEBUG_Tracks.csv"), index=False)

    output_data = [{
        'Name': s,
        'Status': 'Present' if student_presence[s] else 'Absent',
        'Detection Count': student_detection_count[s],
        'Continuous_Score': student_continuous_score[s]
    } for s in target_names]
    pd.DataFrame(output_data).to_csv(os.path.join(output_dir, f"{stem}_output.csv"), index=False)


# ==========================================
# ADAFACE IRESNET-50 ARCHITECTURE
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
        blocks = [
            (64, 64, 2, 3),
            (64, 128, 2, 4),
            (128, 256, 2, 14),
            (256, 512, 2, 3)
        ]
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
    weight_candidates = [
        'adaface_ir50_webface4m.ckpt',
        'adaface_ir50_webface4m.pth',
        'adaface_ir50_webface4m.pt',
    ]
    loaded = False
    for wf in weight_candidates:
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
            loaded = True
            break
    if not loaded:
        raise FileNotFoundError("AdaFace checkpoint 'adaface_ir50_webface4m.ckpt' not found!")
    model.eval().to(device)
    if use_half:
        model = model.half()
    return model


# ==========================================
# YOLO MODEL LOADER & TRACKER CONFIG
# ==========================================
def load_yolo_face(device, use_engine=False):
    candidates = []
    if use_engine:
        candidates.extend([
            ('yolo26n-face.engine', 'detect'),
            ('yolov8n-face.engine', 'detect'),
        ])
    candidates.extend([
        ('yolo26n-face.pt', 'detect'),
        ('yolov8n-face.pt', 'detect'),
    ])
    for model_path, task in candidates:
        if os.path.exists(model_path):
            model = YOLO(model_path, task=task)
            if not model_path.endswith('.engine'):
                model.to(device)
            return model
    raise FileNotFoundError(f"No YOLO face model found! Tried: {[c[0] for c in candidates]}")


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


# ==========================================
# GPU 5-POINT ALIGNMENT UTILITIES (FLOAT32)
# ==========================================
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
# 2. Worker Lifecycle & Models
# ==========================================
yolo_model = None
resnet = None
ref_embeddings_t = None
target_names = []
y_real_t = None
name_to_cls_idx = {}
device = None
use_half = False


def init_worker():
    """Initializes models once per worker process to prevent redundant reload overhead."""
    global yolo_model, resnet, ref_embeddings_t, target_names, y_real_t, name_to_cls_idx, device, use_half

    cv2.setNumThreads(1)
    time.sleep(random.uniform(0.2, 2.5))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_half = torch.cuda.is_available()

    yolo_model = load_yolo_face(device, use_engine=False)
    resnet = load_adaface_model(device, use_half=use_half)

    faiss_index_path = './face_attendance_sota_faiss.bin'
    if os.path.exists(faiss_index_path):
        index = faiss.read_index(faiss_index_path)
        index.nprobe = 20
        try:
            ref_embeddings = index.reconstruct_n(0, index.ntotal)
        except AttributeError:
            index.make_direct_map()
            ref_embeddings = np.array([index.reconstruct(i) for i in range(index.ntotal)])

        ref_tensor = torch.from_numpy(ref_embeddings).to(device).float()
        ref_tensor = F.normalize(ref_tensor, p=2, dim=1)
        if use_half:
            ref_tensor = ref_tensor.half()
        ref_embeddings_t = ref_tensor.t().contiguous()  # (512, G)
    else:
        raise FileNotFoundError("Please run 'python train_yolo_sota.py' first to build './face_attendance_sota_faiss.bin'!")

    meta_path = './face_attendance_sota_meta.pkl'
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Please run 'python train_yolo_sota.py' first to build './face_attendance_sota_meta.pkl'!")
    with open(meta_path, 'rb') as f:
        saved_data = pickle.load(f)
    target_names = saved_data['target_names']
    y_real_t = torch.tensor(saved_data['y_real'], dtype=torch.int64, device=device)
    name_to_cls_idx = {name: idx for idx, name in enumerate(target_names)}


def process_video(video_filename, input_dir, output_dir, progress_q):
    """Processes a single video and pushes live telemetry through progress_q."""
    global yolo_model, resnet, ref_embeddings_t, target_names, y_real_t, name_to_cls_idx, device, use_half

    video_path = os.path.join(input_dir, video_filename)
    video_stem = os.path.splitext(video_filename)[0]
    output_csv = os.path.join(output_dir, f"{video_stem}_output.csv")
    debug_csv = os.path.join(output_dir, f"{video_stem}_DEBUG_Tracks.csv")
    final_video_path = os.path.join(output_dir, f"{video_stem}_output.mp4")

    if os.path.exists(output_csv) and os.path.exists(final_video_path) and os.path.getsize(final_video_path) > 1024:
        progress_q.put({"type": "skip", "file": video_filename})
        return video_filename, "Skipped"

    if hasattr(yolo_model, 'predictor') and yolo_model.predictor is not None:
        if hasattr(yolo_model.predictor, 'trackers') and yolo_model.predictor.trackers:
            for tracker in yolo_model.predictor.trackers:
                tracker.reset()

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

    local_id_counter = 0
    raw_to_local_id = {}

    active_track_memory, archived_tracks, track_identities = {}, {}, {}
    frame_count = 0
    t_start = time.perf_counter()
    processing_succeeded = False
    error_msg = None
    n_classes = len(target_names)

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

            boxes_cpu = []
            local_ids_list = []
            frame_names = []

            if has_detections and ref_embeddings_t is not None:
                boxes = results[0].boxes.xyxy.to(device)
                raw_ids = results[0].boxes.id.to(device).int().cpu().tolist()

                local_ids_list = []
                for r_id in raw_ids:
                    if r_id not in raw_to_local_id:
                        local_id_counter += 1
                        raw_to_local_id[r_id] = local_id_counter

                    t_id = raw_to_local_id[r_id]
                    local_ids_list.append(t_id)

                    if t_id not in active_track_memory:
                        active_track_memory[t_id] = {
                            'start_time': format_timestamp(frame_count, video_stream.fps),
                            'frames_alive': 0,
                            'all_preds': [],
                            'missing_frames': 0,
                            'crop_buffer': [],
                            'ema_emb': None,
                            'centroid_accum': None,
                        }
                    active_track_memory[t_id]['frames_alive'] += 1

                if frame_count % FRAME_SKIP == 0:
                    batch_tensors, batch_track_ids = [], []

                    box_w = boxes[:, 2] - boxes[:, 0]
                    box_h = boxes[:, 3] - boxes[:, 1]
                    aspect_ratios = box_w / (box_h + 1e-6)

                    valid_mask = (
                        (box_w >= MIN_BOX_SIZE) &
                        (box_h >= MIN_BOX_SIZE) &
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

                            if len(active_track_memory[t_id]['crop_buffer']) >= FRAMES_PER_VOTE:
                                batch_tensors.extend(active_track_memory[t_id]['crop_buffer'])
                                batch_track_ids.append(t_id)
                                active_track_memory[t_id]['crop_buffer'] = []

                    if batch_tensors:
                        with torch.inference_mode():
                            batch_tensor = torch.cat(batch_tensors, dim=0)  # (N_windows * fpv, 3, 112, 112)
                            batch_tensor = (batch_tensor / 127.5) - 1.0
                            if use_half:
                                batch_tensor = batch_tensor.half()

                            normed_embeddings, feature_norms = resnet(batch_tensor)
                            norms_flat = feature_norms.squeeze(1).float()  # (N_windows * fpv,)

                            # Process each window of FRAMES_PER_VOTE frames (exact match with sweep_yolo_sota.py)
                            for w_idx, t_id in enumerate(batch_track_ids):
                                s0 = w_idx * FRAMES_PER_VOTE
                                s1 = s0 + FRAMES_PER_VOTE
                                e_win_crops = normed_embeddings[s0:s1].float()  # (fpv, 512)
                                n_win_crops = norms_flat[s0:s1]                 # (fpv,)

                                # Bug A Fix: Mask out frames failing NORM_QUALITY_THRESHOLD with -1e4 before Softmax
                                frame_ok = n_win_crops >= NORM_QUALITY_THRESHOLD
                                valid_window = bool(frame_ok.any().item())

                                logits = torch.where(
                                    frame_ok,
                                    n_win_crops / SOFTMAX_TAU,
                                    torch.tensor(-1e4, device=device, dtype=torch.float32)
                                )
                                weights = torch.softmax(logits, dim=0).unsqueeze(1)  # (fpv, 1)
                                e_win = F.normalize(torch.sum(weights * e_win_crops, dim=0, keepdim=True), p=2, dim=1)

                                if valid_window:
                                    # Accumulate quality-weighted global track centroid
                                    raw_w = (n_win_crops * frame_ok.float()).unsqueeze(1)
                                    cent_step = torch.sum(raw_w * e_win_crops, dim=0, keepdim=True)
                                    if active_track_memory[t_id]['centroid_accum'] is None:
                                        active_track_memory[t_id]['centroid_accum'] = cent_step
                                    else:
                                        active_track_memory[t_id]['centroid_accum'] = (
                                            active_track_memory[t_id]['centroid_accum'] + cent_step
                                        )

                                    # EMA Update
                                    if active_track_memory[t_id]['ema_emb'] is None:
                                        ema_emb = e_win
                                    else:
                                        prev_ema = active_track_memory[t_id]['ema_emb']
                                        ema_emb = F.normalize(
                                            EMA_MOMENTUM * prev_ema + (1.0 - EMA_MOMENTUM) * e_win,
                                            p=2, dim=1
                                        )
                                    active_track_memory[t_id]['ema_emb'] = ema_emb

                                    # Upgrade 1: Exact Per-Class Top-2 Margin Filtering
                                    class_sims = compute_per_class_similarities(
                                        ema_emb, ref_embeddings_t, y_real_t, n_classes
                                    )
                                    top2_vals, top2_cls = class_sims.topk(2, dim=0)
                                    conf_score = float(top2_vals[0].item())
                                    margin_diff = float((top2_vals[0] - top2_vals[1]).item())
                                    win_cls = int(top2_cls[0].item())

                                    if conf_score >= CONFIDENCE_THRESHOLD and margin_diff >= MARGIN_THRESHOLD:
                                        winner = target_names[win_cls]
                                    else:
                                        winner = "Unknown"
                                else:
                                    conf_score = 0.0
                                    winner = "Unknown"

                                track_identities[t_id] = (winner, conf_score)
                                # Exact 1 vote per window (synchronized with sweep_yolo_sota.py)
                                active_track_memory[t_id]['all_preds'].append(winner)

                boxes_cpu = boxes.cpu().numpy()
                for i in range(len(local_ids_list)):
                    t_id = local_ids_list[i]
                    frame_names.append(track_identities.get(t_id, ("Analyzing...", 0.0)))

            write_queue.put((frame, boxes_cpu, local_ids_list, frame_names))

            alive_ids = set(local_ids_list) if has_detections else set()
            for t_id in list(active_track_memory.keys()):
                if t_id not in alive_ids:
                    active_track_memory[t_id]['missing_frames'] += 1
                    if active_track_memory[t_id]['missing_frames'] > 50:
                        archived_tracks[t_id] = active_track_memory.pop(t_id)
                else:
                    active_track_memory[t_id]['missing_frames'] = 0

            frame_count += 1

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

        if frame_count > 0:
            processing_succeeded = True

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return video_filename, error_msg
    finally:
        video_stream.stop()
        write_queue.put(None)
        writer_thread.join()

        if processing_succeeded:
            save_attendance_results(
                video_filename, archived_tracks, active_track_memory, target_names,
                ref_embeddings_t, y_real_t, name_to_cls_idx, output_dir
            )
            progress_q.put({"type": "finish", "file": video_filename})
        else:
            for p in (final_video_path, output_csv, debug_csv):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            progress_q.put({"type": "error", "file": video_filename, "err": error_msg or "Unknown Crash"})

        del active_track_memory, archived_tracks, track_identities, raw_to_local_id
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return video_filename, "Processed"


# ==========================================
# 3. Terminal Live Dashboard
# ==========================================
def run_live_dashboard(progress_q, total_videos, stop_event):
    active_videos = {}
    completed = 0
    failed = 0
    last_error = ""

    with Live(auto_refresh=False, refresh_per_second=4) as live:
        while not stop_event.is_set() or not progress_q.empty():
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
                    elif msg_type == "error":
                        active_videos.pop(filename, None)
                        failed += 1
                        last_error = f"{filename}: {msg.get('err', '')}"
                except queue.Empty:
                    break

            err_str = f" | [bold red]Failed: {failed} ({last_error})[/bold red]" if failed > 0 else ""
            table = Table(
                title=f"[bold green]Batch Processing Progress ({completed}/{total_videos} Finished)[/bold green]{err_str}",
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


# ==========================================
# 4. Main Entry Point
# ==========================================
if __name__ == '__main__':
    mp.freeze_support()
    ensure_ocsort_tracker()
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

    MAX_WORKERS = 4

    if target_videos:
        manager = ctx.Manager()
        progress_q = manager.Queue()
        stop_event = threading.Event()

        dashboard_thread = threading.Thread(
            target=run_live_dashboard,
            args=(progress_q, total_files, stop_event),
            daemon=True
        )
        dashboard_thread.start()

        errors_list = []

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
                    fname, status = future.result()
                    if status.startswith("Error:"):
                        errors_list.append((fname, status))
                except Exception as exc:
                    errors_list.append((video_filename, str(exc)))

        stop_event.set()
        dashboard_thread.join(timeout=1.0)

        if errors_list:
            print(f"\nCompleted with {len(errors_list)} errors:")
            for fname, err in errors_list[:10]:
                print(f"  -> [{fname}] {err}")
        else:
            print("\nAll videos processed successfully.")
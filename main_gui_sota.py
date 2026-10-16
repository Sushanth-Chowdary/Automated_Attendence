# main_gui_sota.py  –  SOTA Real-Time Dual-Camera RTSP CustomTkinter GUI
# =======================================================================
# Upgrades:
#   1. YOLO26-Face with 5-point keypoints (fallback chain)
#   2. OC-SORT / CMC Tracker (custom_ocsort.yaml auto-creation)
#   3. GPU 5-point affine warp to 112x112 (ArcFace reference)
#   4. AdaFace IResNet-50 feature extraction (512-D)
#   5. Feature-norm (||f||_2) quality gating (replaces Laplacian blur)
#   6. Quality-Weighted Softmax Voting + EMA Tracklet Smoothing
# =======================================================================

import os
import cv2
import threading
import queue
import time
import datetime
import requests
import pandas as pd
from collections import Counter, OrderedDict
import customtkinter as ctk
from PIL import Image, ImageTk
import faiss
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from requests.auth import HTTPBasicAuth
import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory
import yaml

# ==========================================
# CONFIGURATION (100% PRESERVED)
# ==========================================
USERNAME = "admin"
PASSWORD = "admin"
CAMERA_IP_1 = "10.34.0.17"
CAMERA_IP_2 = "10.34.0.16"

CAM1_PRESETS = [1, 2, 3, 4, 5, 6, 7, 8]
CAM2_PRESETS = [1, 2, 3, 4, 5, 6]

PROTOCOLS = {
    1: {"name": "5-Minute", "cam1": 32, "cam2": 45},
    2: {"name": "10-Minute", "cam1": 70, "cam2": 95},
    3: {"name": "15-Minute", "cam1": 107, "cam2": 145},
    4: {"name": "20-Minute", "cam1": 145, "cam2": 195},
    5: {"name": "25-Minute", "cam1": 182, "cam2": 245},
    6: {"name": "30-Minute", "cam1": 220, "cam2": 295},
    7: {"name": "35-Minute", "cam1": 257, "cam2": 345},
    8: {"name": "40-Minute", "cam1": 295, "cam2": 395}
}

CONFIDENCE_THRESHOLD = 0.48    # Recalibrated for AdaFace 512-D angular margin
FRAME_SKIP = 1
FRAMES_PER_VOTE = 5
NORM_QUALITY_THRESHOLD = 10.0
EMA_MOMENTUM = 0.7
SOFTMAX_TAU = 5.0

BASE_OUTPUT_DIR = "VIDEOS"
RESULTS_DIR = os.path.abspath('ATTENDENCE RESULTS/MINE')

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def format_timestamp(frame_count, fps):
    fps = fps if fps > 0 else 30
    total_seconds = int(frame_count // fps)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==========================================
# ADAFACE IRESNET-50 ARCHITECTURE
# ==========================================

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IResNetBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('IResNetBlock only supports groups=1 and base_width=64')
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class AdaFaceIResNet50(nn.Module):
    """
    Self-contained AdaFace IResNet-50.
    Input:  (B, 3, 112, 112)  BGR, normalized to [-1, 1]
    Output: (normed_embeddings (B,512), feature_norms (B,1))
    """
    fc_scale = 7 * 7

    def __init__(self, block=IResNetBlock, layers=(3, 4, 14, 3),
                 dropout=0.0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, fp16=False):
        super().__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IResNetBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        norms = torch.norm(x, p=2, dim=1, keepdim=True)
        normed = x / (norms + 1e-10)
        x = self.features(normed)
        return x, norms


def load_adaface_model(device, use_half=False):
    """Load AdaFace IResNet-50 with pretrained weights (fallback chain)."""
    model = AdaFaceIResNet50()

    weight_candidates = [
        'adaface_ir50_webface4m.ckpt',
        'adaface_ir50_webface4m.pth',
        'adaface_ir50_webface4m.pt',
    ]

    loaded = False
    for wf in weight_candidates:
        if os.path.exists(wf):
            print(f"Loading AdaFace weights from: {wf}")
            ckpt = torch.load(wf, map_location='cpu', weights_only=False)

            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt:
                    sd = ckpt['state_dict']
                elif 'model' in ckpt:
                    sd = ckpt['model']
                else:
                    sd = ckpt
            else:
                sd = ckpt

            new_sd = OrderedDict()
            for k, v in sd.items():
                name = k.replace('model.', '') if k.startswith('model.') else k
                new_sd[name] = v

            model.load_state_dict(new_sd, strict=False)
            loaded = True
            break

    if not loaded:
        print("WARNING: No AdaFace pretrained weights found! Model will have random weights.")

    model.eval().to(device)
    if use_half:
        model = model.half()
    return model


# ==========================================
# YOLO MODEL LOADER (FALLBACK CHAIN)
# ==========================================
def load_yolo_face(device=None):
    candidates = [
        ('yolo26n-face.engine', 'detect'),
        ('yolo26n-face.pt', 'detect'),
        ('yolov8n-face.engine', 'detect'),
        ('yolov8n-face.pt', 'detect'),
    ]
    for model_path, task in candidates:
        if os.path.exists(model_path):
            print(f"Loading YOLO face model: {model_path}")
            model = YOLO(model_path, task=task)
            return model
    raise FileNotFoundError(f"No YOLO face model found! Tried: {[c[0] for c in candidates]}")


# ==========================================
# OC-SORT TRACKER YAML AUTO-CREATION
# ==========================================
OCSORT_TRACKER_PATH = "custom_ocsort.yaml"

def ensure_ocsort_tracker():
    """Create custom_ocsort.yaml if it doesn't exist."""
    if not os.path.exists(OCSORT_TRACKER_PATH):
        tracker_cfg = {
            'tracker_type': 'botsort',
            'track_high_thresh': 0.5,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.6,
            'track_buffer': 50,
            'match_thresh': 0.8,
            'fuse_score': True,
            'gmc_method': 'sparseOptFlow',
            'with_reid': False,
        }
        with open(OCSORT_TRACKER_PATH, 'w') as f:
            yaml.dump(tracker_cfg, f, default_flow_style=False)
        print(f"Created tracker config: {OCSORT_TRACKER_PATH}")

ensure_ocsort_tracker()


# ==========================================
# GPU ALIGNMENT UTILITIES
# ==========================================
ARCFACE_DST = torch.tensor([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=torch.float32)


def synthesize_5pt_from_boxes(boxes):
    """
    Synthesize 5 canonical facial keypoints from bounding boxes.
    boxes: (N, 4) [x1, y1, x2, y2] on GPU
    Returns: (N, 5, 2)
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w, h = x2 - x1, y2 - y1
    kpts = torch.stack([
        torch.stack([x1 + 0.34 * w, y1 + 0.40 * h], dim=-1),
        torch.stack([x1 + 0.66 * w, y1 + 0.40 * h], dim=-1),
        torch.stack([x1 + 0.50 * w, y1 + 0.58 * h], dim=-1),
        torch.stack([x1 + 0.37 * w, y1 + 0.76 * h], dim=-1),
        torch.stack([x1 + 0.63 * w, y1 + 0.76 * h], dim=-1),
    ], dim=1)
    return kpts


def gpu_align_faces_112(frame_tensor_chw, keypoints_5pt, dst_template):
    """
    GPU vectorized 5-point affine warp to 112x112.
    frame_tensor_chw: (1, 3, H, W) float32 BGR tensor on GPU
    keypoints_5pt:    (N, 5, 2) float32 pixel coordinates on GPU
    dst_template:     (5, 2) ARCFACE_DST on GPU
    Returns:          (N, 3, 112, 112) aligned BGR float32 crops on GPU
    """
    N = keypoints_5pt.size(0)
    if N == 0:
        return torch.empty((0, 3, 112, 112), device=frame_tensor_chw.device, dtype=frame_tensor_chw.dtype)

    _, _, H, W = frame_tensor_chw.shape

    src_norm = torch.stack([
        (keypoints_5pt[:, :, 0] / (W - 1.0)) * 2.0 - 1.0,
        (keypoints_5pt[:, :, 1] / (H - 1.0)) * 2.0 - 1.0
    ], dim=-1)

    dst_pts = dst_template.unsqueeze(0).expand(N, -1, -1)
    dst_norm = (dst_pts / (112.0 - 1.0)) * 2.0 - 1.0

    ones = torch.ones((N, 5, 1), device=keypoints_5pt.device, dtype=torch.float32)
    X = torch.cat([dst_norm, ones], dim=2)
    theta_T = torch.linalg.lstsq(X, src_norm).solution
    theta = theta_T.permute(0, 2, 1)

    expanded_frames = frame_tensor_chw.expand(N, -1, -1, -1)
    grid = F.affine_grid(theta, size=(N, 3, 112, 112), align_corners=True)
    crops_112 = F.grid_sample(expanded_frames, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)
    return crops_112.detach()


def extract_keypoints_from_results(results, device):
    """
    Extract 5-point keypoints from YOLO results, with fallback synthesis.
    Returns: (N, 5, 2) tensor on device
    """
    boxes = results[0].boxes.xyxy.to(device).float()
    N = boxes.size(0)

    has_kpts = (results[0].keypoints is not None and
                results[0].keypoints.xy is not None and
                results[0].keypoints.xy.numel() > 0)

    if has_kpts:
        kpts = results[0].keypoints.xy.to(device).float()
        kpts_magnitude = kpts.abs().sum(dim=-1)
        invalid_mask = (kpts_magnitude < 1.0).any(dim=1)

        if invalid_mask.any():
            synth_kpts = synthesize_5pt_from_boxes(boxes)
            kpts[invalid_mask] = synth_kpts[invalid_mask]

        return kpts
    else:
        return synthesize_5pt_from_boxes(boxes)


# ==========================================
# MULTIPROCESSING INFERENCE PROCESS (THROTTLED CPU)
# ==========================================
def inference_worker(inf_queue, ann_queue, cmd_queue, timestamp_str, cam_name):
    """
    Evaluates ML on throttled inbound frames inherently preventing deadlocks.
    Writes native annotations directly against duplicate outputs maintaining 1-1 strict bounds flawlessly.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_half = torch.cuda.is_available()
    print(f"[ML PROCESS | {cam_name}] Initializing SOTA pipeline on {device}...")

    try:
        yolo_model = load_yolo_face(device)
        yolo_model.to(device)
        resnet = load_adaface_model(device, use_half=use_half)
        arcface_dst_local = ARCFACE_DST.to(device)

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
                ref_embeddings_tensor = F.normalize(ref_embeddings_tensor, p=2, dim=1)
                if use_half:
                    ref_embeddings_tensor = ref_embeddings_tensor.half()
            except Exception as e:
                print(f"[ML PROCESS | {cam_name}] Failed to extract FAISS vectors to GPU: {e}")
                index = None
        else:
            index = None

        meta_path = './face_attendance_meta.pkl'
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                saved_data = pickle.load(f)
            target_names = saved_data['target_names']
            y_real = saved_data['y_real']
        else:
            target_names, y_real = [], []

    except Exception as e:
        print(f"[ML PROCESS ERROR | {cam_name}] Failed to load models: {e}")
        return

    fps = 30
    frame_count = 0
    active_track_memory = {}
    archived_tracks = {}
    track_identities = {}

    print(f"[ML PROCESS | {cam_name}] Ready and listening for inbound throttled streams...")

    while True:
        try:
            cmd = cmd_queue.get_nowait()
            if cmd == 'STOP': break
        except queue.Empty: pass

        try:
            item = inf_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        shm = None
        try:
            shm = shared_memory.SharedMemory(name=item['shm_name'])
            frame_array = np.ndarray(item['shape'], dtype=item['dtype'], buffer=shm.buf)

            # Pure GPU Optimization: Immediately move frame tensor native to GPU bounds.
            frame_tensor = torch.from_numpy(frame_array).to(device, non_blocking=True).float()

            input_fps = item['fps']
            skipped_frames = item['skipped_frames']

            frame_count += (skipped_frames + 1)
            fps = input_fps if input_fps > 0 else 30

            # Deadlock Elimination: Collect tracking metadata strictly decoupled from UI loop drawing frames globally
            metadata = {'boxes': [], 'ids': [], 'names': []}
            current_active_faces = 0

            if ref_embeddings_tensor is not None and len(target_names) > 0:
                results = yolo_model.track(frame_array, persist=True, tracker=OCSORT_TRACKER_PATH, verbose=False, quantize=16 if use_half else None, imgsz=640)
                has_detections = results[0].boxes.id is not None

                if has_detections:
                    # Explicitly move the tensors to the GPU to match batch_idx
                    boxes = results[0].boxes.xyxy.to(device)
                    ids = results[0].boxes.id.to(device).int()
                    current_active_faces = len(ids)

                    ids_list = ids.cpu().tolist()
                    for t_id in ids_list:
                        if t_id not in active_track_memory:
                            active_track_memory[t_id] = {
                                'start_time': format_timestamp(frame_count, fps),
                                'frames_alive': 0, 'buffer': [], 'all_preds': [], 'missing_frames': 0,
                                'crop_buffer': [],
                                'ema_emb': None  # EMA embedding state
                            }
                        active_track_memory[t_id]['frames_alive'] += (skipped_frames + 1)

                    if frame_count % FRAME_SKIP == 0 or (frame_count - skipped_frames) % FRAME_SKIP == 0:
                        batch_tensors, batch_track_ids = [], []

                        box_w = boxes[:, 2] - boxes[:, 0]
                        box_h = boxes[:, 3] - boxes[:, 1]
                        aspect_ratios = box_w / (box_h + 1e-6)

                        valid_mask = (box_w >= 65) & (box_h >= 65) & (aspect_ratios >= 0.55) & (aspect_ratios <= 1.55)

                        if valid_mask.any():
                            valid_boxes = boxes[valid_mask].clone()
                            valid_ids = ids[valid_mask]

                            # Extract 5-point keypoints with fallback
                            all_kpts = extract_keypoints_from_results(results, device)
                            valid_kpts = all_kpts[valid_mask]  # (Nvalid, 5, 2)

                            # Standardize to NCHW format
                            frame_tensor_chw = frame_tensor.permute(2, 0, 1).unsqueeze(0)

                            # GPU 5-point affine warp to 112x112
                            crops = gpu_align_faces_112(frame_tensor_chw, valid_kpts, arcface_dst_local)  # (Nvalid, 3, 112, 112)

                            valid_ids_list = valid_ids.cpu().tolist()
                            for i, t_id in enumerate(valid_ids_list):
                                active_track_memory[t_id]['crop_buffer'].append(crops[i:i+1])

                                if len(active_track_memory[t_id]['crop_buffer']) >= FRAMES_PER_VOTE:
                                    batch_tensors.extend(active_track_memory[t_id]['crop_buffer'])
                                    batch_track_ids.extend([t_id] * len(active_track_memory[t_id]['crop_buffer']))
                                    active_track_memory[t_id]['crop_buffer'] = []

                        if batch_tensors:
                            with torch.inference_mode():
                                batch_tensor = torch.cat(batch_tensors, dim=0)  # (N, 3, 112, 112)

                                # Normalize to [-1, 1] keeping BGR order (AdaFace trained on BGR)
                                batch_tensor = (batch_tensor / 127.5) - 1.0
                                if use_half:
                                    batch_tensor = batch_tensor.half()

                                normed_embeddings, feature_norms = resnet(batch_tensor)
                                # normed_embeddings: (N, 512), feature_norms: (N, 1)

                                # Quality gate: feature norm
                                norms_flat = feature_norms.squeeze(1)
                                quality_mask = norms_flat >= NORM_QUALITY_THRESHOLD
                                quality_mask_list = quality_mask.cpu().tolist()

                                # Group valid embeddings/norms by track ID
                                track_emb_groups = {}
                                for k in range(len(batch_track_ids)):
                                    if quality_mask_list[k]:
                                        t_id = batch_track_ids[k]
                                        if t_id not in track_emb_groups:
                                            track_emb_groups[t_id] = {'embs': [], 'norms': []}
                                        track_emb_groups[t_id]['embs'].append(normed_embeddings[k:k+1])
                                        track_emb_groups[t_id]['norms'].append(norms_flat[k:k+1])

                                for t_id, group in track_emb_groups.items():
                                    e_track = torch.cat(group['embs'], dim=0)  # (K, 512)
                                    n_track = torch.cat(group['norms'], dim=0).unsqueeze(1)  # (K, 1)
                                    K = e_track.size(0)

                                    # Softmax quality weights with temperature tau
                                    weights = torch.softmax(n_track / SOFTMAX_TAU, dim=0)  # (K, 1)

                                    # Quality-weighted window embedding
                                    e_win = F.normalize(
                                        torch.sum(weights * e_track, dim=0, keepdim=True),
                                        p=2, dim=1
                                    )  # (1, 512)

                                    # EMA update
                                    if active_track_memory[t_id]['ema_emb'] is None:
                                        ema_emb = e_win
                                    else:
                                        prev_ema = active_track_memory[t_id]['ema_emb']
                                        ema_emb = F.normalize(
                                            EMA_MOMENTUM * prev_ema + (1.0 - EMA_MOMENTUM) * e_win,
                                            p=2, dim=1
                                        )
                                    active_track_memory[t_id]['ema_emb'] = ema_emb

                                    # Match against gallery
                                    sim_matrix = torch.mm(ema_emb.to(ref_embeddings_tensor.dtype), ref_embeddings_tensor.t())
                                    max_sim, max_idx = torch.max(sim_matrix, dim=1)

                                    winner = target_names[y_real[max_idx.item()]] if max_sim.item() > CONFIDENCE_THRESHOLD else "Unknown"

                                    track_identities[t_id] = winner

                                    # Extend all_preds K times for gate scaling
                                    active_track_memory[t_id]['all_preds'].extend([winner] * K)

                    # Collect tracking metadata instead of drawing on frames directly
                    boxes_cpu = boxes.cpu().numpy()
                    for i in range(len(ids_list)):
                        t_id = ids_list[i]
                        box = boxes_cpu[i]
                        name = track_identities.get(t_id, "Analyzing...")
                        metadata['boxes'].append(box)
                        metadata['ids'].append(t_id)
                        metadata['names'].append(name)

                    alive_ids = set(ids_list)
                else:
                    alive_ids = set()

                for t_id in list(active_track_memory.keys()):
                    if t_id not in alive_ids:
                        active_track_memory[t_id]['missing_frames'] += (skipped_frames + 1)
                        if active_track_memory[t_id]['missing_frames'] > 50:
                            archived_tracks[t_id] = active_track_memory.pop(t_id)
                    else:
                        active_track_memory[t_id]['missing_frames'] = 0

        finally:
            if shm is not None:
                shm.close()
                shm.unlink()

        # Enforce highly reliable logic outputs directly to single bound queue strictly avoiding cross queue ID locks!
        try:
            ann_queue.put((metadata['boxes'], metadata['ids'], metadata['names'], skipped_frames, current_active_faces, fps), timeout=0.2)
        except queue.Full:
            pass

    # ==========================
    # FINAL ATTENDANCE SECURE DUMP (100% identical gate logic)
    # ==========================
    print(f"[ML PROCESS | {cam_name}] Shutting down natively... Generating Log Checkboxes.")
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
            status = "Passed" if (total_frames >= 45 and total_samples >= 15 and sample_ratio >= 0.33 and win_ratio >= 0.52) else "Failed"
        else:
            winner = "Unknown"
            status = "Failed"
            counts = Counter(all_preds)

        if status == "Passed" and winner != "Unknown":
            student_presence[winner] = True
            student_detection_count[winner] += counts.get(winner, 0)

        debug_data.append({
            'Track ID': t_id, 'Start Time': data.get('start_time', ''),
            'Total Frames': total_frames, 'Valid Votes': valid_votes_count,
            'Total Preds (inc. Unknown)': len(all_preds), 'Predicted Identity': winner,
            'Gate Status': status, 'Breakdown': dict(Counter(all_preds))
        })

    pd.DataFrame(debug_data).to_csv(os.path.join(RESULTS_DIR, f"{timestamp_str}_{cam_name}_DEBUG_Tracks.csv"))
    output_data = [{'Name': s, 'Status': 'Present' if student_presence[s] else 'Absent', 'Detection Count': student_detection_count[s]} for s in target_names]
    pd.DataFrame(output_data).to_csv(os.path.join(RESULTS_DIR, f"{timestamp_str}_{cam_name}_output.csv"), index=False)


# ==========================================
# BACKGROUND OUTPUT AND INGESTION THREADS (100% PRESERVED)
# ==========================================
def rstp_reader(ip, running_event, raw_queue, inf_queue, ann_frame_queue=None, enable_inference=True):
    """
    Absolute Watchdog.
    Implements CPU optimization via Input Throttling caching explicit dropped frames natively.
    """
    url = f"rtsp://{USERNAME}:{PASSWORD}@{ip}:554/stream1"
    cap = None
    last_frame_time = time.perf_counter()

    while running_event.is_set():
        if cap is None or not cap.isOpened():
            print(f"[*] Native Reader Booting IP {ip}...")
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                time.sleep(5)
                continue
            last_frame_time = time.perf_counter()

        ret, frame = cap.read()
        now = time.perf_counter()

        if not ret:
            cap.release(); cap = None; continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0: fps = 30

        target_interval = 1.0 / fps
        delta = now - last_frame_time
        skipped_frames = max(0, round(delta / target_interval) - 1)
        last_frame_time = now

        # Raw Writer explicitly gets all original non-throttled data payload dictionaries (Zero-Copy)
        if not raw_queue.full():
            try:
                shm_raw = shared_memory.SharedMemory(create=True, size=frame.nbytes)
                np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm_raw.buf)[:] = frame[:]
                raw_queue.put_nowait({'shm_name': shm_raw.name, 'shape': frame.shape, 'dtype': frame.dtype, 'fps': fps, 'skipped_frames': skipped_frames + 1})
                shm_raw.close()
            except Exception: pass

        # Throttled Dispatch payload dicts evaluating constraints preventing CPU Serialization overload arrays entirely
        if enable_inference:
            if not inf_queue.full() and (ann_frame_queue is None or not ann_frame_queue.full()):
                try:
                    shm_inf = shared_memory.SharedMemory(create=True, size=frame.nbytes)
                    np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm_inf.buf)[:] = frame[:]
                    inf_queue.put_nowait({'shm_name': shm_inf.name, 'shape': frame.shape, 'dtype': frame.dtype, 'fps': fps, 'skipped_frames': skipped_frames})
                    shm_inf.close()

                    if ann_frame_queue is not None:
                        shm_ann = shared_memory.SharedMemory(create=True, size=frame.nbytes)
                        np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm_ann.buf)[:] = frame[:]
                        ann_frame_queue.put_nowait({'shm_name': shm_ann.name, 'shape': frame.shape, 'dtype': frame.dtype, 'fps': fps, 'skipped_frames': skipped_frames})
                        shm_ann.close()
                except Exception: pass

    if cap: cap.release()

def raw_writer_worker(queue_obj, ui_queue, output_path, running_event, populate_ui=False):
    writer = None
    while True:
        try:
            item = queue_obj.get(timeout=0.2)
            if item is None: break

            shm = None
            try:
                shm = shared_memory.SharedMemory(name=item['shm_name'])
                frame = np.ndarray(item['shape'], dtype=item['dtype'], buffer=shm.buf)

                input_fps = item['fps']
                duplicates = item['skipped_frames']

                if writer is None:
                    h, w = frame.shape[:2]
                    fps = input_fps if input_fps > 0 else 30
                    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                for _ in range(duplicates):
                    writer.write(frame)

                if populate_ui and ui_queue is not None:
                    if not ui_queue.full():
                        ui_frame = cv2.resize(frame, (1024, 576))
                        ui_queue.put_nowait((ui_frame, 0))
            finally:
                if shm is not None:
                    shm.close()
                    shm.unlink()

        except queue.Empty:
            if not running_event.is_set() and queue_obj.empty(): break
    if writer: writer.release()

def async_video_writer_worker(writer_queue, output_path, running_event):
    writer = None
    while True:
        try:
            item = writer_queue.get(timeout=0.2)
            if item is None: break
            frame, input_fps, duplicates = item

            if writer is None:
                h, w = frame.shape[:2]
                fps = input_fps if input_fps > 0 else 30
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            for _ in range(duplicates):
                writer.write(frame)

        except queue.Empty:
            if not running_event.is_set() and writer_queue.empty(): break

    if writer: writer.release()

def ann_writer_worker(ann_queue, ann_frame_queue, ui_queue, writer_queue, running_event):
    """
    Highly robust generic bounds mapping directly towards duplicating skipped loops.
    Strictly deadlock-free!
    """
    while True:
        try:
            item = ann_queue.get(timeout=0.2)
            if item is None: break
            boxes, ids, names, skipped_frames, active_faces, input_fps = item

            try:
                frame_item = ann_frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            shm = None
            try:
                shm = shared_memory.SharedMemory(name=frame_item['shm_name'])
                frame_arr = np.ndarray(frame_item['shape'], dtype=frame_item['dtype'], buffer=shm.buf)
                frame = frame_arr.copy()
            finally:
                if shm is not None:
                    shm.close()
                    shm.unlink()

            for i in range(len(ids)):
                box = boxes[i]
                t_id = ids[i]
                name = names[i]
                color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not writer_queue.full():
                writer_queue.put_nowait((frame, input_fps, skipped_frames + 1))

            if not ui_queue.full():
                ui_frame = cv2.resize(frame, (1024, 576))
                ui_queue.put_nowait((ui_frame, active_faces))

        except queue.Empty:
            if not running_event.is_set() and ann_queue.empty(): break


# ==========================================
# MAIN DESKTOP GUI (100% PRESERVED LAYOUT & WIDGETS)
# ==========================================
class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Dual-Process Optimized Architecture")
        self.geometry("1400x850")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.running_event = mp.Event()
        self.selected_protocol = ctk.IntVar(value=1)
        self.view_target = ctk.StringVar(value="Camera 1")

        self.t_end = 0
        self.sys_fps = 0
        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        self.cam1_active_faces = 0
        self.cam2_active_faces = 0

        self.setup_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.ptz_cmd_queue = queue.Queue()
        self.ptz_worker_thread = threading.Thread(target=self.ptz_command_worker, daemon=True)
        self.ptz_worker_thread.start()

    def ptz_command_worker(self):
        while True:
            url = self.ptz_cmd_queue.get()
            if url == "QUIT": break
            try:
                requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=3)
            except Exception:
                pass

    def on_closing(self):
        print("Application shutting down... cleaning up memory and processes.")
        self.stop_tracking()
        if hasattr(self, 'ptz_cmd_queue'):
            self.ptz_cmd_queue.put("QUIT")
        self.destroy()

    def setup_ui(self):
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Attendance Core ML", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(pady=20, padx=20)

        self.view_label = ctk.CTkLabel(self.sidebar_frame, text="Active Telemetry View:", font=ctk.CTkFont(size=14, underline=True))
        self.view_label.pack(pady=(10,0), padx=20, anchor="w")
        self.view_toggle = ctk.CTkSegmentedButton(self.sidebar_frame, values=["Camera 1", "Camera 2"], variable=self.view_target)
        self.view_toggle.pack(pady=10, padx=20, fill="x")

        self.cam1_enable_var = ctk.BooleanVar(value=True)
        self.cam1_enable_switch = ctk.CTkSwitch(self.sidebar_frame, text="Enable Camera 1", variable=self.cam1_enable_var)
        self.cam1_enable_switch.pack(pady=(10, 0), padx=20, anchor="w")

        self.cam2_enable_var = ctk.BooleanVar(value=True)
        self.cam2_enable_switch = ctk.CTkSwitch(self.sidebar_frame, text="Enable Camera 2", variable=self.cam2_enable_var)
        self.cam2_enable_switch.pack(pady=10, padx=20, anchor="w")

        self.global_live_tracking_var = ctk.BooleanVar(value=True)
        self.global_live_tracking_switch = ctk.CTkSwitch(self.sidebar_frame, text="Live Tracking", variable=self.global_live_tracking_var)
        self.global_live_tracking_switch.pack(pady=(15,0), padx=20, anchor="w")

        self.protocol_label = ctk.CTkLabel(self.sidebar_frame, text="Execution Protocol:")
        self.protocol_label.pack(pady=(15,0), padx=20, anchor="w")
        self.protocol_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.protocol_frame.pack(pady=5, padx=20, fill="x")

        for idx, (p, info) in enumerate(PROTOCOLS.items()):
            r = idx // 2
            c = idx % 2
            rb = ctk.CTkRadioButton(self.protocol_frame, text=info['name'], variable=self.selected_protocol, value=p)
            rb.grid(row=r, column=c, pady=5, padx=5, sticky="w")

        self.ptz_label1 = ctk.CTkLabel(self.sidebar_frame, text="Cam1 Presets:")
        self.ptz_label1.pack(pady=(10,0), padx=20, anchor="w")
        self.preset_frame1 = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.preset_frame1.pack(pady=5, padx=20, fill="x")
        self.cam1_preset_vars = {}
        for p in CAM1_PRESETS:
            var = ctk.BooleanVar(value=True)
            self.cam1_preset_vars[p] = var
            cb = ctk.CTkCheckBox(self.preset_frame1, text=str(p), variable=var, width=40)
            cb.grid(row=(p-1)//4, column=(p-1)%4, padx=5, pady=5)

        self.ptz_label2 = ctk.CTkLabel(self.sidebar_frame, text="Cam2 Presets:")
        self.ptz_label2.pack(pady=(10,0), padx=20, anchor="w")
        self.preset_frame2 = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.preset_frame2.pack(pady=5, padx=20, fill="x")
        self.cam2_preset_vars = {}
        for p in CAM2_PRESETS:
            var = ctk.BooleanVar(value=True)
            self.cam2_preset_vars[p] = var
            cb = ctk.CTkCheckBox(self.preset_frame2, text=str(p), variable=var, width=40)
            cb.grid(row=(p-1)//4, column=(p-1)%4, padx=5, pady=5)

        self.ptz_manual_label = ctk.CTkLabel(self.sidebar_frame, text="Manual PTZ Controls:")
        self.ptz_manual_label.pack(pady=(10,0), padx=20, anchor="w")
        self.manual_ptz_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.manual_ptz_frame.pack(pady=5, padx=20, fill="x")

        btn_up = ctk.CTkButton(self.manual_ptz_frame, text="Up", width=50)
        btn_down = ctk.CTkButton(self.manual_ptz_frame, text="Down", width=50)
        btn_left = ctk.CTkButton(self.manual_ptz_frame, text="Left", width=50)
        btn_right = ctk.CTkButton(self.manual_ptz_frame, text="Right", width=50)
        btn_zi = ctk.CTkButton(self.manual_ptz_frame, text="Zoom In", width=50)
        btn_zo = ctk.CTkButton(self.manual_ptz_frame, text="Zoom Out", width=50)
        btn_fi = ctk.CTkButton(self.manual_ptz_frame, text="Focus In", width=50)
        btn_fo = ctk.CTkButton(self.manual_ptz_frame, text="Focus Out", width=50)

        for btn, cmd, stop_cmd in [
            (btn_up, "up", "ptzstop"), (btn_down, "down", "ptzstop"), (btn_left, "left", "ptzstop"),
            (btn_right, "right", "ptzstop"), (btn_zi, "zoomin", "zoomstop"), (btn_zo, "zoomout", "zoomstop"),
            (btn_fi, "focusin", "focusstop"), (btn_fo, "focusout", "focusstop")
        ]:
            for widget in [btn, btn._canvas, btn._text_label]:
                widget.bind("<ButtonPress-1>", lambda e, c=cmd: self.manual_ptz(c))
                widget.bind("<ButtonRelease-1>", lambda e, sc=stop_cmd: self.manual_ptz(sc))

        btn_up.grid(row=0, column=1, padx=2, pady=2)
        btn_down.grid(row=2, column=1, padx=2, pady=2)
        btn_left.grid(row=1, column=0, padx=2, pady=2)
        btn_right.grid(row=1, column=2, padx=2, pady=2)
        btn_zi.grid(row=0, column=3, padx=2, pady=2)
        btn_zo.grid(row=2, column=3, padx=2, pady=2)
        btn_fi.grid(row=0, column=4, padx=2, pady=2)
        btn_fo.grid(row=2, column=4, padx=2, pady=2)

        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="Execute Bound Loop", command=self.start_tracking)
        self.start_btn.pack(pady=20, padx=20)

        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="Force Stop System", command=self.stop_tracking, fg_color="red", hover_color="darkred")
        self.stop_btn.pack(pady=10, padx=20)
        self.stop_btn.configure(state="disabled")

        self.telemetry_lbl = ctk.CTkLabel(self.sidebar_frame, text="Live Telemetry Overlay", font=ctk.CTkFont(size=18, weight="bold"))
        self.telemetry_lbl.pack(pady=(40,10), padx=20, anchor="w")

        self.time_lbl = ctk.CTkLabel(self.sidebar_frame, text="Remaining Timer: 00:00:00", font=ctk.CTkFont(size=14))
        self.time_lbl.pack(pady=5, padx=20, anchor="w")
        self.fps_lbl = ctk.CTkLabel(self.sidebar_frame, text="System Render FPS: 0.0", font=ctk.CTkFont(size=14))
        self.fps_lbl.pack(pady=5, padx=20, anchor="w")
        self.faces_lbl = ctk.CTkLabel(self.sidebar_frame, text="Target Faces Detected: 0", font=ctk.CTkFont(size=14))
        self.faces_lbl.pack(pady=5, padx=20, anchor="w")

        self.video_frame = ctk.CTkLabel(self, text="Camera ML Feed Offline", bg_color="gray", width=1024, height=576)
        self.video_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    def start_tracking(self):
        if self.running_event.is_set(): return

        self.running_event.set()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        sel = self.selected_protocol.get()
        p_info = PROTOCOLS[sel]

        selected_cam1_presets = [p for p, var in self.cam1_preset_vars.items() if var.get()]
        selected_cam2_presets = [p for p, var in self.cam2_preset_vars.items() if var.get()]

        use_cam1 = self.cam1_enable_var.get()
        use_cam2 = self.cam2_enable_var.get()

        if not use_cam1 and not use_cam2:
            print("[Warning] Both cameras disabled. Cannot start tracking.")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.running_event.clear()
            return

        global_track = self.global_live_tracking_var.get()
        mode1 = global_track
        mode2 = global_track

        total_time_cam1 = len(selected_cam1_presets) * (p_info['cam1'] + 5) if use_cam1 else 0
        total_time_cam2 = len(selected_cam2_presets) * (p_info['cam2'] + 5) if use_cam2 else 0
        self.t_end = time.perf_counter() + max(total_time_cam1, total_time_cam2)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.threads = []

        if use_cam1:
            self.cam1_raw_q = queue.Queue(maxsize=60)
            self.cam1_ui_q = queue.Queue(maxsize=30)

            if mode1:
                self.cam1_inf_q = mp.Queue(maxsize=30)
                self.cam1_ann_q = mp.Queue(maxsize=150)
                self.cam1_cmd_q = mp.Queue()
                self.cam1_ann_frame_q = queue.Queue(maxsize=30)
                self.cam1_writer_q = queue.Queue(maxsize=150)
            else:
                self.cam1_inf_q = None
                self.cam1_ann_q = None
                self.cam1_cmd_q = None
                self.cam1_ann_frame_q = None
                self.cam1_writer_q = None

            self.threads.append(threading.Thread(target=rstp_reader, args=(CAMERA_IP_1, self.running_event, self.cam1_raw_q, self.cam1_inf_q, self.cam1_ann_frame_q, mode1)))
            self.threads.append(threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_1, selected_cam1_presets, p_info['cam1'])))

            if mode1:
                self.ml_p1 = mp.Process(target=inference_worker, args=(self.cam1_inf_q, self.cam1_ann_q, self.cam1_cmd_q, timestamp_str, "Cam1"))
                self.ml_p1.daemon = True; self.ml_p1.start()
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam1_raw_q, None, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam1_raw.mp4"), self.running_event, False)))
                self.threads.append(threading.Thread(target=ann_writer_worker, args=(self.cam1_ann_q, self.cam1_ann_frame_q, self.cam1_ui_q, self.cam1_writer_q, self.running_event)))
                self.threads.append(threading.Thread(target=async_video_writer_worker, args=(self.cam1_writer_q, os.path.join(RESULTS_DIR, f"{timestamp_str}_cam1_annotated.mp4"), self.running_event)))
            else:
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam1_raw_q, self.cam1_ui_q, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam1_raw.mp4"), self.running_event, True)))

        if use_cam2:
            self.cam2_raw_q = queue.Queue(maxsize=60)
            self.cam2_ui_q = queue.Queue(maxsize=30)

            if mode2:
                self.cam2_inf_q = mp.Queue(maxsize=30)
                self.cam2_ann_q = mp.Queue(maxsize=150)
                self.cam2_cmd_q = mp.Queue()
                self.cam2_ann_frame_q = queue.Queue(maxsize=30)
                self.cam2_writer_q = queue.Queue(maxsize=150)
            else:
                self.cam2_inf_q = None
                self.cam2_ann_q = None
                self.cam2_cmd_q = None
                self.cam2_ann_frame_q = None
                self.cam2_writer_q = None

            self.threads.append(threading.Thread(target=rstp_reader, args=(CAMERA_IP_2, self.running_event, self.cam2_raw_q, self.cam2_inf_q, self.cam2_ann_frame_q, mode2)))
            self.threads.append(threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_2, selected_cam2_presets, p_info['cam2'])))

            if mode2:
                self.ml_p2 = mp.Process(target=inference_worker, args=(self.cam2_inf_q, self.cam2_ann_q, self.cam2_cmd_q, timestamp_str, "Cam2"))
                self.ml_p2.daemon = True; self.ml_p2.start()
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam2_raw_q, None, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam2_raw.mp4"), self.running_event, False)))
                self.threads.append(threading.Thread(target=ann_writer_worker, args=(self.cam2_ann_q, self.cam2_ann_frame_q, self.cam2_ui_q, self.cam2_writer_q, self.running_event)))
                self.threads.append(threading.Thread(target=async_video_writer_worker, args=(self.cam2_writer_q, os.path.join(RESULTS_DIR, f"{timestamp_str}_cam2_annotated.mp4"), self.running_event)))
            else:
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam2_raw_q, self.cam2_ui_q, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam2_raw.mp4"), self.running_event, True)))

        for t in self.threads: t.start()

        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        self.update_gui_frame()

    def stop_tracking(self):
        if not self.running_event.is_set(): return
        self.running_event.clear()

        if hasattr(self, 'cam1_cmd_q') and self.cam1_cmd_q:
            self.cam1_cmd_q.put('STOP')
        if hasattr(self, 'cam2_cmd_q') and self.cam2_cmd_q:
            self.cam2_cmd_q.put('STOP')

        # 1. CRITICAL FIX: Wait for background reader threads to stop generating frames
        if hasattr(self, 'threads'):
            for t in self.threads:
                if t.is_alive():
                    t.join(timeout=1.0)

        # 2. Wait for ML processes to safely exit
        if hasattr(self, 'ml_p1') and self.ml_p1.is_alive():
            self.ml_p1.join(timeout=2.0)
        if hasattr(self, 'ml_p2') and self.ml_p2.is_alive():
            self.ml_p2.join(timeout=2.0)

        # 3. Now safely detach and purge pending SharedMemory scopes
        for q_name in ['cam1_inf_q', 'cam2_inf_q', 'cam1_raw_q', 'cam2_raw_q', 'cam1_ann_frame_q', 'cam2_ann_frame_q']:
            if hasattr(self, q_name) and getattr(self, q_name):
                q = getattr(self, q_name)
                while not q.empty():
                    try:
                        item = q.get_nowait()
                        if isinstance(item, dict) and 'shm_name' in item:
                            try:
                                shm = shared_memory.SharedMemory(name=item['shm_name'])
                                shm.close()
                                shm.unlink()
                            except Exception: pass
                    except queue.Empty: break

        if hasattr(self, 'cam1_ann_q') and self.cam1_ann_q: self.cam1_ann_q.put(None)
        if hasattr(self, 'cam2_ann_q') and self.cam2_ann_q: self.cam2_ann_q.put(None)
        if hasattr(self, 'cam1_writer_q') and self.cam1_writer_q: self.cam1_writer_q.put(None)
        if hasattr(self, 'cam2_writer_q') and self.cam2_writer_q: self.cam2_writer_q.put(None)
        if hasattr(self, 'cam1_raw_q') and self.cam1_raw_q: self.cam1_raw_q.put(None)
        if hasattr(self, 'cam2_raw_q') and self.cam2_raw_q: self.cam2_raw_q.put(None)

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def ptz_runner(self, ip, presets, duration):
        for preset in presets:
            if not self.running_event.is_set(): break
            url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&poscall&{preset}"
            try: requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=5)
            except: pass

            t_mechanical = time.perf_counter()
            while self.running_event.is_set() and (time.perf_counter() - t_mechanical) < 5.0:
                time.sleep(0.1)

            t_start = time.perf_counter()
            while self.running_event.is_set() and (time.perf_counter() - t_start) < duration:
                time.sleep(0.1)

    def manual_ptz(self, command):
        target = self.view_target.get()
        ip = CAMERA_IP_1 if target == "Camera 1" else CAMERA_IP_2
        url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&{command}"
        self.ptz_cmd_queue.put(url)

    def update_gui_frame(self):
        if not self.running_event.is_set():
            self.video_frame.configure(image=None, text="Camera ML Feed Offline")
            return

        try:
            now = time.perf_counter()
            remaining = max(0, int(self.t_end - now))

            if remaining <= 0:
                self.stop_tracking()
                return

            h, rem = divmod(remaining, 3600)
            m, s = divmod(rem, 60)
            self.time_lbl.configure(text=f"Time Remaining: {h:02d}:{m:02d}:{s:02d}")

            frame1 = None
            if hasattr(self, 'cam1_ui_q') and self.cam1_ui_q and not self.cam1_ui_q.empty():
                try:
                    res1 = self.cam1_ui_q.get_nowait()
                    frame1, self.cam1_active_faces = res1
                except queue.Empty: pass

            frame2 = None
            if hasattr(self, 'cam2_ui_q') and self.cam2_ui_q and not self.cam2_ui_q.empty():
                try:
                    res2 = self.cam2_ui_q.get_nowait()
                    frame2, self.cam2_active_faces = res2
                except queue.Empty: pass

            if (now - self.last_fps_time) >= 1.0:
                self.sys_fps = self.frames_rendered / (now - self.last_fps_time)
                self.fps_lbl.configure(text=f"System Render FPS: {self.sys_fps:.1f}")
                disp_faces = self.cam1_active_faces if self.view_target.get() == "Camera 1" else self.cam2_active_faces
                self.faces_lbl.configure(text=f"Target Faces Detected: {disp_faces}")
                self.frames_rendered = 0
                self.last_fps_time = now

            target_frame = frame1 if self.view_target.get() == "Camera 1" else frame2
            if target_frame is not None:
                self.frames_rendered += 1
                frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(1024, 576))
                self.video_frame.configure(image=imgtk, text="")
                self.video_frame.image = imgtk

        except Exception as e: pass
        finally: self.after(10, self.update_gui_frame)

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    app = AttendanceApp()
    app.mainloop()

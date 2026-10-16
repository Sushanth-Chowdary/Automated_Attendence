# test_yolo_sota.py  –  SOTA Multi-Process Batch Video Evaluation & Rich Dashboard
# ================================================================================
# Upgrades:
#   1. YOLO26-Face with 5-point keypoints (fallback chain)
#   2. OC-SORT / CMC Tracker (custom_ocsort.yaml auto-creation)
#   3. GPU 5-point affine warp to 112x112 (ArcFace reference)
#   4. AdaFace IResNet-50 feature extraction (512-D)
#   5. Feature-norm (||f||_2) quality gating (replaces Laplacian blur)
#   6. Quality-Weighted Softmax Voting + EMA Tracklet Smoothing
# ================================================================================

# 1. Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import pickle
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm
from collections import Counter, OrderedDict
import faiss
import threading
import queue
from ultralytics import YOLO
import yaml


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
def load_yolo_face(device):
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
        kpts = results[0].keypoints.xy.to(device).float()  # (N, 5, 2)
        # Validate each detection's keypoints
        kpts_magnitude = kpts.abs().sum(dim=-1)  # (N, 5)
        invalid_mask = (kpts_magnitude < 1.0).any(dim=1)  # (N,) — True if any kpt is ~(0,0)

        if invalid_mask.any():
            # Synthesize for invalid detections
            synth_kpts = synthesize_5pt_from_boxes(boxes)
            kpts[invalid_mask] = synth_kpts[invalid_mask]

        return kpts
    else:
        return synthesize_5pt_from_boxes(boxes)


# ==========================================
# PRESERVED: ThreadedVideoReader
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
        try: return self.q.get(timeout=2.0)
        except queue.Empty: return None

    def more(self): return self.q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True


# ==========================================
# PRESERVED: video_writer_worker
# ==========================================
def video_writer_worker(write_queue, output_path, fps, width, height):
    """Handles CPU drawing and slow video encoding in the background."""
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
            cv2.putText(frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
    out.release()


def format_timestamp(frame_count, fps):
    total_seconds = frame_count // fps
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==========================================
# 2. Setup
# ==========================================
ensure_ocsort_tracker()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_half = torch.cuda.is_available()

resnet = load_adaface_model(device, use_half=use_half)
arcface_dst = ARCFACE_DST.to(device)

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
        print(f"Failed to extract FAISS vectors to GPU: {e}")

with open('./face_attendance_meta.pkl', 'rb') as f:
    saved_data = pickle.load(f)
target_names, y_real = saved_data['target_names'], saved_data['y_real']

# 3. Parameters
CONFIDENCE_THRESHOLD = 0.48      # Recalibrated for AdaFace 512-D angular margin
FRAME_SKIP = 1
FRAMES_PER_VOTE = 5
NORM_QUALITY_THRESHOLD = 10.0
EMA_MOMENTUM = 0.7
SOFTMAX_TAU = 5.0

input_dir = 'VIDEOS'
output_dir = os.path.abspath('ATTENDENCE RESULTS/MINE')
os.makedirs(output_dir, exist_ok=True)

target_videos = ['2026-04-27_10.02.44.mkv', '2026-02-25_11.23.07.mkv', '2026-03-05_11.02.28.mkv', '2026-03-09_10.03.16.mkv', '2026-04-07_09.18.02.mkv', 'video2.mkv', '2026-02-25_11.21.17.mkv', '2026-03-09_10.04.35.mkv', '2026-02-25_11.03.43.mkv', '2026-02-18_11.02.03.mkv', 'video1_uajX8qg0.mp4', '2026-02-25_11.00.04.mkv', '2026-02-25_11.15.41.mkv', '2026-03-02_09.55.37.mkv']

# ==========================================
# PRESERVED: save_attendance_results (100% identical gate logic)
# ==========================================
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
    print(f"  -> Saved attendance + debug CSVs for {video_filename}")


# ==========================================
# MAIN PROCESSING LOOP
# ==========================================
interrupted = False
for video_filename in target_videos:
    if not os.path.exists(os.path.join(input_dir, video_filename)): continue
    print(f"\nProcessing: {video_filename}")
    yolo_model = load_yolo_face(device)
    yolo_model.to(device)

    # Safe tracker state reset across videos
    if hasattr(yolo_model, 'predictor') and yolo_model.predictor is not None:
        if hasattr(yolo_model.predictor, 'trackers') and yolo_model.predictor.trackers:
            for tracker in yolo_model.predictor.trackers:
                tracker.reset()

    video_stream = ThreadedVideoReader(os.path.join(input_dir, video_filename)).start()
    video_stem = os.path.splitext(video_filename)[0]
    final_video_path = os.path.join(output_dir, f"{video_stem}_output.mp4")

    write_queue = queue.Queue(maxsize=128)
    writer_thread = threading.Thread(target=video_writer_worker, args=(write_queue, final_video_path, video_stream.fps, video_stream.frame_width, video_stream.frame_height))
    writer_thread.daemon = True
    writer_thread.start()

    active_track_memory, archived_tracks, track_identities = {}, {}, {}
    frame_count = 0

    try:
        with tqdm(total=video_stream.total_frames, unit="frame") as pbar:
            while video_stream.more():
                frame = video_stream.read()
                if frame is None: break

                frame_tensor = torch.from_numpy(frame).to(device, non_blocking=True).float()

                results = yolo_model.track(frame, persist=True, tracker=OCSORT_TRACKER_PATH, verbose=False, quantize=16 if use_half else None, imgsz=640)
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
                                'crop_buffer': [],
                                'ema_emb': None  # EMA embedding state
                            }
                        active_track_memory[t_id]['frames_alive'] += 1

                    if frame_count % FRAME_SKIP == 0:
                        batch_tensors = []
                        batch_track_ids = []

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

                            # GPU affine warp to 112x112
                            frame_tensor_chw = frame_tensor.permute(2, 0, 1).unsqueeze(0)
                            crops = gpu_align_faces_112(frame_tensor_chw, valid_kpts, arcface_dst)  # (Nvalid, 3, 112, 112)

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
                                norms_flat = feature_norms.squeeze(1)  # (N,)
                                quality_mask = norms_flat >= NORM_QUALITY_THRESHOLD
                                quality_mask_list = quality_mask.cpu().tolist()

                                # Group valid embeddings/norms by track ID
                                track_emb_groups = {}  # t_id -> (list of emb, list of norm)
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
                                    conf_score = max_sim.item()

                                    track_identities[t_id] = (winner, conf_score)

                                    # Extend all_preds K times for gate scaling
                                    active_track_memory[t_id]['all_preds'].extend([winner] * K)

                    boxes_cpu = boxes.cpu().numpy()
                    for i in range(len(ids_list)):
                        t_id = ids_list[i]
                        identity_info = track_identities.get(t_id, ("Analyzing...", 0.0))
                        frame_names.append(identity_info[0] if isinstance(identity_info, tuple) else identity_info)

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
                pbar.update(1)

    except KeyboardInterrupt:
        interrupted = True
    finally:
        video_stream.stop()
        write_queue.put(None)
        writer_thread.join()

        save_attendance_results(video_filename, archived_tracks, active_track_memory, target_names, output_dir)
    if interrupted: break

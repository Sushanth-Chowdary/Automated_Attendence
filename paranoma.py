import os
import glob
import time
import datetime
import threading
import subprocess
import requests
from requests.auth import HTTPBasicAuth
import cv2
import numpy as np
import pandas as pd
import pickle
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from ultralytics import YOLO

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =====================================================================
# 1. CAMERA, PTZ SWEEP & RECOGNITION CONFIGURATION
# =====================================================================
CAMERA_IP = "10.34.0.17"       # Or "10.34.0.16" for Camera 2
USERNAME = "admin"
PASSWORD = "admin"

RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/stream1"

OUTPUT_DIR = "sweep_images"
RESULTS_DIR = os.path.abspath("ATTENDENCE RESULTS/PANO")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_PANO_TIF = os.path.join(RESULTS_DIR, "raw_classroom_panorama.tif")
RAW_PANO_JPG = os.path.join(RESULTS_DIR, "raw_classroom_panorama.jpg")
ANNOTATED_PANO_JPG = os.path.join(RESULTS_DIR, "annotated_classroom_panorama.jpg")
ATTENDANCE_CSV = os.path.join(RESULTS_DIR, "panorama_attendance_output.csv")

# Optional: Set to a preset integer (e.g., 1) or leave None to auto-home via PTZ commands
START_PRESET = None

# Full Classroom Sweep Grid & PTZ Timing Parameters
# Covers full horizontal ("left" <-> "right") and vertical ("up" <-> "down") classroom field of view
GRID_ROWS = 5               # Vertical tilt rows (Top -> Bottom)
GRID_COLS = 8               # Horizontal pan columns per row (Left <-> Right)

INIT_LEFT_TIME = 8.0        # Seconds to pan 'left' to hit the far-left boundary
INIT_UP_TIME = 5.0          # Seconds to tilt 'up' to hit the top boundary
PAN_STEP_TIME = 0.65        # Seconds per horizontal step ('right' or 'left')
TILT_STEP_TIME = 0.50       # Seconds per vertical step ('down' or 'up')
ZOOM_IN_TIME = 2.5          # Seconds to zoom in for clear distant faces (set 0.0 for widest FOV)
SETTLE_TIME = 1.8           # Seconds to wait after PTZ stop for mechanical stabilization & autofocus

# SOTA Recognition Hyperparameters (Matched to main_gui_sota.py)
FAISS_INDEX_PATH = "./face_attendance_sota_faiss.bin"
META_PATH = "./face_attendance_sota_meta.pkl"
CONFIDENCE_THRESHOLD = 0.32
NORM_QUALITY_THRESHOLD = 6.0
MIN_FACE_SIZE = 35          # Minimum face box width/height in pixels

# Tiled High-Res Detection Parameters (Essential for Gigapixel Panoramas)
TILE_SIZE = 1280            # Slice size for YOLO inference on mega-panorama
TILE_OVERLAP = 0.25         # 25% overlap between slices so faces on borders aren't split
GLOBAL_NMS_IOU = 0.45       # IoU threshold to merge duplicate detections across tile borders & grid seams


# =====================================================================
# 2. CONTINUOUS RTSP STREAM READER (Prevents Stale/Buffered Frames)
# =====================================================================
class LiveRTSPStream:
    """Continuously drains the RTSP buffer in a background thread so get_frame() always returns the latest live image."""
    def __init__(self, url: str):
        self.url = url
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        print(f"[*] Connecting to RTSP stream: {self.url} ...")
        t0 = time.perf_counter()
        while self.get_frame() is None and (time.perf_counter() - t0) < 12.0:
            time.sleep(0.1)
        if self.get_frame() is None:
            raise RuntimeError(f"Failed to receive frames from {self.url}")
        print("[*] RTSP stream active and receiving live frames.")

    def _reader_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.url)
                if not self.cap.isOpened():
                    time.sleep(2)
                    continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.cap.release()
                self.cap = None
                continue
            with self.lock:
                self.latest_frame = frame

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()


# =====================================================================
# 3. EXACT PTZ HARDWARE CONTROL (Left / Right / Up / Down / Zoom)
# =====================================================================
def manual_ptz(command: str, ip: str = CAMERA_IP) -> bool:
    """Sends exact PTZ command matching main_gui_sota.py: /cgi-bin/ptzctrl.cgi?ptzcmd&{command}"""
    url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&{command}"
    try:
        resp = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=3)
        return resp.status_code == 200
    except Exception as e:
        print(f"[PTZ ERROR] Failed sending '{command}' to {url}: {e}")
        return False


def call_preset(preset_num: int, ip: str = CAMERA_IP) -> bool:
    """Calls a camera preset: /cgi-bin/ptzctrl.cgi?ptzcmd&poscall&{preset}"""
    url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&poscall&{preset_num}"
    try:
        print(f"  -> Moving to Preset {preset_num}...")
        resp = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=5)
        return resp.status_code == 200
    except Exception as e:
        print(f"[PRESET ERROR] Failed calling preset {preset_num}: {e}")
        return False


def move_ptz_timed(start_cmd: str, stop_cmd: str, duration: float):
    """Executes directional PTZ movement ('left', 'right', 'up', 'down', 'zoomin', 'zoomout') for `duration` seconds."""
    if duration <= 0:
        return
    print(f"  -> PTZ '{start_cmd}' for {duration:.2f}s -> '{stop_cmd}'")
    manual_ptz(start_cmd)
    time.sleep(duration)
    manual_ptz(stop_cmd)


# =====================================================================
# 4. FULL CLASSROOM SNAKE-MATRIX SWEEP (Left, Right, Up, Down)
# =====================================================================
def run_classroom_sweep():
    for old_file in glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")):
        os.remove(old_file)

    stream = LiveRTSPStream(RTSP_URL)
    stream.start()

    captured_files = []
    grid_frames = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    try:
        print("\n=== STEP 1: INITIALIZING STARTING CORNER (LEFT + UP) & ZOOM ===")
        # Reset zoom completely out first so homing is consistent
        move_ptz_timed("zoomout", "zoomstop", duration=4.0)

        if START_PRESET is not None:
            call_preset(START_PRESET)
            time.sleep(5.0)
        else:
            print("Sweeping fully LEFT and UP to top-left classroom origin...")
            move_ptz_timed("left", "ptzstop", duration=INIT_LEFT_TIME)
            move_ptz_timed("up", "ptzstop", duration=INIT_UP_TIME)

        if ZOOM_IN_TIME > 0:
            print(f"Applying optical zoom ({ZOOM_IN_TIME}s) for classroom face detail...")
            move_ptz_timed("zoomin", "zoomstop", duration=ZOOM_IN_TIME)

        time.sleep(SETTLE_TIME)

        print(f"\n=== STEP 2: RUNNING {GRID_ROWS}x{GRID_COLS} FULL CLASSROOM SWEEP ===")
        for row in range(GRID_ROWS):
            # Even rows pan Left -> Right ("right"); Odd rows pan Right -> Left ("left")
            pan_cmd = "right" if (row % 2 == 0) else "left"

            for col in range(GRID_COLS):
                actual_col = col if (row % 2 == 0) else (GRID_COLS - 1 - col)
                filename = os.path.join(OUTPUT_DIR, f"tile_r{row:02d}_c{actual_col:02d}.jpg")

                print(f"\n[Row {row + 1}/{GRID_ROWS}, Col {actual_col + 1}/{GRID_COLS}] Settling {SETTLE_TIME}s...")
                time.sleep(SETTLE_TIME)

                frame = stream.get_frame()
                if frame is not None:
                    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    print(f"  [CAPTURED] {filename} ({frame.shape[1]}x{frame.shape[0]})")
                    captured_files.append(filename)
                    grid_frames[row][actual_col] = frame
                else:
                    print(f"  [WARNING] Dropped frame at Row {row + 1}, Col {actual_col + 1}")

                # Pan horizontally ("right" or "left") to the next column
                if col < GRID_COLS - 1:
                    move_ptz_timed(pan_cmd, "ptzstop", duration=PAN_STEP_TIME)

            # Tilt vertically ("down") to the next row
            if row < GRID_ROWS - 1:
                print(f"\n[Row {row + 1} Complete] Tilting DOWN to Row {row + 2}...")
                move_ptz_timed("down", "ptzstop", duration=TILT_STEP_TIME)

        print("\n=== STEP 3: RETURNING CAMERA UP & ZOOMING OUT ===")
        move_ptz_timed("up", "ptzstop", duration=TILT_STEP_TIME * (GRID_ROWS - 1))
        if ZOOM_IN_TIME > 0:
            move_ptz_timed("zoomout", "zoomstop", duration=ZOOM_IN_TIME)

    finally:
        manual_ptz("ptzstop")
        manual_ptz("zoomstop")
        stream.stop()

    return sorted(captured_files), grid_frames


# =====================================================================
# 5. RAW PANORAMA GENERATION (Hugin CLI -> OpenCV Stitcher -> Grid Canvas)
# =====================================================================
def stitch_with_hugin(image_files):
    """Aligns, optimizes, and blends captured tiles using Hugin CLI tools."""
    print("\n=== STITCHING PIPELINE A: HUGIN GIGAPIXEL ENGINE ===")
    pto_file = "project.pto"
    try:
        subprocess.run(["pto_gen", "-o", pto_file] + image_files, check=True)
        subprocess.run(["cpfind", "-o", pto_file, "--multirow", pto_file], check=True)
        subprocess.run(["autooptimiser", "-a", "-m", "-l", "-s", "-o", pto_file, pto_file], check=True)
        subprocess.run(["nona", "-o", "output_pano", "-m", "TIFF_m", pto_file], check=True)

        remapped_tiffs = sorted(glob.glob("output_pano*.tif"))
        if not remapped_tiffs:
            raise RuntimeError("nona did not produce any output_pano*.tif files.")

        subprocess.run(["enblend", "-o", RAW_PANO_TIF] + remapped_tiffs, check=True)
        pano_img = cv2.imread(RAW_PANO_TIF)
        if pano_img is not None:
            cv2.imwrite(RAW_PANO_JPG, pano_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
            for t in remapped_tiffs:
                try: os.remove(t)
                except OSError: pass
            print(f"[SUCCESS] Hugin Raw Panorama saved to: {RAW_PANO_JPG}")
            return pano_img
        return None
    except Exception as e:
        print(f"[INFO] Hugin CLI unavailable or failed ({e}). Falling back to OpenCV Stitcher...")
        return None


def stitch_with_opencv(image_files, grid_frames):
    """Attempts spherical/cylindrical stitching via OpenCV; falls back to seamless 2D grid canvas so no seat is ever lost."""
    print("\n=== STITCHING PIPELINE B: OPENCV MULTI-ROW STITCHER ===")
    images = [cv2.imread(f) for f in image_files if os.path.exists(f)]
    images = [img for img in images if img is not None]

    if len(images) >= 2:
        for mode_name, mode in [("PANORAMA", cv2.Stitcher_PANORAMA), ("SCANS", cv2.Stitcher_SCANS)]:
            try:
                stitcher = cv2.Stitcher_create(mode)
                status, pano = stitcher.stitch(images)
                if status == cv2.Stitcher_OK and pano is not None:
                    cv2.imwrite(RAW_PANO_JPG, pano, [cv2.IMWRITE_JPEG_QUALITY, 98])
                    cv2.imwrite(RAW_PANO_TIF, pano)
                    print(f"[SUCCESS] OpenCV ({mode_name}) Raw Panorama saved to: {RAW_PANO_JPG} ({pano.shape[1]}x{pano.shape[0]})")
                    return pano
                else:
                    print(f"[WARN] OpenCV {mode_name} mode returned status {status}.")
            except Exception as e:
                print(f"[WARN] OpenCV {mode_name} exception: {e}")

    print("\n=== STITCHING PIPELINE C: EXACT CLASSROOM MATRIX CANVAS (GUARANTEED COVERAGE) ===")
    valid_sample = next((f for row in grid_frames for f in row if f is not None), None)
    if valid_sample is None:
        raise RuntimeError("No valid frames were captured during the sweep!")

    th, tw = valid_sample.shape[:2]
    canvas = np.zeros((GRID_ROWS * th, GRID_COLS * tw, 3), dtype=np.uint8)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            tile = grid_frames[r][c]
            if tile is not None:
                if tile.shape[:2] != (th, tw):
                    tile = cv2.resize(tile, (tw, th))
                canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile

    cv2.imwrite(RAW_PANO_JPG, canvas, [cv2.IMWRITE_JPEG_QUALITY, 98])
    cv2.imwrite(RAW_PANO_TIF, canvas)
    print(f"[SUCCESS] Full Matrix Raw Panorama saved to: {RAW_PANO_JPG} ({canvas.shape[1]}x{canvas.shape[0]})")
    return canvas


# =====================================================================
# 6. SOTA MODEL DEFINITIONS (YOLO Face + 5-Point Alignment + AdaFace IR50)
# =====================================================================
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
        "adaface_ir50_webface4m.ckpt",
        "adaface_ir50_webface4m.pth",
        "adaface_ir50_webface4m.pt",
    ]
    loaded = False
    for wf in weight_candidates:
        if os.path.exists(wf):
            print(f"[*] Loading AdaFace weights from: {wf}")
            ckpt = torch.load(wf, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            clean_sd = {}
            for k, v in sd.items():
                if k.startswith("model."):
                    clean_sd[k[len("model."):]] = v
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


def load_yolo_face(device):
    candidates = [
        ("yolo26n-face.pt", "detect"),
        ("yolov8n-face.pt", "detect"),
        ("yolo26n-face.engine", "detect"),
        ("yolov8n-face.engine", "detect"),
    ]
    for model_path, task in candidates:
        if os.path.exists(model_path):
            print(f"[*] Loading YOLO face detector: {model_path}")
            model = YOLO(model_path, task=task)
            if not model_path.endswith(".engine"):
                model.to(device)
            return model
    raise FileNotFoundError(f"No YOLO face model found! Tried: {[c[0] for c in candidates]}")


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


def align_and_crop_faces_112(pano_tensor_chw, boxes_tensor, kpts_tensor, has_kpts_mask, pano_h, pano_w, device):
    """Performs GPU 5-point affine ArcFace alignment (or 15% margin ROI align fallback) to 112x112."""
    N = boxes_tensor.size(0)
    crops_out = torch.empty((N, 3, 112, 112), device=device, dtype=torch.float32)

    if has_kpts_mask.any():
        idx_k = torch.where(has_kpts_mask)[0]
        kpts_sub = kpts_tensor[idx_k]
        nk = idx_k.numel()
        src_norm = torch.stack([
            (kpts_sub[:, :, 0] / (pano_w - 1.0)) * 2.0 - 1.0,
            (kpts_sub[:, :, 1] / (pano_h - 1.0)) * 2.0 - 1.0
        ], dim=-1)
        pinv = ARCFACE_PINV.to(device).float()
        theta_T = torch.matmul(pinv.unsqueeze(0), src_norm)
        theta = theta_T.permute(0, 2, 1).contiguous()
        grid = F.affine_grid(theta, size=(nk, 3, 112, 112), align_corners=True)
        sampled = F.grid_sample(
            pano_tensor_chw,
            grid.view(1, nk * 112, 112, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        crops_out[idx_k] = sampled.squeeze(0).view(3, nk, 112, 112).permute(1, 0, 2, 3).contiguous()

    if (~has_kpts_mask).any():
        idx_b = torch.where(~has_kpts_mask)[0]
        boxes_m = boxes_tensor[idx_b].clone().float()
        margin_x = (boxes_m[:, 2] - boxes_m[:, 0]) * 0.15
        margin_y = (boxes_m[:, 3] - boxes_m[:, 1]) * 0.15
        boxes_m[:, 0] = torch.clamp(boxes_m[:, 0] - margin_x, min=0.0)
        boxes_m[:, 1] = torch.clamp(boxes_m[:, 1] - margin_y, min=0.0)
        boxes_m[:, 2] = torch.clamp(boxes_m[:, 2] + margin_x, max=float(pano_w))
        boxes_m[:, 3] = torch.clamp(boxes_m[:, 3] + margin_y, max=float(pano_h))
        batch_idx = torch.zeros((boxes_m.size(0), 1), device=device, dtype=torch.float32)
        roi_boxes = torch.cat((batch_idx, boxes_m), dim=1)
        crops_out[idx_b] = ops.roi_align(pano_tensor_chw, roi_boxes, output_size=(112, 112))

    return crops_out.detach()


# =====================================================================
# 7. TILED DETECTION, FEATURE EXTRACTION & RECOGNITION (NO TRACKING)
# =====================================================================
def detect_faces_on_panorama(yolo_model, pano_img, device, use_half):
    """
    Slices the high-resolution panorama into overlapping windows
    so small/distant classroom faces are never lost to downscaling. Merges with global NMS.
    """
    H, W = pano_img.shape[:2]
    all_boxes = []
    all_scores = []
    all_kpts = []
    all_has_kpts = []

    stride = int(TILE_SIZE * (1.0 - TILE_OVERLAP))
    y_starts = list(range(0, max(1, H - TILE_SIZE + 1), stride))
    if H > TILE_SIZE and (H - TILE_SIZE) not in y_starts:
        y_starts.append(H - TILE_SIZE)
    elif H <= TILE_SIZE:
        y_starts = [0]

    x_starts = list(range(0, max(1, W - TILE_SIZE + 1), stride))
    if W > TILE_SIZE and (W - TILE_SIZE) not in x_starts:
        x_starts.append(W - TILE_SIZE)
    elif W <= TILE_SIZE:
        x_starts = [0]

    print(f"[*] Running Tiled YOLO Face Detection across {len(y_starts) * len(x_starts)} high-res patches ({W}x{H})...")

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(H, y0 + TILE_SIZE)
            x1 = min(W, x0 + TILE_SIZE)
            patch = pano_img[y0:y1, x0:x1]

            results = yolo_model.predict(patch, verbose=False, half=use_half, imgsz=640, conf=0.25)
            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue

            boxes = res.boxes.xyxy.cpu().clone()
            scores = res.boxes.conf.cpu().clone()
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0

            has_kpts = (
                res.keypoints is not None and
                res.keypoints.xy is not None and
                res.keypoints.xy.numel() > 0 and
                res.keypoints.xy.shape[-2] >= 5 and
                (res.keypoints.xy.abs().sum() > 1.0).item()
            )
            if has_kpts:
                kpts = res.keypoints.xy.cpu().clone()[:, :5, :]
                kpts[:, :, 0] += x0
                kpts[:, :, 1] += y0
                kpt_flags = torch.ones(boxes.size(0), dtype=torch.bool)
            else:
                kpts = torch.zeros((boxes.size(0), 5, 2), dtype=torch.float32)
                kpt_flags = torch.zeros(boxes.size(0), dtype=torch.bool)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_kpts.append(kpts)
            all_has_kpts.append(kpt_flags)

    if not all_boxes:
        return (
            torch.empty((0, 4), device=device),
            torch.empty((0,), device=device),
            torch.empty((0, 5, 2), device=device),
            torch.empty((0,), dtype=torch.bool, device=device)
        )

    boxes_cat = torch.cat(all_boxes, dim=0).to(device)
    scores_cat = torch.cat(all_scores, dim=0).to(device)
    kpts_cat = torch.cat(all_kpts, dim=0).to(device)
    has_kpts_cat = torch.cat(all_has_kpts, dim=0).to(device)

    # Filter out tiny noise or extreme aspect ratios
    bw = boxes_cat[:, 2] - boxes_cat[:, 0]
    bh = boxes_cat[:, 3] - boxes_cat[:, 1]
    ar = bw / (bh + 1e-6)
    valid_mask = (bw >= MIN_FACE_SIZE) & (bh >= MIN_FACE_SIZE) & (ar >= 0.50) & (ar <= 1.65)

    boxes_cat = boxes_cat[valid_mask]
    scores_cat = scores_cat[valid_mask]
    kpts_cat = kpts_cat[valid_mask]
    has_kpts_cat = has_kpts_cat[valid_mask]

    if boxes_cat.size(0) == 0:
        return boxes_cat, scores_cat, kpts_cat, has_kpts_cat

    # Global Non-Maximum Suppression across overlapping tiles
    keep = ops.nms(boxes_cat, scores_cat, GLOBAL_NMS_IOU)
    return boxes_cat[keep], scores_cat[keep], kpts_cat[keep], has_kpts_cat[keep]


def run_panorama_recognition(pano_img):
    """
    Pure single-panorama pipeline:
      1. Face Detection (Tiled YOLO)
      2. Image Processing (5-point ArcFace Affine Alignment to 112x112 + Normalization)
      3. Feature Extraction (AdaFace IResNet-50 512-D L2-Normalized Embeddings)
      4. Recognition & Deduplication (FAISS Cosine Similarity + Annotated Image + Attendance CSV)
    """
    print("\n=== STEP 4: RUNNING DETECTION, ALIGNMENT, FEATURE EXTRACTION & RECOGNITION ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_half = torch.cuda.is_available()
    print(f"[*] Compute Device: {device} (FP16={use_half})")

    # Load Models & Gallery
    yolo_model = load_yolo_face(device)
    resnet = load_adaface_model(device, use_half=use_half)

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"Missing '{FAISS_INDEX_PATH}' or '{META_PATH}'. Run train_yolo_sota.py first!"
        )

    index = faiss.read_index(FAISS_INDEX_PATH)
    try:
        ref_embeddings = index.reconstruct_n(0, index.ntotal)
    except AttributeError:
        index.make_direct_map()
        ref_embeddings = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    ref_embeddings_tensor = torch.from_numpy(ref_embeddings).to(device).float()
    ref_embeddings_tensor = F.normalize(ref_embeddings_tensor, p=2, dim=1)
    if use_half:
        ref_embeddings_tensor = ref_embeddings_tensor.half()

    with open(META_PATH, "rb") as f:
        saved_data = pickle.load(f)
    target_names = sorted(list(set(saved_data["target_names"])))
    raw_target_names = saved_data["target_names"]
    y_real = saved_data["y_real"]

    # 1. Detection
    boxes, det_scores, kpts, has_kpts = detect_faces_on_panorama(yolo_model, pano_img, device, use_half)
    num_faces = boxes.size(0)
    print(f"[*] Total Unique Faces Detected in Panorama: {num_faces}")

    annotated_pano = pano_img.copy()
    student_best_sim = {name: 0.0 for name in target_names}
    student_detection_count = {name: 0 for name in target_names}

    if num_faces > 0:
        H, W = pano_img.shape[:2]
        pano_tensor_chw = (
            torch.from_numpy(pano_img)
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
            .float()
        )

        # 2. Image Processing (Alignment & Cropping in Batches to prevent OOM on Mega-Panoramas)
        crops_list = []
        B_ALIGN = 64
        for start in range(0, num_faces, B_ALIGN):
            end = min(num_faces, start + B_ALIGN)
            c_batch = align_and_crop_faces_112(
                pano_tensor_chw,
                boxes[start:end],
                kpts[start:end],
                has_kpts[start:end],
                H, W, device
            )
            crops_list.append(c_batch)
        del pano_tensor_chw
        crops_112 = torch.cat(crops_list, dim=0)

        # 3. Feature Extraction (AdaFace IResNet-50 + Horizontal Flip Test-Time Augmentation)
        identities = []
        similarities = []

        with torch.inference_mode():
            for start in range(0, num_faces, B_ALIGN):
                end = min(num_faces, start + B_ALIGN)
                batch = (crops_112[start:end] / 127.5) - 1.0
                batch_flip = torch.flip(batch, dims=[3])
                if use_half:
                    batch = batch.half()
                    batch_flip = batch_flip.half()

                emb1, norms1 = resnet(batch)
                emb2, norms2 = resnet(batch_flip)
                normed_emb = F.normalize(emb1 + emb2, p=2, dim=1)
                norms_flat = ((norms1 + norms2) * 0.5).squeeze(1)

                # 4. Recognition against FAISS Reference Embeddings
                sim_matrix = torch.mm(normed_emb.to(ref_embeddings_tensor.dtype), ref_embeddings_tensor.t())
                max_sims, max_idxs = torch.max(sim_matrix, dim=1)

                for i in range(end - start):
                    sim_val = float(max_sims[i].item())
                    norm_val = float(norms_flat[i].item())
                    matched_idx = int(max_idxs[i].item())

                    if norm_val >= NORM_QUALITY_THRESHOLD and sim_val >= CONFIDENCE_THRESHOLD:
                        pred_name = raw_target_names[y_real[matched_idx]]
                    else:
                        pred_name = "Unknown"

                    identities.append(pred_name)
                    similarities.append(sim_val)

        # Annotate Panorama & Aggregate Attendance Roster
        boxes_np = boxes.cpu().numpy()
        for i in range(num_faces):
            name = identities[i]
            sim = similarities[i]
            x1, y1, x2, y2 = map(int, boxes_np[i])

            if name != "Unknown":
                student_detection_count[name] = student_detection_count.get(name, 0) + 1
                if sim > student_best_sim.get(name, 0.0):
                    student_best_sim[name] = sim
                color = (0, 255, 0)
                label = f"{name} ({sim:.2f})"
            else:
                color = (0, 0, 255)
                label = f"Unknown ({sim:.2f})"

            cv2.rectangle(annotated_pano, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                annotated_pano, label, (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA
            )

    # Save Timestamped & Fixed Output Files
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_raw_jpg = os.path.join(RESULTS_DIR, f"{ts}_raw_panorama.jpg")
    ts_ann_jpg = os.path.join(RESULTS_DIR, f"{ts}_annotated_panorama.jpg")
    ts_csv = os.path.join(RESULTS_DIR, f"{ts}_panorama_attendance.csv")

    cv2.imwrite(RAW_PANO_JPG, pano_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    cv2.imwrite(ts_raw_jpg, pano_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    cv2.imwrite(ANNOTATED_PANO_JPG, annotated_pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(ts_ann_jpg, annotated_pano, [cv2.IMWRITE_JPEG_QUALITY, 95])

    csv_rows = []
    present_count = 0
    for student in target_names:
        is_present = student_detection_count.get(student, 0) > 0
        if is_present:
            present_count += 1
        csv_rows.append({
            "Name": student,
            "Status": "Present" if is_present else "Absent",
            "Max Similarity": round(student_best_sim.get(student, 0.0), 4),
            "Detections in Panorama": student_detection_count.get(student, 0)
        })

    df_out = pd.DataFrame(csv_rows)
    df_out.to_csv(ATTENDANCE_CSV, index=False)
    df_out.to_csv(ts_csv, index=False)

    print("\n=====================================================================")
    print("                PANORAMA ATTENDANCE SUMMARY                          ")
    print("=====================================================================")
    print(f"  Total Enrolled Students : {len(target_names)}")
    print(f"  Present                 : {present_count}")
    print(f"  Absent                  : {len(target_names) - present_count}")
    print(f"  Raw Panorama Image      : {RAW_PANO_JPG}")
    print(f"  Annotated Output Image  : {ANNOTATED_PANO_JPG}")
    print(f"  Attendance CSV Report   : {ATTENDANCE_CSV}")
    print("=====================================================================\n")


if __name__ == "__main__":
    captured_tiles, grid_frames = run_classroom_sweep()
    if len(captured_tiles) >= 1:
        raw_pano = stitch_with_hugin(captured_tiles)
        if raw_pano is None:
            raw_pano = stitch_with_opencv(captured_tiles, grid_frames)
        run_panorama_recognition(raw_pano)
    else:
        print("[ABORT] No images were captured during the sweep.")
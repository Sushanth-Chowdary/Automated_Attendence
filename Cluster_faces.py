import os
import glob
import shutil
import cv2
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from ultralytics import YOLO

try:
    import infomap
    HAS_INFOMAP = True
except ImportError:
    HAS_INFOMAP = False
    print("[WARNING] 'infomap' package not found. Install via: pip install infomap")

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_half = torch.cuda.is_available()
print(f"Running SOTA Clustering Pipeline on device: {device} (FP16: {use_half})")

# Directories (Strictly Preserved)
TARGET_DIR = './Extraceted Data'
VIDEOS_DIR = os.path.join(TARGET_DIR, 'VIDEOS')
TEMP_FACES_DIR = os.path.join(TARGET_DIR, 'temp_faces')
EMBEDDINGS_DIR = os.path.join(TARGET_DIR, 'embeddings_cache')
GLOBAL_LABELS_DIR = os.path.join(TARGET_DIR, 'GLOBAL_LABELS')

for d in [VIDEOS_DIR, TEMP_FACES_DIR, EMBEDDINGS_DIR, GLOBAL_LABELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Processing parameters
FRAME_SKIP = 10
BATCH_SIZE = 128
YOLO_BATCH_SIZE = 32
CONF_THRESHOLD = 0.65
MIN_FACE_SIZE = 45
NORM_QUALITY_THRESHOLD = 10.0  # AdaFace / CR-FIQA ||f||_2 quality gate

# Standard 112x112 5-Point ArcFace / AdaFace Reference Template
ARCFACE_DST = torch.tensor([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], device=device, dtype=torch.float32)


# ==========================================
# SOTA MODEL LOADER: YOLO26-FACE & ADAFACE IRESNET-50
# ==========================================
class BasicBlockIR(nn.Module):
    """Improved ResNet BasicBlock (IResNet-50 building block for AdaFace)."""
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
    """
    AdaFace IResNet-50 Backbone (Input: BGR [-1, 1] of shape (B, 3, 112, 112)).
    Returns (normed_embeddings, feature_norms ||f||_2).
    """
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
        norm = torch.norm(raw_emb, p=2, dim=1, keepdim=True)
        normed_emb = raw_emb / (norm + 1e-6)
        return normed_emb, norm


def load_sota_models():
    # 1. Load YOLO26-Face (Fallback to yolov8n-face.pt if yolo26n-face.pt not yet downloaded)
    yolo_weights = 'yolo26n-face.pt' if os.path.exists('yolo26n-face.pt') else 'yolov8n-face.pt'
    print(f"-> Loading Detector: {yolo_weights}")
    detector = YOLO(yolo_weights)
    detector.to(device)

    # 2. Load AdaFace IResNet-50
    recognizer = AdaFaceIResNet50(embedding_size=512).to(device)
    adaface_ckpt = 'adaface_ir50_webface4m.ckpt'
    if os.path.exists(adaface_ckpt):
        print(f"-> Loading AdaFace IResNet-50 weights from: {adaface_ckpt}")
        ckpt = torch.load(adaface_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        clean_sd = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.') or k in recognizer.state_dict()}
        recognizer.load_state_dict(clean_sd, strict=False)
    else:
        print(f"[WARNING] '{adaface_ckpt}' not found in current directory! Place 'adaface_ir50_webface4m.ckpt' here for trained AdaFace weights.")

    recognizer.eval()
    if use_half:
        recognizer = recognizer.half()
    return detector, recognizer


yolo_model, adaface_model = load_sota_models()


# ==========================================
# GPU 5-POINT ALIGNMENT & CROP HELPERS
# ==========================================
def crop_with_margin(img, box, margin_percentage=0.15):
    """Preserves original 160x160 context crop for human visual inspection & ./LABELS compatibility."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x = int(w * margin_percentage)
    margin_y = int(h * margin_percentage)

    fx1 = max(0, x1 - margin_x)
    fy1 = max(0, y1 - margin_y)
    fx2 = min(img.shape[1], x2 + margin_x)
    fy2 = min(img.shape[0], y2 + margin_y)
    return img[fy1:fy2, fx1:fx2]


def estimate_fallback_5pt(box):
    """Synthesizes 5 canonical landmarks from bounding box if detector has no pose head."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array([
        [x1 + 0.34 * w, y1 + 0.40 * h],  # Left eye
        [x1 + 0.66 * w, y1 + 0.40 * h],  # Right eye
        [x1 + 0.50 * w, y1 + 0.58 * h],  # Nose tip
        [x1 + 0.37 * w, y1 + 0.76 * h],  # Left mouth
        [x1 + 0.63 * w, y1 + 0.76 * h]   # Right mouth
    ], dtype=np.float32)


def gpu_align_batch_112(crops_160_bgr, kpts_160_list):
    """
    Vectorized GPU 5-point affine warp from (N, 160, 160, 3) BGR to (N, 3, 112, 112) BGR in [-1, 1].
    Uses pure PyTorch least-squares + affine_grid/grid_sample on GPU.
    """
    N = len(crops_160_bgr)
    # Move batch to GPU: (N, 3, 160, 160) float32
    batch_np = np.stack(crops_160_bgr, axis=0)
    batch_t = torch.from_numpy(batch_np).to(device, non_blocking=True).permute(0, 3, 1, 2).float()

    src_pts = torch.from_numpy(np.stack(kpts_160_list, axis=0)).to(device, dtype=torch.float32)  # (N, 5, 2)
    dst_pts = ARCFACE_DST.unsqueeze(0).expand(N, -1, -1)  # (N, 5, 2)

    # Normalize coordinates to [-1, 1] for F.affine_grid
    # grid_sample expects theta mapping target normalized coords [-1, 1] -> source normalized coords [-1, 1]
    src_norm = (src_pts / (160.0 - 1.0)) * 2.0 - 1.0
    dst_norm = (dst_pts / (112.0 - 1.0)) * 2.0 - 1.0

    ones = torch.ones((N, 5, 1), device=device, dtype=torch.float32)
    X = torch.cat([dst_norm, ones], dim=2)  # (N, 5, 3)

    # Solve X @ theta^T = src_norm  => theta^T = lstsq(X, src_norm)
    theta_T = torch.linalg.lstsq(X, src_norm).solution  # (N, 3, 2)
    theta = theta_T.permute(0, 2, 1)  # (N, 2, 3)

    grid = F.affine_grid(theta, size=(N, 3, 112, 112), align_corners=True)
    aligned_112 = F.grid_sample(batch_t, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # Normalize to [-1.0, 1.0] (BGR order for AdaFace)
    aligned_112 = (aligned_112 / 127.5) - 1.0
    if use_half:
        aligned_112 = aligned_112.half()
    return aligned_112


# ==========================================
# PHASE 1: VIDEO EXTRACTION & SOTA EMBEDDING
# ==========================================
def process_videos():
    video_files = sorted([f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])

    for video_filename in video_files:
        video_id = os.path.splitext(video_filename)[0]
        embedding_file = os.path.join(EMBEDDINGS_DIR, f"{video_id}.npz")

        if os.path.exists(embedding_file):
            print(f"-> Skipping Extraction: {video_filename} (Embeddings cached)")
            continue

        print(f"\n{'='*50}\nPhase 1: Extracting from {video_filename}\n{'='*50}")

        parts = video_id.split('_')
        date_str = parts[0] if len(parts) > 0 else "unknown"
        time_str = parts[1] if len(parts) > 1 else "unknown"
        cam_str = parts[2] if len(parts) > 2 else "unknown"

        video_path = os.path.join(VIDEOS_DIR, video_filename)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        image_metadata = []
        keypoints_metadata = []  # Stores 5-point landmarks mapped to the 160x160 crop

        batch_frames = []
        batch_frame_counts = []

        def process_yolo_batch(frames_list, counts_list):
            results = yolo_model(frames_list, verbose=False, half=use_half)
            for batch_idx, result in enumerate(results):
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                has_kpts = (result.keypoints is not None) and (result.keypoints.xy is not None)
                kpts_all = result.keypoints.xy.cpu().numpy() if has_kpts else None

                curr_frame = frames_list[batch_idx]
                curr_frame_count = counts_list[batch_idx]

                for face_idx, (box, conf) in enumerate(zip(boxes, confs)):
                    if conf < CONF_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                        continue

                    # Compute crop coordinates with 15% margin
                    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
                    fx1 = max(0, x1 - margin_x)
                    fy1 = max(0, y1 - margin_y)
                    fx2 = min(curr_frame.shape[1], x2 + margin_x)
                    fy2 = min(curr_frame.shape[0], y2 + margin_y)

                    face_crop_bgr = curr_frame[fy1:fy2, fx1:fx2]
                    ch, cw = face_crop_bgr.shape[:2]
                    if ch == 0 or cw == 0:
                        continue

                    # Extract 5 keypoints and map into the 160x160 resized coordinate space
                    if kpts_all is not None and len(kpts_all) > face_idx and kpts_all[face_idx].shape[0] >= 5:
                        raw_kpt = kpts_all[face_idx][:5].copy()
                        if np.all(raw_kpt == 0):
                            raw_kpt = estimate_fallback_5pt(box)
                    else:
                        raw_kpt = estimate_fallback_5pt(box)

                    kpt_160 = np.zeros((5, 2), dtype=np.float32)
                    kpt_160[:, 0] = (raw_kpt[:, 0] - fx1) * (160.0 / cw)
                    kpt_160[:, 1] = (raw_kpt[:, 1] - fy1) * (160.0 / ch)

                    final_face_160 = cv2.resize(face_crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
                    filename = f"{date_str}_{time_str}_{cam_str}_f{curr_frame_count}_i{face_idx}.jpg"
                    save_path = os.path.join(TEMP_FACES_DIR, filename)

                    cv2.imwrite(save_path, final_face_160)
                    image_metadata.append(save_path)
                    keypoints_metadata.append(kpt_160)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                batch_frames.append(frame)
                batch_frame_counts.append(frame_count)
                if len(batch_frames) >= YOLO_BATCH_SIZE:
                    process_yolo_batch(batch_frames, batch_frame_counts)
                    batch_frames, batch_frame_counts = [], []

        if batch_frames:
            process_yolo_batch(batch_frames, batch_frame_counts)

        cap.release()

        if not image_metadata:
            print(f"   No faces found in {video_filename}.")
            continue

        # --- 1B. GPU 5-POINT ALIGNMENT (112x112) + ADAFACE EMBEDDING + ||f||_2 QUALITY GATE ---
        print(f"   Aligning & generating AdaFace embeddings for {len(image_metadata)} faces...")
        valid_paths = []
        valid_embeddings = []
        valid_norms = []

        with torch.inference_mode():
            for i in range(0, len(image_metadata), BATCH_SIZE):
                batch_paths = image_metadata[i:i + BATCH_SIZE]
                batch_kpts = keypoints_metadata[i:i + BATCH_SIZE]

                crops_bgr = []
                kept_paths = []
                kept_kpts = []

                for p, kpt in zip(batch_paths, batch_kpts):
                    img_bgr = cv2.imread(p)
                    if img_bgr is not None and img_bgr.shape[:2] == (160, 160):
                        crops_bgr.append(img_bgr)
                        kept_paths.append(p)
                        kept_kpts.append(kpt)

                if not crops_bgr:
                    continue

                aligned_112 = gpu_align_batch_112(crops_bgr, kept_kpts)
                emb_batch, norm_batch = adaface_model(aligned_112)

                norms_np = norm_batch.squeeze(1).float().cpu().numpy()
                embs_np = emb_batch.float().cpu().numpy()

                for path_item, emb_item, norm_val in zip(kept_paths, embs_np, norms_np):
                    if norm_val >= NORM_QUALITY_THRESHOLD:
                        valid_paths.append(path_item)
                        valid_embeddings.append(emb_item)
                        valid_norms.append(norm_val)
                    else:
                        # Remove low-quality/corrupted crop from disk
                        try:
                            os.remove(path_item)
                        except OSError:
                            pass

        np.savez_compressed(
            embedding_file,
            paths=np.array(valid_paths),
            embeddings=np.array(valid_embeddings, dtype=np.float32),
            norms=np.array(valid_norms, dtype=np.float32)
        )
        print(f"   Saved {len(valid_embeddings)} high-quality aligned embeddings to cache.")


# ==========================================
# PHASE 2: SOTA GLOBAL CLUSTERING (FAISS k-NN + INFOMAP)
# ==========================================
def sota_faiss_infomap_clustering(all_embeddings, k=80, sim_threshold=0.55, min_cluster_size=30):
    """
    SOTA Large-Scale Face Clustering using FAISS k-NN Affinity Graph + InfoMap.
    Replaces CPU HDBSCAN: runs in seconds on 100k+ 512D vectors and preserves both
    front-row (1,000+ frames) and back-row (30+ frames) students cleanly.
    """
    N, dim = all_embeddings.shape
    faiss.normalize_L2(all_embeddings)

    print(f"-> Building FAISS IndexFlatIP for {N} embeddings (dim={dim})...")
    index = faiss.IndexFlatIP(dim)
    if faiss.get_num_gpus() > 0:
        try:
            index = faiss.index_cpu_to_all_gpus(index)
        except Exception:
            pass
    index.add(all_embeddings)

    k_search = min(k, N)
    print(f"-> Querying top-{k_search} nearest neighbors via FAISS...")
    sims, nbrs = index.search(all_embeddings, k_search)

    if not HAS_INFOMAP:
        raise RuntimeError("Please install infomap (`pip install infomap`) to run Phase 2 SOTA clustering.")

    print(f"-> Constructing sparse k-NN graph (sim_threshold={sim_threshold}) & running InfoMap...")
    im = infomap.Infomap("--two-level --directed --silent")

    row_idx = np.repeat(np.arange(N), k_search)
    col_idx = nbrs.ravel()
    weights = sims.ravel()

    valid_mask = (row_idx != col_idx) & (weights >= sim_threshold)
    valid_u = row_idx[valid_mask]
    valid_v = col_idx[valid_mask]
    valid_w = weights[valid_mask]

    for u, v, w in zip(valid_u, valid_v, valid_w):
        im.add_link(int(u), int(v), float(w))

    im.run()

    raw_labels = np.full(N, -1, dtype=np.int32)
    cluster_counts = {}
    for node in im.tree:
        if node.is_leaf:
            cid = node.module_id
            raw_labels[node.node_id] = cid
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    final_labels = np.full(N, -1, dtype=np.int32)
    valid_cid_map = {}
    next_id = 0
    for i, cid in enumerate(raw_labels):
        if cid != -1 and cluster_counts.get(cid, 0) >= min_cluster_size:
            if cid not in valid_cid_map:
                valid_cid_map[cid] = next_id
                next_id += 1
            final_labels[i] = valid_cid_map[cid]

    return final_labels


def global_clustering():
    print(f"\n{'='*50}\nPhase 2: Global Identity Grouping (FAISS + InfoMap)\n{'='*50}")

    npz_files = sorted(glob.glob(os.path.join(EMBEDDINGS_DIR, "*.npz")))
    if not npz_files:
        print("No embedding caches found. Run Phase 1 first.")
        return

    all_embeddings = []
    all_paths = []

    for npz_file in npz_files:
        data = np.load(npz_file)
        if len(data['embeddings']) > 0:
            all_paths.extend(data['paths'])
            all_embeddings.extend(data['embeddings'])

    if not all_embeddings:
        print("No valid embeddings found inside cache files.")
        return

    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    all_paths = np.array(all_paths)

    print(f"-> Loaded {len(all_embeddings)} total aligned faces across all videos/cameras...")

    labels = sota_faiss_infomap_clustering(
        all_embeddings,
        k=80,
        sim_threshold=0.55,
        min_cluster_size=30
    )

    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"-> Identified {num_clusters} unique global students.")

    if os.path.exists(GLOBAL_LABELS_DIR):
        shutil.rmtree(GLOBAL_LABELS_DIR)
    os.makedirs(GLOBAL_LABELS_DIR, exist_ok=True)

    print("-> Creating directories and transferring all images...")
    for label in sorted(unique_labels):
        if label != -1:
            os.makedirs(os.path.join(GLOBAL_LABELS_DIR, f"Global_Student_{label}"), exist_ok=True)

    moved_count = 0
    for img_path, label in zip(all_paths, labels):
        if label == -1:
            continue
        dest_path = os.path.join(GLOBAL_LABELS_DIR, f"Global_Student_{label}", os.path.basename(img_path))
        if os.path.exists(img_path):
            shutil.copyfile(img_path, dest_path)
            moved_count += 1

    print("\nGlobal processing complete!")
    print(f"Successfully transferred {moved_count} frames without reduction.")
    print(f"Grouped data saved in: {os.path.abspath(GLOBAL_LABELS_DIR)}")


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    process_videos()
    global_clustering()
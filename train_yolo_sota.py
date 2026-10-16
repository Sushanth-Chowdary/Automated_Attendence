# train_yolo_sota.py  –  SOTA Gallery Enrollment & FAISS Index Builder
# =====================================================================
# Upgrades:
#   1. YOLO26-Face with 5-point keypoints (fallback chain)
#   2. GPU 5-point affine warp to 112x112 (ArcFace reference)
#   3. AdaFace IResNet-50 feature extraction (512-D)
#   4. Feature-norm (||f||_2) quality gating (replaces Laplacian blur)
#   5. Sub-Center + Centroid gallery augmentation per class
# =====================================================================

# 1. Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os
import cv2
from collections import OrderedDict
import pickle
import faiss

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
            print(f"Loading AdaFace weights from: {wf}")
            ckpt = torch.load(wf, map_location='cpu', weights_only=False)
            sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            clean_sd = {}
            for k, v in sd.items():
                if k.startswith('model.'):
                    clean_sd[k[len('model.'):]] = v
                elif k in model.state_dict():
                    clean_sd[k] = v
            missing, unexpected = model.load_state_dict(clean_sd, strict=False)
            loaded = True
            break
    if not loaded:
        raise FileNotFoundError("AdaFace checkpoint 'adaface_ir50_webface4m.ckpt' not found!")
    model.eval().to(device)
    if use_half:
        model = model.half()
    return model


# ==========================================
# YOLO MODEL LOADER (FALLBACK CHAIN)
# ==========================================
def load_yolo_face(device, use_engine=True):
    """Load YOLO-Face with fallback: .engine (no .to(device)) -> .pt (with .to(device))"""
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
            print(f"Loading YOLO face model: {model_path}")
            model = YOLO(model_path, task=task)
            if not model_path.endswith('.engine'):
                model.to(device)
            return model
    raise FileNotFoundError(f"No YOLO face model found! Tried: {[c[0] for c in candidates]}")


# ==========================================
# GPU ALIGNMENT UTILITIES (FLOAT32)
# ==========================================
ARCFACE_DST = torch.tensor([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], dtype=torch.float32)


def synthesize_5pt_from_boxes(boxes):
    """
    Synthesize 5 canonical facial keypoints from bounding boxes.
    boxes: (N, 4) [x1, y1, x2, y2] tensor on GPU
    Returns: (N, 5, 2) keypoints tensor float32
    """
    boxes = boxes.float()
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w, h = x2 - x1, y2 - y1
    kpts = torch.stack([
        torch.stack([x1 + 0.34 * w, y1 + 0.40 * h], dim=-1),  # Left eye
        torch.stack([x1 + 0.66 * w, y1 + 0.40 * h], dim=-1),  # Right eye
        torch.stack([x1 + 0.50 * w, y1 + 0.58 * h], dim=-1),  # Nose tip
        torch.stack([x1 + 0.37 * w, y1 + 0.76 * h], dim=-1),  # Left mouth
        torch.stack([x1 + 0.63 * w, y1 + 0.76 * h], dim=-1),  # Right mouth
    ], dim=1)  # (N, 5, 2)
    return kpts.float()


def gpu_align_faces_112(frame_tensor_chw, keypoints_5pt, dst_template):
    """
    GPU vectorized 5-point affine warp to 112x112 in float32.
    frame_tensor_chw: (1, 3, H, W) float32 BGR tensor on GPU
    keypoints_5pt:    (N, 5, 2) float32 pixel coordinates on GPU
    dst_template:     (5, 2) ARCFACE_DST on GPU
    Returns:          (N, 3, 112, 112) aligned BGR float32 crops on GPU
    """
    N = keypoints_5pt.size(0)
    if N == 0:
        return torch.empty((0, 3, 112, 112), device=frame_tensor_chw.device, dtype=torch.float32)

    _, _, H, W = frame_tensor_chw.shape
    frame_tensor_chw = frame_tensor_chw.float()
    keypoints_5pt = keypoints_5pt.float()
    dst_template = dst_template.float()

    # Normalize source keypoints to [-1, 1]
    src_norm = torch.stack([
        (keypoints_5pt[:, :, 0] / (W - 1.0)) * 2.0 - 1.0,
        (keypoints_5pt[:, :, 1] / (H - 1.0)) * 2.0 - 1.0
    ], dim=-1).float()  # (N, 5, 2)

    # Normalize destination template to [-1, 1]
    dst_pts = dst_template.unsqueeze(0).expand(N, -1, -1).float()
    dst_norm = ((dst_pts / (112.0 - 1.0)) * 2.0 - 1.0).float()  # (N, 5, 2)

    # Solve affine: X @ theta_T = src_norm, where X = [dst_norm, 1]
    ones = torch.ones((N, 5, 1), device=keypoints_5pt.device, dtype=torch.float32)
    X = torch.cat([dst_norm, ones], dim=2).float()  # (N, 5, 3)
    theta_T = torch.linalg.lstsq(X, src_norm).solution.float()  # (N, 3, 2)
    theta = theta_T.permute(0, 2, 1)  # (N, 2, 3)

    expanded_frames = frame_tensor_chw.expand(N, -1, -1, -1)
    grid = F.affine_grid(theta, size=(N, 3, 112, 112), align_corners=True)
    crops_112 = F.grid_sample(expanded_frames, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
    return crops_112.detach().float()


# ==========================================
# NORM-BASED QUALITY THRESHOLD
# ==========================================
NORM_QUALITY_THRESHOLD = 10.0


# ==========================================
# 2. SETUP DEVICES & MODELS
# ==========================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

yolo_model = load_yolo_face(device)
resnet = load_adaface_model(device)

dataset_path = './LABELS'
BATCH_SIZE = 32

# Move ARCFACE_DST to device
arcface_dst = ARCFACE_DST.to(device)


# ==========================================
# PHASE 1: Smart Extract & Save Embeddings
# ==========================================
print("\n--- PHASE 1: Extracting and Saving Embeddings (SOTA Pipeline) ---")

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)

    if os.path.isdir(person_dir):
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_images_in_folder = len(image_files)

        if num_images_in_folder == 0:
            continue

        # Use _sota_embeddings.csv so old FaceNet CSVs don't cause Phase 1 to skip
        csv_path = os.path.join(person_dir, f"{person_name}_sota_embeddings.csv")
        needs_processing = True

        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                num_rows_in_csv = len(existing_df)
                if num_rows_in_csv == num_images_in_folder:
                    print(f"Skipping {person_name}: Up-to-date ({num_images_in_folder} images).")
                    needs_processing = False
                else:
                    print(f"Updating {person_name}: Folder has {num_images_in_folder} images, but CSV has {num_rows_in_csv}. Recalculating...")
            except Exception:
                print(f"Updating {person_name}: CSV corrupted, recalculating...")

        if not needs_processing:
            continue

        valid_face_tensors = []
        person_norms = []

        for image_name in image_files:
            image_path = os.path.join(person_dir, image_name)
            try:
                img_cv2 = cv2.imread(image_path)
                if img_cv2 is None:
                    continue

                h, w = img_cv2.shape[:2]

                # Convert image to GPU tensor (BGR, float32)
                img_tensor = torch.from_numpy(img_cv2).to(device).float()
                img_chw = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

                # Run YOLO on the image to get face box + keypoints
                results = yolo_model(img_cv2, verbose=False)
                boxes_xyxy = results[0].boxes.xyxy

                if len(boxes_xyxy) == 0:
                    # --- SMART BYPASS for 160x160 crops with no YOLO detection ---
                    if h == 160 and w == 160:
                        # Synthesize keypoints from inner 85% box [12, 12, 148, 148]
                        fallback_box = torch.tensor([[12.0, 12.0, 148.0, 148.0]], device=device)
                        kpts = synthesize_5pt_from_boxes(fallback_box)
                    else:
                        print(f"  -> No face detected in {image_name}, skipping.")
                        continue
                else:
                    boxes_gpu = boxes_xyxy.to(device).float()

                    # Pick largest face
                    areas = (boxes_gpu[:, 2] - boxes_gpu[:, 0]) * (boxes_gpu[:, 3] - boxes_gpu[:, 1])
                    largest_idx = torch.argmax(areas).item()
                    best_box = boxes_gpu[largest_idx:largest_idx+1]  # (1, 4)

                    # Try to get keypoints from YOLO
                    has_kpts = (results[0].keypoints is not None and
                                results[0].keypoints.xy is not None and
                                results[0].keypoints.xy.numel() > 0)

                    if has_kpts:
                        all_kpts = results[0].keypoints.xy.to(device).float()  # (N, 5, 2)
                        kpts = all_kpts[largest_idx:largest_idx+1]  # (1, 5, 2)

                        # Validate: check for (0,0) keypoints
                        kpts_valid = (kpts.abs().sum(dim=-1) > 1.0).all()
                        if not kpts_valid:
                            kpts = synthesize_5pt_from_boxes(best_box)
                    else:
                        kpts = synthesize_5pt_from_boxes(best_box)

                # GPU affine warp to 112x112
                crop_112 = gpu_align_faces_112(img_chw, kpts, arcface_dst)  # (1, 3, 112, 112)
                valid_face_tensors.append(crop_112)

            except Exception as e:
                print(f"  -> Failed on image {image_name}: {e}")

        # Batch Processing with AdaFace
        person_embeddings = []
        if valid_face_tensors:
            for i in range(0, len(valid_face_tensors), BATCH_SIZE):
                batch = valid_face_tensors[i:i + BATCH_SIZE]
                batch_tensor = torch.cat(batch, dim=0).to(device)  # (B, 3, 112, 112)

                # Normalize to [-1, 1] keeping BGR order (matches AdaFace training)
                batch_tensor = (batch_tensor / 127.5) - 1.0

                with torch.no_grad():
                    normed_embs, feat_norms = resnet(batch_tensor)
                    # normed_embs: (B, 512), feat_norms: (B, 1)

                # Quality gate: filter by feature norm
                norms_flat = feat_norms.squeeze(1)  # (B,)
                quality_mask = norms_flat >= NORM_QUALITY_THRESHOLD

                valid_embs = normed_embs[quality_mask].detach().cpu().numpy()
                valid_norms = norms_flat[quality_mask].detach().cpu().numpy()

                rejected = (~quality_mask).sum().item()
                if rejected > 0:
                    print(f"  -> {person_name}: Rejected {rejected} low-quality crops (norm < {NORM_QUALITY_THRESHOLD})")

                person_embeddings.extend(valid_embs)

        if person_embeddings:
            df = pd.DataFrame(person_embeddings)
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(person_embeddings)} SOTA embeddings to {csv_path}")


# ==========================================
# PHASE 2: Load Data + Sub-Center Augmentation
# ==========================================
print("\n--- PHASE 2: Loading Data & Computing Sub-Centers ---")

X_real = []
y_real = []
target_names = []
name_to_label = {}
current_label_id = 0

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_dir):
        csv_path = os.path.join(person_dir, f"{person_name}_sota_embeddings.csv")
        if os.path.exists(csv_path):
            if person_name not in name_to_label:
                name_to_label[person_name] = current_label_id
                target_names.append(person_name)
                current_label_id += 1

            label_id = name_to_label[person_name]
            df = pd.read_csv(csv_path)
            class_embs = np.ascontiguousarray(df.values, dtype='float32')
            M = len(class_embs)

            if M == 0:
                continue

            # L2-normalize individual embeddings
            faiss.normalize_L2(class_embs)

            # 1. Add all individual L2-normalized embeddings
            X_real.extend(class_embs.tolist())
            y_real.extend([label_id] * M)

            # 2. Compute Class Centroid
            centroid = np.mean(class_embs, axis=0, keepdims=True).astype('float32')
            faiss.normalize_L2(centroid)
            X_real.extend(centroid.tolist())
            y_real.extend([label_id])

            # 3. Compute K = min(3, M) Sub-Centers via spherical K-Means
            if M >= 3:
                k_sub = min(3, M)
                kmeans = faiss.Kmeans(d=512, k=k_sub, niter=20, spherical=True, verbose=False)
                kmeans.train(class_embs)
                sub_centers = kmeans.centroids.astype('float32')
                faiss.normalize_L2(sub_centers)
                X_real.extend(sub_centers.tolist())
                y_real.extend([label_id] * k_sub)
                print(f"  {person_name}: {M} embs + 1 centroid + {k_sub} sub-centers")
            else:
                print(f"  {person_name}: {M} embs + 1 centroid (too few for sub-centers)")

print(f"Loaded a total of {len(X_real)} embeddings across {len(target_names)} classes.")

# ==========================================
# PHASE 3: Build FAISS Index
# ==========================================
print("\nBuilding FAISS Index...")

X_real = np.array(X_real).astype('float32')
y_real = np.array(y_real)
faiss.normalize_L2(X_real)

embedding_dimension = 512

if len(X_real) > 10000:
    print("Large dataset detected. Training FAISS IndexIVFFlat...")
    nlist = min(1000, len(X_real) // 39)
    quantizer = faiss.IndexFlatIP(embedding_dimension)
    index = faiss.IndexIVFFlat(quantizer, embedding_dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    index.train(X_real)
    index.add(X_real)
    index.nprobe = 20
else:
    print("Small dataset detected. Using brute-force IndexFlatIP...")
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(X_real)

faiss_index_path = './face_attendance_faiss.bin'
faiss.write_index(index, faiss_index_path)

metadata_path = './face_attendance_meta.pkl'
data_to_save = {
    'target_names': target_names,
    'y_real': y_real
}

with open(metadata_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"\nFAISS Index saved successfully to {faiss_index_path}")
print("Training Complete! Ready for testing.")

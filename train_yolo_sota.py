# train_yolo_sota.py  –  SOTA Gallery Enrollment & FAISS Index Builder
# =====================================================================
# Upgrades:
#   1. YOLO26-Face with 5-point keypoints (fallback chain)
#   2. GPU 5-point affine warp to 112x112 (ArcFace reference)
#   3. AdaFace IResNet-50 feature extraction (512-D)
#   4. Feature-norm (||f||_2) quality gating (replaces Laplacian blur)
#   5. Sub-Center + Centroid gallery augmentation per class
# =====================================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
import pandas as pd
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

_dst_norm_const = (ARCFACE_DST / 111.0) * 2.0 - 1.0
_X_const = torch.cat([_dst_norm_const, torch.ones((5, 1), dtype=torch.float32)], dim=1)
ARCFACE_PINV = torch.linalg.pinv(_X_const).contiguous()  # (3, 5)


def align_face_kpts_112(frame_tensor_chw, kpts, arcface_pinv, frame_h, frame_w, device):
    """
    Zero-expansion GPU 5-point affine warp to 112x112 in float32 using precomputed ARCFACE_PINV.
    """
    N = kpts.size(0)
    if N == 0:
        return torch.empty((0, 3, 112, 112), device=device, dtype=torch.float32)

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


# ==========================================
# NORM-BASED QUALITY THRESHOLD
# ==========================================
NORM_QUALITY_THRESHOLD = 6.0


# ==========================================
# 2. SETUP DEVICES & MODELS
# ==========================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

yolo_model = load_yolo_face(device)
resnet = load_adaface_model(device)

dataset_path = './LABELS'
BATCH_SIZE = 32


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

        # Use _sota_v2_embeddings.csv so previous runs are automatically recalculated
        csv_path = os.path.join(person_dir, f"{person_name}_sota_v2_embeddings.csv")
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

                if h == 160 and w == 160:
                    results = yolo_model(img_cv2, verbose=False)
                    has_real_kpts = (
                        results[0].keypoints is not None and
                        results[0].keypoints.xy is not None and
                        results[0].keypoints.xy.numel() > 0 and
                        results[0].keypoints.xy.shape[-2] >= 5 and
                        (results[0].keypoints.xy.abs().sum() > 1.0).item()
                    )
                    if has_real_kpts:
                        img_tensor = torch.from_numpy(img_cv2).to(device).float()
                        img_chw = img_tensor.permute(2, 0, 1).unsqueeze(0)
                        kpts = results[0].keypoints.xy.to(device).float()[0:1, :5, :]
                        crop_112 = align_face_kpts_112(img_chw, kpts, ARCFACE_PINV, h, w, device)
                        valid_face_tensors.append(crop_112)
                    else:
                        # Direct resize from 160x160 to 112x112 preserving the exact 15% margin
                        crop_112_cv = cv2.resize(img_cv2, (112, 112), interpolation=cv2.INTER_LINEAR)
                        crop_112 = torch.from_numpy(crop_112_cv).permute(2, 0, 1).unsqueeze(0).to(device).float()
                        valid_face_tensors.append(crop_112)
                else:
                    results = yolo_model(img_cv2, verbose=False)
                    boxes_xyxy = results[0].boxes.xyxy
                    if len(boxes_xyxy) == 0:
                        print(f"  -> No face detected in {image_name}, skipping.")
                        continue

                    has_real_kpts = (
                        results[0].keypoints is not None and
                        results[0].keypoints.xy is not None and
                        results[0].keypoints.xy.numel() > 0 and
                        results[0].keypoints.xy.shape[-2] >= 5 and
                        (results[0].keypoints.xy.abs().sum() > 1.0).item()
                    )

                    boxes_gpu = boxes_xyxy.to(device).float()
                    areas = (boxes_gpu[:, 2] - boxes_gpu[:, 0]) * (boxes_gpu[:, 3] - boxes_gpu[:, 1])
                    largest_idx = torch.argmax(areas).item()

                    if has_real_kpts:
                        img_tensor = torch.from_numpy(img_cv2).to(device).float()
                        img_chw = img_tensor.permute(2, 0, 1).unsqueeze(0)
                        kpts = results[0].keypoints.xy.to(device).float()[largest_idx:largest_idx+1, :5, :]
                        crop_112 = align_face_kpts_112(img_chw, kpts, ARCFACE_PINV, h, w, device)
                        valid_face_tensors.append(crop_112)
                    else:
                        best_box = boxes_gpu[largest_idx].cpu().numpy()
                        bw = best_box[2] - best_box[0]
                        bh = best_box[3] - best_box[1]
                        margin_x = bw * 0.15
                        margin_y = bh * 0.15
                        x1 = max(0, int(round(best_box[0] - margin_x)))
                        y1 = max(0, int(round(best_box[1] - margin_y)))
                        x2 = min(w, int(round(best_box[2] + margin_x)))
                        y2 = min(h, int(round(best_box[3] + margin_y)))
                        face_crop = img_cv2[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            crop_112_cv = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
                            crop_112 = torch.from_numpy(crop_112_cv).permute(2, 0, 1).unsqueeze(0).to(device).float()
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
        csv_path = os.path.join(person_dir, f"{person_name}_sota_v2_embeddings.csv")
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

# Save to BOTH face_attendance_sota_faiss.bin AND face_attendance_faiss.bin
faiss_sota_path = './face_attendance_sota_faiss.bin'
faiss_standard_path = './face_attendance_faiss.bin'
faiss.write_index(index, faiss_sota_path)
faiss.write_index(index, faiss_standard_path)

meta_sota_path = './face_attendance_sota_meta.pkl'
meta_standard_path = './face_attendance_meta.pkl'
data_to_save = {
    'target_names': target_names,
    'y_real': y_real
}

with open(meta_sota_path, 'wb') as f:
    pickle.dump(data_to_save, f)

with open(meta_standard_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"\nFAISS Index saved successfully to {faiss_sota_path} and {faiss_standard_path}")
print("Training Complete! Ready for testing.")

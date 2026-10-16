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


def load_adaface_model(device):
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

            # Handle various checkpoint formats
            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt:
                    sd = ckpt['state_dict']
                elif 'model' in ckpt:
                    sd = ckpt['model']
                else:
                    sd = ckpt
            else:
                sd = ckpt

            # Strip 'model.' prefix if present
            new_sd = OrderedDict()
            for k, v in sd.items():
                name = k.replace('model.', '') if k.startswith('model.') else k
                new_sd[name] = v

            model.load_state_dict(new_sd, strict=False)
            loaded = True
            break

    if not loaded:
        print("WARNING: No AdaFace pretrained weights found! Model will have random weights.")
        print("  Expected one of:", weight_candidates)

    model.eval().to(device)
    return model


# ==========================================
# YOLO MODEL LOADER (FALLBACK CHAIN)
# ==========================================
def load_yolo_face(device):
    """Load YOLO-Face with fallback: yolo26n-face.engine -> yolo26n-face.pt -> yolov8n-face.engine -> yolov8n-face.pt"""
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
# GPU ALIGNMENT UTILITIES
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
    Returns: (N, 5, 2) keypoints tensor
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w, h = x2 - x1, y2 - y1
    kpts = torch.stack([
        torch.stack([x1 + 0.34 * w, y1 + 0.40 * h], dim=-1),  # Left eye
        torch.stack([x1 + 0.66 * w, y1 + 0.40 * h], dim=-1),  # Right eye
        torch.stack([x1 + 0.50 * w, y1 + 0.58 * h], dim=-1),  # Nose tip
        torch.stack([x1 + 0.37 * w, y1 + 0.76 * h], dim=-1),  # Left mouth
        torch.stack([x1 + 0.63 * w, y1 + 0.76 * h], dim=-1),  # Right mouth
    ], dim=1)  # (N, 5, 2)
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

    # Normalize source keypoints to [-1, 1]
    src_norm = torch.stack([
        (keypoints_5pt[:, :, 0] / (W - 1.0)) * 2.0 - 1.0,
        (keypoints_5pt[:, :, 1] / (H - 1.0)) * 2.0 - 1.0
    ], dim=-1)  # (N, 5, 2)

    # Normalize destination template to [-1, 1]
    dst_pts = dst_template.unsqueeze(0).expand(N, -1, -1)
    dst_norm = (dst_pts / (112.0 - 1.0)) * 2.0 - 1.0  # (N, 5, 2)

    # Solve affine: X @ theta_T = src_norm, where X = [dst_norm, 1]
    ones = torch.ones((N, 5, 1), device=keypoints_5pt.device, dtype=torch.float32)
    X = torch.cat([dst_norm, ones], dim=2)  # (N, 5, 3)
    theta_T = torch.linalg.lstsq(X, src_norm).solution  # (N, 3, 2)
    theta = theta_T.permute(0, 2, 1)  # (N, 2, 3)

    expanded_frames = frame_tensor_chw.expand(N, -1, -1, -1)
    grid = F.affine_grid(theta, size=(N, 3, 112, 112), align_corners=True)
    crops_112 = F.grid_sample(expanded_frames, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)
    return crops_112.detach()


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
            class_embs = df.values.astype('float32')
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

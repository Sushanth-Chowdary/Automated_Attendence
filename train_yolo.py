# 1. Imports
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import pickle
import faiss

# ==========================================
# ALIGNMENT HELPER FUNCTIONS
# ==========================================
def crop_standard(img, box):
    """Fallback standard crop with 15% margin."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img.shape[1], x2 + margin_x)
    y2 = min(img.shape[0], y2 + margin_y)
    
    return img[y1:y2, x1:x2]

def align_face(img, box, keypoints):
    """Rotates the face to align the eyes horizontally."""
    if keypoints is None or len(keypoints) < 2:
        return crop_standard(img, box)
        
    x1, y1, x2, y2 = map(int, box)
    left_eye, right_eye = keypoints[0], keypoints[1]
    
    # Ensure left_eye is physically on the left side of the image
    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye
        
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    
    if dx == 0:
        return crop_standard(img, box)
        
    # Calculate angle for affine transformation
    angle = np.degrees(np.arctan2(dy, dx))
    
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Take a larger 50% margin crop first to prevent losing corners during rotation
    margin_x, margin_y = int(w * 0.5), int(h * 0.5)
    X1 = max(0, cx - w - margin_x)
    Y1 = max(0, cy - h - margin_y)
    X2 = min(img.shape[1], cx + w + margin_x)
    Y2 = min(img.shape[0], cy + h + margin_y)
    
    large_crop = img[Y1:Y2, X1:X2]
    if large_crop.size == 0:
        return crop_standard(img, box)
        
    # Shift eye center to the local coordinate space of the large crop
    eye_center = ((left_eye[0] + right_eye[0]) / 2 - X1, (left_eye[1] + right_eye[1]) / 2 - Y1)
    
    # Rotate the large crop
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated_crop = cv2.warpAffine(large_crop, M, (large_crop.shape[1], large_crop.shape[0]), flags=cv2.INTER_CUBIC)
    
    # Transform the original box center to the new rotated space
    local_cx, local_cy = cx - X1, cy - Y1
    new_cx = M[0, 0] * local_cx + M[0, 1] * local_cy + M[0, 2]
    new_cy = M[1, 0] * local_cx + M[1, 1] * local_cy + M[1, 2]
    
    # Apply final 15% margin on the straightened face
    fin_margin_x, fin_margin_y = int(w * 0.15), int(h * 0.15)
    fx1 = int(new_cx - w/2 - fin_margin_x)
    fy1 = int(new_cy - h/2 - fin_margin_y)
    fx2 = int(new_cx + w/2 + fin_margin_x)
    fy2 = int(new_cy + h/2 + fin_margin_y)
    
    # Boundary checks
    fx1, fy1 = max(0, fx1), max(0, fy1)
    fx2, fy2 = min(rotated_crop.shape[1], fx2), min(rotated_crop.shape[0], fy2)
    
    final_crop = rotated_crop[fy1:fy2, fx1:fx2]
    if final_crop.size == 0:
        return crop_standard(img, box)
        
    return final_crop

# 2. Initialize YOLOv8 and FaceNet
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

yolo_model = YOLO('yolov8n-face.pt')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

dataset_path = './LABELS'

to_tensor = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# ==========================================
# PHASE 1: Smart Extract & Save Embeddings
# ==========================================
print("\n--- PHASE 1: Extracting and Saving Embeddings ---")

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)

    if os.path.isdir(person_dir):
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_images_in_folder = len(image_files)
        
        if num_images_in_folder == 0:
            continue

        csv_path = os.path.join(person_dir, f"{person_name}_embeddings.csv")
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

        person_embeddings = []
        for image_name in image_files:
            image_path = os.path.join(person_dir, image_name)
            try:
                # Load with OpenCV for easier alignment transformations
                img_cv2 = cv2.imread(image_path)
                if img_cv2 is None:
                    continue
                
                # YOLOv8 Detection
                results = yolo_model(img_cv2, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                keypoints = results[0].keypoints.xy.cpu().numpy() if hasattr(results[0], 'keypoints') and results[0].keypoints is not None else None

                if len(boxes) == 0:
                    print(f"  -> No face detected in {image_name}, skipping.")
                    continue
                
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_face_idx = np.argmax(areas)

                box = boxes[largest_face_idx]
                kpts = keypoints[largest_face_idx] if keypoints is not None else None
                
                # Align and crop the face
                face_crop_bgr = align_face(img_cv2, box, kpts)
                
                # --- NEW QUALITY GATE ---
                sharpness = cv2.Laplacian(face_crop_bgr, cv2.CV_64F).var()
                if sharpness < 50.0:
                    print(f"  -> Skipping {image_name}: Too blurry (Sharpness: {sharpness:.1f})")
                    continue
                # ------------------------
                
                # Convert back to format FaceNet expects
                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                face_crop_pil = Image.fromarray(face_crop_rgb)
                
                face = to_tensor(face_crop_pil).unsqueeze(0).to(device)
                face = (face - 0.5) * 2

                embedding = resnet(face).detach().cpu().numpy()[0]
                person_embeddings.append(embedding)

            except Exception as e:
                print(f"  -> Failed on image {image_name}: {e}")

        if person_embeddings:
            df = pd.DataFrame(person_embeddings)
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(person_embeddings)} updated embeddings to {csv_path}")

# ==========================================
# PHASE 2: Load ALL Data
# ==========================================
print("\n--- PHASE 2: Loading Data ---")

X_real = []
y_real = []
target_names = []
name_to_label = {}
current_label_id = 0

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_dir):
        csv_path = os.path.join(person_dir, f"{person_name}_embeddings.csv")
        if os.path.exists(csv_path):
            if person_name not in name_to_label:
                name_to_label[person_name] = current_label_id
                target_names.append(person_name)
                current_label_id += 1

            label_id = name_to_label[person_name]
            df = pd.read_csv(csv_path)
            embeddings = df.values.tolist()
            
            X_real.extend(embeddings)
            y_real.extend([label_id] * len(embeddings))

print(f"Loaded a total of {len(X_real)} embeddings across {len(target_names)} classes.")

# ==========================================
# PHASE 3: Build FAISS Index (Dynamic Fallback)
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
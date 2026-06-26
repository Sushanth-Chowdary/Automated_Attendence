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
# CROP HELPER FUNCTION (Alignment Removed)
# ==========================================
def crop_standard(img, box):
    """Grabs the raw face with a 15% margin for context, matching test_yolo.py"""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img.shape[1], x2 + margin_x)
    y2 = min(img.shape[0], y2 + margin_y)
    
    return img[y1:y2, x1:x2]


# ==========================================
# 2. SETUP DEVICES & MODELS
# ==========================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Using your optimized TensorRT engine
yolo_model = YOLO('yolov8n-face.engine', task='detect') 
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

dataset_path = './LABELS'
BATCH_SIZE = 32  

to_tensor = transforms.Compose([
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

        valid_face_tensors = []  
        person_embeddings = []
        
        for image_name in image_files:
            image_path = os.path.join(person_dir, image_name)
            try:
                img_cv2 = cv2.imread(image_path)
                if img_cv2 is None:
                    continue
                
                h, w = img_cv2.shape[:2]
                
                # --- SMART BYPASS LOGIC ---
                if h == 160 and w == 160:
                    # Assume this is already an MTCNN crop or processed by extract_faces.py
                    face_crop_160 = img_cv2
                else:
                    # Run YOLOv8 Detection for raw, uncropped images
                    results = yolo_model(img_cv2, verbose=False)
                    boxes = results[0].boxes.xyxy.cpu().numpy()

                    if len(boxes) == 0:
                        print(f"  -> No face detected in {image_name}, skipping.")
                        continue
                    
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    largest_face_idx = np.argmax(areas)

                    box = boxes[largest_face_idx]
                    
                    # Standard unaligned crop matching test_yolo.py
                    face_crop_bgr = crop_standard(img_cv2, box)
                    
                    # STRICT RESIZE to 160x160
                    face_crop_160 = cv2.resize(face_crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
                
                # Quality Gate (Applies to both bypassed and YOLO-cropped images)
                sharpness = cv2.Laplacian(face_crop_160, cv2.CV_64F).var()
                if sharpness < 4.0:
                    print(f"  -> Skipping {image_name}: Too blurry (Sharpness: {sharpness:.1f})")
                    continue
                
                # Convert to Tensor for FaceNet
                face_crop_rgb = cv2.cvtColor(face_crop_160, cv2.COLOR_BGR2RGB)
                face_crop_pil = Image.fromarray(face_crop_rgb)
                
                face = to_tensor(face_crop_pil).unsqueeze(0)
                face = (face - 0.5) * 2
                
                valid_face_tensors.append(face)

            except Exception as e:
                print(f"  -> Failed on image {image_name}: {e}")

        # Batch Processing
        if valid_face_tensors:
            for i in range(0, len(valid_face_tensors), BATCH_SIZE):
                batch = valid_face_tensors[i:i + BATCH_SIZE]
                batch_tensor = torch.cat(batch, dim=0).to(device)
                
                with torch.no_grad():
                    batch_embs = resnet(batch_tensor).detach().cpu().numpy()
                
                person_embeddings.extend(batch_embs)

        if person_embeddings:
            df = pd.DataFrame(person_embeddings)
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(person_embeddings)} updated embeddings to {csv_path}")

# ==========================================
# PHASE 2: Load Data
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
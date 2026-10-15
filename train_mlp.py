# 1. Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import pickle

# ==========================================
# MLP MODEL ARCHITECTURE
# ==========================================
class FaceMLP(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super(FaceMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


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
                if sharpness < 5.0:
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

for person_name in sorted(os.listdir(dataset_path)):
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
# PHASE 3: Train MLP Classifier
# ==========================================
print("\n--- PHASE 3: Training MLP Classifier ---")

X_tensor = torch.tensor(X_real, dtype=torch.float32)
X_tensor = torch.nn.functional.normalize(X_tensor, p=2, dim=1)
y_tensor = torch.tensor(y_real, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
# drop_last=False; if last batch has size 1, we handle BatchNorm gracefully or ensure batch size works
train_loader = DataLoader(dataset, batch_size=min(64, max(2, len(X_tensor))), shuffle=True, drop_last=(len(X_tensor) % 64 == 1))

embedding_dimension = 512
num_classes = len(target_names)

mlp_model = FaceMLP(input_dim=embedding_dimension, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

EPOCHS = 50
mlp_model.train()

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        if batch_x.size(0) < 2:
            continue
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = mlp_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    scheduler.step()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        epoch_loss = running_loss / max(total, 1)
        epoch_acc = (correct / max(total, 1)) * 100
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")

mlp_model.eval()
mlp_model_path = './face_attendance_mlp.pth'
torch.save({
    'model_state_dict': mlp_model.state_dict(),
    'input_dim': embedding_dimension,
    'num_classes': num_classes
}, mlp_model_path)

metadata_path = './face_attendance_meta.pkl'
data_to_save = {
    'target_names': target_names,
    'y_real': np.array(y_real)
}

with open(metadata_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"\nMLP Model saved successfully to {mlp_model_path}")
print("Training Complete! Ready for testing.")
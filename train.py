# 1. Imports
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
import pickle
import faiss

# 2. Initialize MTCNN and FaceNet
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

mtcnn = MTCNN(keep_all=False, device=device)
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
                img = Image.open(image_path).convert('RGB')
                face = mtcnn(img)

                if face is None:
                    face = to_tensor(img).unsqueeze(0).to(device)
                    face = (face - 0.5) * 2
                else:
                    if face.ndim == 4:
                        face = face[0].unsqueeze(0)
                    elif face.ndim == 3:
                        face = face.unsqueeze(0)
                    face = face.to(device)

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
# PHASE 3: Build FAISS Index
# ==========================================
print("\nBuilding FAISS Index...")

# 1. FAISS strictly requires float32 arrays
X_real = np.array(X_real).astype('float32')
y_real = np.array(y_real)

# 2. To use Cosine Similarity in FAISS, we must L2-normalize the vectors first
faiss.normalize_L2(X_real)

# 3. Build the Inner Product (IP) Index (512 is the FaceNet dimension)
embedding_dimension = 512
index = faiss.IndexFlatIP(embedding_dimension)
index.add(X_real)

# 4. Save the FAISS index and the metadata
faiss_index_path = './face_attendance_faiss.bin'
faiss.write_index(index, faiss_index_path)

metadata_path = './face_attendance_meta.pkl'
data_to_save = {
    'target_names': target_names,
    'y_real': y_real # We need this to map the FAISS row index back to the label ID
}

with open(metadata_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"\nFAISS Index saved successfully to {faiss_index_path}")
print("Training Complete! Ready for testing.")
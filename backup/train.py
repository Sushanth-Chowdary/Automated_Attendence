# 1. Imports
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
import pickle

# 2. Initialize MTCNN and FaceNet
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Ensure this matches your uppercase folder name!
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
        # 1. Count how many image files are actually in the folder
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_images_in_folder = len(image_files)
        
        if num_images_in_folder == 0:
            continue # Skip empty folders

        csv_path = os.path.join(person_dir, f"{person_name}_embeddings.csv")
        needs_processing = True

        # 2. SMART SKIP LOGIC: Compare folder images to CSV rows
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                num_rows_in_csv = len(existing_df)
                
                # If they match, the folder hasn't changed. Skip it!
                if num_rows_in_csv == num_images_in_folder:
                    print(f"Skipping {person_name}: Up-to-date ({num_images_in_folder} images).")
                    needs_processing = False
                else:
                    print(f"Updating {person_name}: Folder has {num_images_in_folder} images, but CSV has {num_rows_in_csv}. Recalculating...")
            except Exception:
                print(f"Updating {person_name}: CSV corrupted, recalculating...")

        if not needs_processing:
            continue

        # 3. Process the images if needed
        person_embeddings = []
        for image_name in image_files:
            image_path = os.path.join(person_dir, image_name)
            try:
                img = Image.open(image_path).convert('RGB')
                
                # Detect face
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

                # Get embedding
                embedding = resnet(face).detach().cpu().numpy()[0]
                person_embeddings.append(embedding)

            except Exception as e:
                print(f"  -> Failed on image {image_name}: {e}")

        # 4. Overwrite/Save the new CSV
        if person_embeddings:
            df = pd.DataFrame(person_embeddings)
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(person_embeddings)} updated embeddings to {csv_path}")

# ==========================================
# PHASE 2: Load ALL Data and Train Model
# ==========================================
print("\n--- PHASE 2: Loading Data and Retraining Model ---")

X_real = []
y_real = []
target_names = []
name_to_label = {}
current_label_id = 0

# Iterate through folders and gather every single CSV
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

X_real = np.array(X_real)
y_real = np.array(y_real)

print(f"Loaded a total of {len(X_real)} embeddings across {len(target_names)} classes.")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.20, random_state=42)

# Train the Classifier
print("\nTraining the MLP Classifier...")
clf = MLPClassifier(hidden_layer_sizes=(512,), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n--- Training Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=[target_names[i] for i in np.unique(y_test)]))

# Save the updated Model
model_save_path = './face_attendance_model.pkl'
data_to_save = {
    'classifier': clf,
    'target_names': target_names,
    'name_to_label': name_to_label
}

with open(model_save_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"\nModel updated and saved successfully to {model_save_path}")
print("Training Complete! Ready for testing.")
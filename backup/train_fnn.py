# 1. Imports
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
import pickle

# --- NEW KERAS IMPORTS ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Prevent TensorFlow from stealing all GPU VRAM from PyTorch
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 2. Initialize MTCNN and FaceNet
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running PyTorch on device: {device}")

mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

dataset_path = './BALANCED_LABELS'

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
# PHASE 2: Load ALL Data and Train Model
# ==========================================
print("\n--- PHASE 2: Loading Data and Retraining Model ---")

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

X_real = np.array(X_real)
y_real = np.array(y_real)
num_classes = len(target_names)

print(f"Loaded a total of {len(X_real)} embeddings across {num_classes} classes.")

X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.20, random_state=42)

# ==========================================
# PHASE 3: Build & Train Keras FNN
# ==========================================
print("\nTraining the Keras FNN Classifier...")

model = Sequential([
    Dense(512, activation='relu', input_shape=(512,)),
    BatchNormalization(),
    Dropout(0.4), # 40% dropout to prevent overfitting
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax') # Softmax outputs probabilities
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Early stopping will stop training if the validation accuracy stops improving
early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- Training Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=[target_names[i] for i in np.unique(y_test)]))

# ==========================================
# PHASE 4: Save Model and Metadata
# ==========================================
keras_model_path = './face_attendance_model.keras'
metadata_path = './face_attendance_metadata.pkl'

# Save the Keras model (Weights & Architecture)
model.save(keras_model_path)

# Save the label names (Metadata)
with open(metadata_path, 'wb') as f:
    pickle.dump({
        'target_names': target_names,
        'name_to_label': name_to_label
    }, f)

print(f"\nModel saved successfully to {keras_model_path}")
print(f"Metadata saved successfully to {metadata_path}")
print("Training Complete! Ready for testing.")
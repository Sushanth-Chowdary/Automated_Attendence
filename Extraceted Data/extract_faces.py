import cv2
import os
import torch
import numpy as np
import shutil
from ultralytics import YOLO
import hdbscan
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# ==========================================
# CROP HELPER FUNCTION
# ==========================================
def crop_with_margin(img, box, margin_percentage=0.15):
    """
    Crops the face directly from the bounding box with an added margin,
    strictly staying within the frame boundaries to avoid black borders.
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    
    # Calculate margin based on face size
    margin_x = int(w * margin_percentage)
    margin_y = int(h * margin_percentage)
    
    # Apply margin and clip to image dimensions
    fx1 = max(0, x1 - margin_x)
    fy1 = max(0, y1 - margin_y)
    fx2 = min(img.shape[1], x2 + margin_x)
    fy2 = min(img.shape[0], y2 + margin_y)
    
    return img[fy1:fy2, fx1:fx2]

# ==========================================
# 1. SETUP
# ==========================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Load YOLOv8 Face Model
yolo_model = YOLO('yolov8n-face.pt', task='detect')

# Configuration
videos_dir = './videos'              
base_output_dir = './extracted_faces' 
clustered_output_dir = './clustered_faces'

os.makedirs(base_output_dir, exist_ok=True)
os.makedirs(clustered_output_dir, exist_ok=True)

# ==========================================
# 2. PROCESSING LOOP (FACE EXTRACTION)
# ==========================================
print("Scanning the 'videos' folder...")

if not os.path.exists(videos_dir):
    print(f"Error: Could not find folder '{videos_dir}'")
else:
    for video_filename in os.listdir(videos_dir):
        if not (video_filename.lower().endswith(('.mkv', '.mp4'))):
            continue

        video_path = os.path.join(videos_dir, video_filename)
        
        # Extract name
        person_name = video_filename.split('-')[0]
        
        person_output_dir = os.path.join(base_output_dir, person_name)
        os.makedirs(person_output_dir, exist_ok=True)

        print(f"\n--- Processing video for: {person_name} ---")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_faces = 0
        frame_skip = 10  

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % frame_skip == 0:
                # Run YOLOv8 on the raw BGR frame
                results = yolo_model(frame, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                if len(boxes) > 0:
                    # Find the most prominent face (largest area) to avoid background faces
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    largest_face_idx = np.argmax(areas)

                    box = boxes[largest_face_idx]
                    
                    # Crop directly from the frame with a 15% padding margin
                    face_crop_bgr = crop_with_margin(frame, box, margin_percentage=0.15)
                    
                    # Ensure the crop isn't empty before proceeding
                    if face_crop_bgr.size > 0:
                        # STRICT RESIZE: Exactly 160x160 to match FaceNet expectations
                        final_face_160 = cv2.resize(face_crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
                        
                        filename = f"{person_name}_frame_{frame_count}.jpg"
                        save_path = os.path.join(person_output_dir, filename)
                        
                        try:
                            cv2.imwrite(save_path, final_face_160)
                            saved_faces += 1
                        except Exception as e:
                            print(f"      Error saving frame {frame_count}: {e}")

        cap.release()
        print(f"Done! Saved {saved_faces} standardized 160x160 faces for {person_name}.")

print("\nFace extraction completed successfully!")

# ==========================================
# 3. GENERATE EMBEDDINGS (FACENET)
# ==========================================
print("\n--- Starting Embedding Generation ---")
# Load FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Standard transformation for FaceNet (scales pixel values)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

image_paths = []
embeddings = []

# Gather all saved faces
for root, dirs, files in os.walk(base_output_dir):
    for file in files:
        if file.lower().endswith('.jpg'):
            img_path = os.path.join(root, file)
            image_paths.append(img_path)

print(f"Found {len(image_paths)} faces to cluster.")

# Extract embeddings
with torch.no_grad():
    for img_path in image_paths:
        # Load image with PIL (convert to RGB because cv2 saves in BGR)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension
        
        # Get embedding vector
        emb = resnet(img_tensor).cpu().numpy().flatten()
        embeddings.append(emb)

embeddings = np.array(embeddings)

# ==========================================
# 4. HDBSCAN CLUSTERING & ORGANIZATION
# ==========================================
if len(embeddings) > 0:
    print("\n--- Running HDBSCAN Clustering ---")
    
    # Configure HDBSCAN (tweak min_cluster_size based on your dataset size)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(embeddings)
    
    unique_labels = set(labels)
    print(f"Found {len(unique_labels) - (1 if -1 in labels else 0)} clusters (excluding noise).")
    
    # Move images to their respective cluster folders
    for img_path, label in zip(image_paths, labels):
        # label -1 is used by HDBSCAN for noisy/unclustered data
        folder_name = "noise" if label == -1 else f"cluster_{label}"
        
        cluster_folder_path = os.path.join(clustered_output_dir, folder_name)
        os.makedirs(cluster_folder_path, exist_ok=True)
        
        # Copy the image to the new clustered directory
        shutil.copy(img_path, cluster_folder_path)

    print(f"Clustering finished! Images organized in: {clustered_output_dir}")
else:
    print("No embeddings were generated. Skipping clustering.")
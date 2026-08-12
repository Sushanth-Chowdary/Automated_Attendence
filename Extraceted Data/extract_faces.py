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
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    
    margin_x = int(w * margin_percentage)
    margin_y = int(h * margin_percentage)
    
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

# Video directory and Output directories
videos_dir = './VIDEOS/' 
base_output_dir = './extracted_faces_temp' 
labels_dir = './LABELS' 

os.makedirs(base_output_dir, exist_ok=True)
if os.path.exists(labels_dir):
    shutil.rmtree(labels_dir) # Clear previous run labels
os.makedirs(labels_dir, exist_ok=True)

# ==========================================
# 2. MULTI-PERSON FACE EXTRACTION FROM ALL VIDEOS
# ==========================================
print(f"\n--- Extracting ALL faces from videos in: {videos_dir} ---")

image_metadata = [] # Store path, date, and camera info for later
saved_faces = 0
frame_skip = 10 # Sample every 10th frame

for video_filename in os.listdir(videos_dir):
    if not video_filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue
        
    # Parse filename based on standard format shown in your screenshot
    # Example: 20260807_110016_cam1_raw.mp4
    parts = video_filename.split('_')
    if len(parts) >= 3:
        date_str = parts[0]   # '20260807'
        cam_str = parts[2]    # 'cam1' or 'cam2'
    else:
        date_str = "unknown_date"
        cam_str = "unknown_cam"

    video_path = os.path.join(videos_dir, video_filename)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print(f"Processing {video_filename}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % frame_skip == 0:
            results = yolo_model(frame, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # PROCESS ALL DETECTED FACES IN THE FRAME
            for face_idx, box in enumerate(boxes):
                face_crop_bgr = crop_with_margin(frame, box, margin_percentage=0.15)
                
                if face_crop_bgr.size > 0:
                    final_face_160 = cv2.resize(face_crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
                    
                    # Prefix with video name to prevent overwriting frames with the same number from different videos
                    filename = f"{video_filename}_frame_{frame_count}_face_{face_idx}.jpg"
                    save_path = os.path.join(base_output_dir, filename)
                    
                    try:
                        cv2.imwrite(save_path, final_face_160)
                        
                        # Store extraction data for directory structuring later
                        image_metadata.append({
                            'path': save_path,
                            'date': date_str,
                            'cam': cam_str
                        })
                        saved_faces += 1
                    except Exception as e:
                        print(f"Error saving frame {frame_count} in {video_filename}: {e}")

    cap.release()

print(f"Done! Saved {saved_faces} total face crops across all videos.")

# ==========================================
# 3. GENERATE EMBEDDINGS (FACENET)
# ==========================================
print("\n--- Starting Embedding Generation ---")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

embeddings = []

with torch.no_grad():
    for meta in image_metadata:
        img = Image.open(meta['path']).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        emb = resnet(img_tensor).cpu().numpy().flatten()
        embeddings.append(emb)

embeddings = np.array(embeddings)

# ==========================================
# 4. HDBSCAN CLUSTERING TO CREATING LABELS
# ==========================================
if len(embeddings) > 0:
    print("\n--- Running HDBSCAN Clustering ---")
    
    # Adjust min_cluster_size depending on how many frames per student you have
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(embeddings)
    
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    print(f"Found {num_clusters} student clusters (excluding noise).")
    
    # Move images into ./LABELS/<Date>/<Cam>/Student_X directories
    for meta, label in zip(image_metadata, labels):
        if label == -1:
            continue # Skip noise
            
        folder_name = f"Student_{label}"
        
        # Construct the targeted nested directory
        target_folder = os.path.join(labels_dir, meta['date'], meta['cam'], folder_name)
        os.makedirs(target_folder, exist_ok=True)
        
        shutil.copy(meta['path'], target_folder)

    print(f"\nClustering complete! Organized labels saved to: {labels_dir}")
else:
    print("No faces found to cluster.")
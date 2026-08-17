import cv2
import os
import torch
import numpy as np
import shutil
import hdbscan
import torch.nn.functional as F
from ultralytics import YOLO
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

# Load Models
yolo_model = YOLO('yolov8n-face.pt', task='detect')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Directories
videos_dir = './VIDEOS/' 
base_output_dir = './extracted_faces_temp' 
labels_dir = './LABELS' 

# Reset directories
if os.path.exists(base_output_dir):
    shutil.rmtree(base_output_dir)
os.makedirs(base_output_dir, exist_ok=True)

if os.path.exists(labels_dir):
    shutil.rmtree(labels_dir)
os.makedirs(labels_dir, exist_ok=True)

# Restored original frame skip as requested
frame_skip = 10 
batch_size = 128

# ==========================================
# 2. PROCESS VIDEO BY VIDEO
# ==========================================
video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

for video_filename in video_files:
    print(f"\n{'='*50}\nProcessing Video: {video_filename}\n{'='*50}")
    
    # Parse filename: 20260807_110016_cam1_raw.mp4
    parts = video_filename.split('_')
    if len(parts) >= 3:
        date_str = parts[0]   # '20260807'
        time_str = parts[1]   # '110016'
        cam_str = parts[2]    # 'cam1'
    else:
        date_str, time_str, cam_str = "unknown", "unknown", "unknown"

    video_path = os.path.join(videos_dir, video_filename)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_faces_this_video = 0
    image_metadata = [] # Resets for each video
    
    print("-> Extracting faces...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % frame_skip == 0:
            results = yolo_model(frame, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for face_idx, (box, conf) in enumerate(zip(boxes, confs)):
                if conf < 0.65:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                if (x2 - x1) < 45 or (y2 - y1) < 45:
                    continue

                face_crop_bgr = crop_with_margin(frame, box, margin_percentage=0.15)
                
                if face_crop_bgr.size > 0:
                    final_face_160 = cv2.resize(face_crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
                    
                    filename = f"frame_{frame_count}_face_{face_idx}.jpg"
                    save_path = os.path.join(base_output_dir, filename)
                    
                    try:
                        cv2.imwrite(save_path, final_face_160)
                        image_metadata.append(save_path)
                        saved_faces_this_video += 1
                    except Exception as e:
                        print(f"   Error saving frame {frame_count}: {e}")

    cap.release()
    print(f"   Extracted {saved_faces_this_video} faces.")

    # Skip to next video if no faces found
    if saved_faces_this_video == 0:
        continue

    # ------------------------------------------
    # GENERATE EMBEDDINGS (For this video only)
    # ------------------------------------------
    print("-> Generating embeddings...")
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(image_metadata), batch_size):
            batch_paths = image_metadata[i:i+batch_size]
            batch_tensors = []
            
            for img_path in batch_paths:
                img = Image.open(img_path).convert('RGB')
                batch_tensors.append(transform(img))
                
            batch_tensor = torch.stack(batch_tensors).to(device)
            
            emb_batch = resnet(batch_tensor)
            emb_batch = F.normalize(emb_batch, p=2, dim=1).cpu().numpy()
            
            embeddings.extend(emb_batch)

    embeddings = np.array(embeddings)

    # ------------------------------------------
    # CLUSTER AND MOVE FILES (For this video only)
    # ------------------------------------------
    print("-> Running HDBSCAN Clustering...")
    
    # Tuned for a single video with frame_skip=10
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,            # High because frame_skip=10 captures the same student many times
        min_samples=15,                 
        metric='euclidean', 
        cluster_selection_epsilon=0.45, 
        cluster_selection_method='eom'
    )
    
    labels = clusterer.fit_predict(embeddings)
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    print(f"   Found {num_clusters} student clusters.")

    print("-> Moving images to structured folders...")
    for img_path, label in zip(image_metadata, labels):
        if label == -1:
            continue
            
        folder_name = f"Student_{label}"
        
        # New Output Structure: LABELS/date/cam/time/Student_X/
        target_folder = os.path.join(labels_dir, date_str, cam_str, time_str, folder_name)
        os.makedirs(target_folder, exist_ok=True)
        
        shutil.move(img_path, target_folder)

    # Clean out the temp directory so the next video starts fresh
    for filename in os.listdir(base_output_dir):
        file_path = os.path.join(base_output_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            pass

print("\n==================================================")
print("ALL VIDEOS PROCESSED SUCCESSFULLY!")
print(f"Data saved in: {os.path.abspath(labels_dir)}")
print("==================================================")
import cv2
import os
import glob
import torch
import numpy as np
import shutil
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# ==========================================
# GPU CLUSTERING & MULTI-CORE CPU FALLBACK
# ==========================================
try:
    from cuml.cluster import HDBSCAN
    print("Using GPU-accelerated cuML HDBSCAN")
    USE_CUML = True
except ImportError:
    import hdbscan
    print("cuML not found. Falling back to multi-core CPU HDBSCAN")
    USE_CUML = False

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Directories
TARGET_DIR = './Extraceted Data'
VIDEOS_DIR = os.path.join(TARGET_DIR, 'VIDEOS')
TEMP_FACES_DIR = os.path.join(TARGET_DIR, 'temp_faces')
EMBEDDINGS_DIR = os.path.join(TARGET_DIR, 'embeddings_cache')
GLOBAL_LABELS_DIR = os.path.join(TARGET_DIR, 'GLOBAL_LABELS')

# Create necessary directories (Safe creation)
for d in [VIDEOS_DIR, TEMP_FACES_DIR, EMBEDDINGS_DIR, GLOBAL_LABELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Processing parameters
FRAME_SKIP = 10 
BATCH_SIZE = 128
YOLO_BATCH_SIZE = 32  # Added YOLO Batch Inference Size
CONF_THRESHOLD = 0.65
MIN_FACE_SIZE = 45

# Load Models
yolo_model = YOLO('yolov8n-face.pt', task='detect')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==========================================
# HELPER FUNCTIONS
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
# PHASE 1: VIDEO EXTRACTION & EMBEDDING
# ==========================================
def process_videos():
    video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_filename in video_files:
        video_id = os.path.splitext(video_filename)[0]
        embedding_file = os.path.join(EMBEDDINGS_DIR, f"{video_id}.npz")
        
        # DYNAMIC SKIP: If we already extracted embeddings for this video, skip Phase 1
        if os.path.exists(embedding_file):
            print(f"-> Skipping Extraction: {video_filename} (Embeddings cached)")
            continue
            
        print(f"\n{'='*50}\nPhase 1: Extracting from {video_filename}\n{'='*50}")
        
        # Parse filename dynamically (Fallback if format doesn't match)
        parts = video_id.split('_')
        date_str = parts[0] if len(parts) > 0 else "unknown"
        time_str = parts[1] if len(parts) > 1 else "unknown"
        cam_str = parts[2] if len(parts) > 2 else "unknown"

        video_path = os.path.join(VIDEOS_DIR, video_filename)
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        image_metadata = [] 
        
        batch_frames = []
        batch_frame_counts = []
        
        # --- 1A. FACE DETECTION (Batched YOLO Inference) ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                batch_frames.append(frame)
                batch_frame_counts.append(frame_count)
                
                if len(batch_frames) >= YOLO_BATCH_SIZE:
                    results = yolo_model(batch_frames, verbose=False)
                    
                    for batch_idx, result in enumerate(results):
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        curr_frame = batch_frames[batch_idx]
                        curr_frame_count = batch_frame_counts[batch_idx]
                        
                        for face_idx, (box, conf) in enumerate(zip(boxes, confs)):
                            if conf < CONF_THRESHOLD: continue
                            
                            x1, y1, x2, y2 = map(int, box)
                            if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE: continue

                            face_crop_bgr = crop_with_margin(curr_frame, box, margin_percentage=0.15)
                            if face_crop_bgr.size > 0:
                                final_face_160 = cv2.resize(face_crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
                                
                                filename = f"{date_str}_{time_str}_{cam_str}_f{curr_frame_count}_i{face_idx}.jpg"
                                save_path = os.path.join(TEMP_FACES_DIR, filename)
                                
                                cv2.imwrite(save_path, final_face_160)
                                image_metadata.append(save_path)
                                
                    batch_frames = []
                    batch_frame_counts = []

        # Process any remaining frames in the batch
        if batch_frames:
            results = yolo_model(batch_frames, verbose=False)
            
            for batch_idx, result in enumerate(results):
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                curr_frame = batch_frames[batch_idx]
                curr_frame_count = batch_frame_counts[batch_idx]
                
                for face_idx, (box, conf) in enumerate(zip(boxes, confs)):
                    if conf < CONF_THRESHOLD: continue
                    
                    x1, y1, x2, y2 = map(int, box)
                    if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE: continue

                    face_crop_bgr = crop_with_margin(curr_frame, box, margin_percentage=0.15)
                    if face_crop_bgr.size > 0:
                        final_face_160 = cv2.resize(face_crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
                        
                        filename = f"{date_str}_{time_str}_{cam_str}_f{curr_frame_count}_i{face_idx}.jpg"
                        save_path = os.path.join(TEMP_FACES_DIR, filename)
                        
                        cv2.imwrite(save_path, final_face_160)
                        image_metadata.append(save_path)

        cap.release()
        
        if not image_metadata:
            print(f"   No faces found in {video_filename}.")
            continue

        # --- 1B. GENERATE EMBEDDINGS ---
        print(f"   Generating embeddings for {len(image_metadata)} faces...")
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(image_metadata), BATCH_SIZE):
                batch_paths = image_metadata[i:i+BATCH_SIZE]
                batch_tensors = []
                
                for img_path in batch_paths:
                    img = Image.open(img_path).convert('RGB')
                    batch_tensors.append(transform(img))
                    
                batch_tensor = torch.stack(batch_tensors).to(device)
                emb_batch = resnet(batch_tensor)
                emb_batch = F.normalize(emb_batch, p=2, dim=1).cpu().numpy()
                embeddings.extend(emb_batch)

        # Save to cache so we never have to run ResNet on this video again
        np.savez_compressed(embedding_file, paths=image_metadata, embeddings=np.array(embeddings))
        print(f"   Saved {len(embeddings)} embeddings to cache.")

# ==========================================
# PHASE 2: GLOBAL CLUSTERING
# ==========================================
def global_clustering():
    print(f"\n{'='*50}\nPhase 2: Global Identity Grouping\n{'='*50}")
    
    npz_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*.npz"))
    if not npz_files:
        print("No embedding caches found. Run Phase 1 first.")
        return

    all_embeddings = []
    all_paths = []

    # Load all cached embeddings globally
    for npz_file in npz_files:
        data = np.load(npz_file)
        all_paths.extend(data['paths'])
        all_embeddings.extend(data['embeddings'])

    # 1. FORCE FLOAT32 TO HALVE MEMORY USAGE
    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    all_paths = np.array(all_paths)
    
    print(f"-> Loaded {len(all_embeddings)} total faces across all videos/cameras...")

    # GPU clustering with cuML, or fallback to CPU multi-core
    if USE_CUML:
        print("-> Fitting HDBSCAN on entire dataset directly on GPU...")
        
        clusterer = HDBSCAN(
            min_cluster_size=400,           
            min_samples=40,                 
            metric='euclidean', 
            cluster_selection_epsilon=0.35, 
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(all_embeddings)
            
    else:
        print("-> Clustering (CPU Multi-core)...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=400,           
            min_samples=40,                 
            metric='euclidean', 
            cluster_selection_epsilon=0.35, 
            cluster_selection_method='eom',
            core_dist_n_jobs=-1
        )
        labels = clusterer.fit_predict(all_embeddings)
    
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    print(f"-> Identified {num_clusters} unique global students.")
    
    # Clear the old global structure
    if os.path.exists(GLOBAL_LABELS_DIR):
        shutil.rmtree(GLOBAL_LABELS_DIR)
    os.makedirs(GLOBAL_LABELS_DIR, exist_ok=True)

    print("-> Creating directories and transferring all images...")
    
    # Pre-create all folders to save I/O time during the loop
    for label in unique_labels:
        if label != -1:
            os.makedirs(os.path.join(GLOBAL_LABELS_DIR, f"Global_Student_{label}"), exist_ok=True)

    moved_count = 0

    # Fast loop: No distance calculations, just straight copying
    for img_path, label in zip(all_paths, labels):
        if label == -1: continue # Skip HDBSCAN noise
            
        dest_path = os.path.join(GLOBAL_LABELS_DIR, f"Global_Student_{label}", os.path.basename(img_path))
        
        if os.path.exists(img_path):
            shutil.copyfile(img_path, dest_path) 
            moved_count += 1

    print(f"\nGlobal processing complete!")
    print(f"Successfully transferred {moved_count} frames without reduction.")
    print(f"Grouped data saved in: {os.path.abspath(GLOBAL_LABELS_DIR)}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    process_videos()     
    global_clustering()
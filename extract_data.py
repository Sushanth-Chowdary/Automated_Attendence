import cv2
import os
import torch
from ultralytics import YOLO

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Directories
TARGET_DIR = './Extraceted Data'
VIDEOS_DIR = os.path.join(TARGET_DIR, 'VIDEOS')
FACES_DIR = os.path.join(TARGET_DIR, 'extracted_faces') 

# Create necessary directories (Safe creation)
for d in [VIDEOS_DIR, FACES_DIR]:
    os.makedirs(d, exist_ok=True)

# Processing parameters
FRAME_SKIP = 1 
YOLO_BATCH_SIZE = 32  
CONF_THRESHOLD = 0.70  

# Load Models
yolo_model = YOLO('yolov8n-face.pt', task='detect')


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
# VIDEO EXTRACTION 
# ==========================================
def process_videos():
    video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_filename in video_files:
        print(f"\n{'='*50}\nExtracting from {video_filename}\n{'='*50}")
        
        # Parse filename to get the base folder name (e.g., 'Dharanii' from 'Dharanii-2026-05-29...')
        video_id = os.path.splitext(video_filename)[0]
        
        if '-' in video_id:
            folder_name = video_id.split('-')[0]
        elif '_' in video_id:
            folder_name = video_id.split('_')[0]
        else:
            folder_name = video_id
            
        # Create the specific output directory for this person/prefix
        video_output_dir = os.path.join(FACES_DIR, folder_name)
        os.makedirs(video_output_dir, exist_ok=True)

        video_path = os.path.join(VIDEOS_DIR, video_filename)
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        batch_frames = []
        batch_frame_counts = []
        
        # --- FACE DETECTION (Batched YOLO Inference) ---
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

                            face_crop_bgr = crop_with_margin(curr_frame, box, margin_percentage=0.15)
                            if face_crop_bgr.size > 0:
                                # Resize exactly to 160x160 as requested
                                final_face_160 = cv2.resize(face_crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
                                
                                # Include video_id in filename to prevent overwriting when merging folders
                                filename = f"{video_id}_f{curr_frame_count}_i{face_idx}.jpg"
                                save_path = os.path.join(video_output_dir, filename)
                                
                                cv2.imwrite(save_path, final_face_160)
                                
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

                    face_crop_bgr = crop_with_margin(curr_frame, box, margin_percentage=0.15)
                    if face_crop_bgr.size > 0:
                        final_face_160 = cv2.resize(face_crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
                        
                        filename = f"{video_id}_f{curr_frame_count}_i{face_idx}.jpg"
                        save_path = os.path.join(video_output_dir, filename)
                        
                        cv2.imwrite(save_path, final_face_160)

        cap.release()
        print(f"Finished extracting faces from {video_filename}. Saved to {video_output_dir}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    process_videos()
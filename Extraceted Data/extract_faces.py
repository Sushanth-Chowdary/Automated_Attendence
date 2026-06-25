import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO

# ==========================================
# ALIGNMENT HELPER FUNCTIONS 
# ==========================================
def crop_standard(img, box):
    """Fallback standard crop with 15% margin."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img.shape[1], x2 + margin_x)
    y2 = min(img.shape[0], y2 + margin_y)
    
    return img[y1:y2, x1:x2]

def align_face(img, box, keypoints):
    """Rotates the face to align the eyes horizontally."""
    if keypoints is None or len(keypoints) < 2:
        return crop_standard(img, box)
        
    x1, y1, x2, y2 = map(int, box)
    left_eye, right_eye = keypoints[0], keypoints[1]
    
    # Ensure left_eye is physically on the left side of the image
    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye
        
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    
    if dx == 0:
        return crop_standard(img, box)
        
    # Calculate angle for affine transformation
    angle = np.degrees(np.arctan2(dy, dx))
    
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Take a larger 50% margin crop first to prevent losing corners during rotation
    margin_x, margin_y = int(w * 0.5), int(h * 0.5)
    X1 = max(0, cx - w - margin_x)
    Y1 = max(0, cy - h - margin_y)
    X2 = min(img.shape[1], cx + w + margin_x)
    Y2 = min(img.shape[0], cy + h + margin_y)
    
    large_crop = img[Y1:Y2, X1:X2]
    if large_crop.size == 0:
        return crop_standard(img, box)
        
    # Shift eye center to the local coordinate space of the large crop
    eye_center = ((left_eye[0] + right_eye[0]) / 2 - X1, (left_eye[1] + right_eye[1]) / 2 - Y1)
    
    # Rotate the large crop
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated_crop = cv2.warpAffine(large_crop, M, (large_crop.shape[1], large_crop.shape[0]), flags=cv2.INTER_CUBIC)
    
    # Transform the original box center to the new rotated space
    local_cx, local_cy = cx - X1, cy - Y1
    new_cx = M[0, 0] * local_cx + M[0, 1] * local_cy + M[0, 2]
    new_cy = M[1, 0] * local_cx + M[1, 1] * local_cy + M[1, 2]
    
    # Apply final 15% margin on the straightened face
    fin_margin_x, fin_margin_y = int(w * 0.15), int(h * 0.15)
    fx1 = int(new_cx - w/2 - fin_margin_x)
    fy1 = int(new_cy - h/2 - fin_margin_y)
    fx2 = int(new_cx + w/2 + fin_margin_x)
    fy2 = int(new_cy + h/2 + fin_margin_y)
    
    # Boundary checks
    fx1, fy1 = max(0, fx1), max(0, fy1)
    fx2, fy2 = min(rotated_crop.shape[1], fx2), min(rotated_crop.shape[0], fy2)
    
    final_crop = rotated_crop[fy1:fy2, fx1:fx2]
    if final_crop.size == 0:
        return crop_standard(img, box)
        
    return final_crop

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

os.makedirs(base_output_dir, exist_ok=True)

# ==========================================
# 2. PROCESSING LOOP
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
                keypoints = results[0].keypoints.xy.cpu().numpy() if hasattr(results[0], 'keypoints') and results[0].keypoints is not None else None
                
                if len(boxes) > 0:
                    # Find the most prominent face (largest area) to avoid capturing people in the background
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    largest_face_idx = np.argmax(areas)

                    box = boxes[largest_face_idx]
                    kpts = keypoints[largest_face_idx] if keypoints is not None else None
                    
                    # Align and crop the face
                    face_crop_bgr = align_face(frame, box, kpts)
                    
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

print("\nAll videos have been processed successfully!")
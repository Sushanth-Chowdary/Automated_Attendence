import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# CROP HELPER FUNCTION (Alignment Removed)
# ==========================================
def crop_standard(img, box):
    """Grabs the raw face with a 15% margin for context. Must match train/test scripts."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img.shape[1], x2 + margin_x)
    y2 = min(img.shape[0], y2 + margin_y)
    
    return img[y1:y2, x1:x2]

# ==========================================
# EXTRACTION LOGIC
# ==========================================
def extract_faces_for_dataset(raw_images_dir, output_labels_dir):
    print("Loading YOLO model...")
    # Use your base model or the TensorRT engine if preferred
    yolo_model = YOLO('yolov8n-face.pt', task='detect') 

    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    # Loop through each person's folder in the raw directory
    for person_name in os.listdir(raw_images_dir):
        person_raw_dir = os.path.join(raw_images_dir, person_name)
        
        if not os.path.isdir(person_raw_dir):
            continue
            
        person_out_dir = os.path.join(output_labels_dir, person_name)
        os.makedirs(person_out_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(person_raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            continue
            
        print(f"\nProcessing {len(image_files)} images for: {person_name}")
        saved_count = 0
        
        for idx, image_name in enumerate(image_files):
            image_path = os.path.join(person_raw_dir, image_name)
            img_cv2 = cv2.imread(image_path)
            
            if img_cv2 is None:
                continue
                
            # Run YOLO Detection
            results = yolo_model(img_cv2, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            if len(boxes) == 0:
                print(f"  -> No face detected in {image_name}, skipping.")
                continue
                
            # Grab the largest face (assumes the target person is the main subject)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_face_idx = np.argmax(areas)
            box = boxes[largest_face_idx]
            
            # Crop with standard 15% margin
            face_crop_bgr = crop_standard(img_cv2, box)
            
            # Strict resize to 160x160 to trigger the bypass logic in train_yolo.py later
            face_crop_160 = cv2.resize(face_crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
            
            # Quality Check: Skip heavily blurred images
            sharpness = cv2.Laplacian(face_crop_160, cv2.CV_64F).var()
            if sharpness < 4.0:
                print(f"  -> Skipping {image_name}: Too blurry (Sharpness: {sharpness:.1f})")
                continue
            
            # Save the valid 160x160 crop
            save_name = f"{person_name}_{idx:04d}.jpg"
            save_path = os.path.join(person_out_dir, save_name)
            cv2.imwrite(save_path, face_crop_160)
            saved_count += 1
            
        print(f"Successfully saved {saved_count} valid crops for {person_name}.")

if __name__ == "__main__":
    # Define your paths here
    # RAW_DIR should contain subfolders of uncropped pictures (e.g., RAW_DIR/Sushanth/pic1.jpg)
    RAW_DIR = './RAW_DATA' 
    TARGET_DIR = './LABELS'
    
    if os.path.exists(RAW_DIR):
        extract_faces_for_dataset(RAW_DIR, TARGET_DIR)
        print("\nAll extractions complete! You can now run train_yolo.py")
    else:
        print(f"Error: The raw images directory '{RAW_DIR}' does not exist. Please create it and add your images.")
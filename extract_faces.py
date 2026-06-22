import cv2
import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image

# 1. Initialize MTCNN
# Setting image_size=160 automatically forces the output crop to be 160x160 pixels.
# Setting keep_all=False ensures it only grabs the single most prominent face in each frame.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, image_size=160, margin=20, device=device)

# 2. Configuration
# These paths are based on your 'Extraceted Data' folder structure
videos_dir = './videos'              
base_output_dir = './extracted_faces' 

# Create the main output folder if it doesn't exist
os.makedirs(base_output_dir, exist_ok=True)

# 3. Process every video in the folder
print(f"Running on device: {device}")
print("Scanning the 'videos' folder...")

# Check if videos folder exists to avoid errors
if not os.path.exists(videos_dir):
    print(f"Error: Could not find folder '{videos_dir}'")
else:
    for video_filename in os.listdir(videos_dir):
        # Process .mkv and .mp4 files
        if not (video_filename.lower().endswith(('.mkv', '.mp4'))):
            continue

        video_path = os.path.join(videos_dir, video_filename)
        
        # Extract name: splits "Arfa-2026..." at the first '-' and takes "Arfa"
        person_name = video_filename.split('-')[0]
        
        # Create a specific folder for this person
        person_output_dir = os.path.join(base_output_dir, person_name)
        os.makedirs(person_output_dir, exist_ok=True)

        print(f"\n--- Processing video for: {person_name} ---")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_faces = 0
        frame_skip = 10  # Reduced skip slightly to get more data per video

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process one frame every 'frame_skip' frames
            if frame_count % frame_skip == 0:
                # Convert BGR (OpenCV) to RGB (PIL/Facenet)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                
                # Detect the face
                boxes, _ = mtcnn.detect(img_pil)
                
                if boxes is not None:
                    # Create the unique filename
                    filename = f"{person_name}_frame_{frame_count}.jpg"
                    save_path = os.path.join(person_output_dir, filename)
                    
                    # Extract and save. Passing 'boxes' directly fixes the TypeError.
                    try:
                        mtcnn.extract(img_pil, boxes, save_path=save_path)
                        saved_faces += 1
                    except Exception as e:
                        print(f"      Error saving frame {frame_count}: {e}")

        cap.release()
        print(f"Done! Saved {saved_faces} faces for {person_name}.")

print("\nAll videos have been processed successfully!")
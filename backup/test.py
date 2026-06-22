# 1. Imports
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import pickle
import pandas as pd
from datetime import datetime
import os
import shutil  # <-- Added for moving the file over the network
from tqdm import tqdm  # Progress bar

# 2. Setup Devices and Models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 3. Load the Trained Model and Mappings
model_save_path = './face_attendance_model.pkl'

with open(model_save_path, 'rb') as f:
    saved_data = pickle.load(f)

clf = saved_data['classifier']
target_names = saved_data['target_names']

# 4. Define Parameters & Setup Batch Processing
CONFIDENCE_THRESHOLD = 0.85
REQUIRED_FRAMES_TO_ATTEND = 100 
FRAME_SKIP = 2                 

input_dir = 'VIDEOS'
output_dir = 'ATTENDENCE RESULTS/MINE'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Put your specific videos in a list
target_videos = [
    '2026-04-27_10.02.44.mkv', 
    '2026-02-25_11.23.07.mkv', 
    '2026-03-05_11.02.28.mkv', 
    '2026-03-09_10.03.16.mkv',
    '2026-04-07_09.18.02.mkv',
    'video2.mkv',
    '2026-02-25_11.21.17.mkv',
    '2026-03-09_10.04.35.mkv',
    '2026-02-25_11.03.43.mkv',
    '2026-02-18_11.02.03.mkv',
    'video1_uajX8qg0.mp4',
    '2026-02-25_11.00.04.mkv',
    '2026-02-25_11.15.41.mkv',
    '2026-03-02_09.55.37.mkv'
]

video_files = []
for video in target_videos:
    if os.path.exists(os.path.join(input_dir, video)):
        video_files.append(video)
        print(f"Found: {video}")
    else:
        print(f"Warning: Could not find '{video}'")

print(f"\nTotal videos queued for processing: {len(video_files)}\n")
# -----------------------------------------------------

# 5. Iterate through each video
for video_filename in video_files:
    video_input_path = os.path.join(input_dir, video_filename)
    file_name_no_ext, extension = os.path.splitext(video_filename)
    
    # <-- UPDATED: Define local temp path and final network path as .mp4
    video_output_filename = f"{file_name_no_ext}_output.mp4"
    final_network_path = os.path.join(output_dir, video_output_filename)
    temp_local_path = f"./temp_{video_output_filename}" 
    
    attendance_csv_path = os.path.join(output_dir, f"{file_name_no_ext}_output.csv")

    # Initialize Video Capture
    cap = cv2.VideoCapture(video_input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_input_path}")
        continue
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

    # <-- UPDATED: Save to local temp path using mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_local_path, fourcc, fps, (frame_width, frame_height))

    recognition_counts = {}
    current_faces = [] 

    to_tensor = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    frame_count = 0
    
    print(f"--- Processing: {video_filename} ---")
    
    # Wrap the while loop with tqdm for the progress bar
    with tqdm(total=total_frames, desc="Progress", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break 

            # Logical update every 2nd frame
            if frame_count % FRAME_SKIP == 0:
                current_faces = [] 
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                boxes, probs = mtcnn.detect(pil_img)

                if boxes is not None:
                    for box, prob in zip(boxes, probs):
                        if prob > 0.90:
                            x1, y1, x2, y2 = [int(b) for b in box]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame_width, x2), min(frame_height, y2)

                            # Safety check to prevent crop crashes on frame edges
                            if x2 - x1 < 10 or y2 - y1 < 10: 
                                continue

                            face_crop = pil_img.crop((x1, y1, x2, y2))

                            try:
                                face_tensor = to_tensor(face_crop).unsqueeze(0).to(device)
                                face_tensor = (face_tensor - 0.5) * 2
                                embedding = resnet(face_tensor).detach().cpu().numpy()

                                probabilities = clf.predict_proba(embedding)[0]
                                max_prob_index = np.argmax(probabilities)
                                max_prob = probabilities[max_prob_index]

                                if max_prob > CONFIDENCE_THRESHOLD:
                                    name = target_names[max_prob_index]
                                    color = (0, 255, 0) 
                                    recognition_counts[name] = recognition_counts.get(name, 0) + 1
                                else:
                                    name = "Unknown"
                                    color = (0, 0, 255) 

                                label_text = f"{name} ({max_prob*100:.1f}%)"
                                current_faces.append({'coords': (x1, y1, x2, y2), 'label': label_text, 'color': color})

                            except Exception:
                                pass

            # Draw the boxes
            for face in current_faces:
                x1, y1, x2, y2 = face['coords']
                cv2.rectangle(frame, (x1, y1), (x2, y2), face['color'], 2)
                cv2.putText(frame, face['label'], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, face['color'], 2)

            out.write(frame)
            frame_count += 1
            pbar.update(1) 

    cap.release()
    out.release()
    
    # <-- UPDATED: Move the .mp4 file from local storage to network path
    print("\nMoving completed video to network drive...")
    try:
        shutil.move(temp_local_path, final_network_path)
        print(f"Output saved to: {final_network_path}")
    except Exception as e:
        print(f"Error moving file to network drive: {e}")
        print(f"Your video is safely stored locally at: {os.path.abspath(temp_local_path)}")

    # ==========================================
    # 6. UPDATED: Comprehensive Attendance Summary 
    # ==========================================
    attendance_records = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Iterate through ALL students the model was trained on
    for student in target_names:
        # Get the detection count (defaults to 0 if they weren't seen at all)
        count = recognition_counts.get(student, 0)
        
        # Determine Present/Absent based on threshold
        if count >= REQUIRED_FRAMES_TO_ATTEND:
            status = 'Present'
        else:
            status = 'Absent'
            
        attendance_records.append({
            'Name': student,
            'Time': current_time,
            'Status': status,
            'Detection Count': count
        })

    # Generate the CSV containing everyone
    pd.DataFrame(attendance_records).to_csv(attendance_csv_path, index=False)
    print(f"CSV Report generated: {attendance_csv_path}")
    
    print("-" * 50 + "\n")

print("Processing complete!")
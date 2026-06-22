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
import shutil  
import time
from tqdm import tqdm  
from collections import Counter
from imutils.video import FileVideoStream 
import faiss

# Tracker Import
from deep_sort_realtime.deepsort_tracker import DeepSort

# 2. Setup Devices and Models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# OPTIMIZATION 1: min_face_size and thresholds to save CPU
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=60, thresholds=[0.6, 0.7, 0.7])

# OPTIMIZATION 2: .half() for FP16 precision
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device).half()

tracker = DeepSort(max_age=10, n_init=3, embedder_gpu=True, half=True)

# 3. Load the FAISS Index and Mappings
faiss_index_path = './face_attendance_faiss.bin'
index = faiss.read_index(faiss_index_path)

with open('./face_attendance_meta.pkl', 'rb') as f:
    saved_data = pickle.load(f)

target_names = saved_data['target_names']
y_real = saved_data['y_real']

# 4. Define Parameters
CONFIDENCE_THRESHOLD = 0.80   
FRAME_SKIP = 3                
FRAMES_PER_VOTE = 5          
REQUIRED_VOTES = 8           

input_dir = 'VIDEOS'
output_dir = 'ATTENDENCE RESULTS/MINE'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

video_files = [v for v in target_videos if os.path.exists(os.path.join(input_dir, v))]
print(f"\nTotal videos queued for processing: {len(video_files)}\n")

to_tensor = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# 5. Iterate through each video
for video_filename in video_files:
    network_input_path = os.path.join(input_dir, video_filename)
    file_name_no_ext, _ = os.path.splitext(video_filename)
    
    local_input_path = f"/tmp/input_{video_filename}"
    video_output_filename = f"{file_name_no_ext}_output.mp4"
    temp_local_path = f"/tmp/temp_{video_output_filename}" 
    
    final_network_path = os.path.join(output_dir, video_output_filename)
    attendance_csv_path = os.path.join(output_dir, f"{file_name_no_ext}_Attendance.csv")
    debug_csv_path = os.path.join(output_dir, f"{file_name_no_ext}_DEBUG_Tracks.csv")

    print(f"\n--- Preparing: {video_filename} ---")
    try:
        shutil.copy(network_input_path, local_input_path)
    except Exception as e:
        print(f"Error copying input video locally: {e}")
        continue

    fvs = FileVideoStream(local_input_path).start()
    time.sleep(1.0) 
    
    frame_width = int(fvs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(fvs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(fvs.stream.get(cv2.CAP_PROP_FPS))
    total_frames = int(fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT)) 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_local_path, fourcc, fps, (frame_width, frame_height))

    # --- NEW VISIBILITY & IDENTITY TRACKERS ---
    track_visibility = {}  
    track_identities = {}  
    
    global_student_votes = {name: 0 for name in target_names}
    track_memory = {} # Used for the rolling buffer and debug reporting
    frame_count = 0
    
    print("Processing local video with async pipeline...")
    
    with tqdm(total=total_frames, desc="Progress", unit="frame") as pbar:
        while fvs.more():
            frame = fvs.read()
            if frame is None:
                break 

            is_keyframe = (frame_count % FRAME_SKIP == 0)

            # --- 1. TRACKING PHASE ---
            if is_keyframe:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                with torch.no_grad():
                    boxes, probs = mtcnn.detect(pil_img)
                    
                detections = []
                if boxes is not None:
                    for box, prob in zip(boxes, probs):
                        if prob > 0.90: # MTCNN threshold
                            x1, y1, x2, y2 = map(int, box)
                            w, h = x2 - x1, y2 - y1
                            detections.append(([x1, y1, w, h], prob, 'face'))

                tracks = tracker.update_tracks(detections, frame=frame)
            else:
                # Skipped Frames: Feed EMPTY detections to keep Kalman filter running
                tracks = tracker.update_tracks([], frame=frame)

            # --- 2. LOGIC & VISIBILITY PHASE ---
            if is_keyframe:
                batch_tensors = []
                batch_track_ids = []

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                        
                    t_id = track.track_id
                    
                    if t_id not in track_memory:
                        video_seconds = frame_count / fps if fps > 0 else 0
                        time_str = f"{int(video_seconds // 60)}m {int(video_seconds % 60)}s"
                        track_memory[t_id] = {'start_time': time_str, 'buffer': [], 'all_preds': []}

                    # If the track matched a REAL MTCNN detection on this exact frame
                    if track.time_since_update == 0:
                        track_visibility[t_id] = True
                        
                        x1, y1, x2, y2 = map(int, track.to_ltrb())
                        if (x2 - x1) < 50 or (y2 - y1) < 50: 
                            continue

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame_width, x2), min(frame_height, y2)

                        try:
                            face_crop = pil_img.crop((x1, y1, x2, y2))
                            face_tensor = to_tensor(face_crop).unsqueeze(0).to(device)
                            face_tensor = (face_tensor - 0.5) * 2
                            
                            batch_tensors.append(face_tensor)
                            batch_track_ids.append(t_id)
                        except Exception:
                            pass 
                    else:
                        # MTCNN ran, but this specific person was NOT detected. Instantly hide.
                        track_visibility[t_id] = False

                if len(batch_tensors) > 0:
                    combined_tensors = torch.cat(batch_tensors, dim=0).half() 
                    
                    with torch.no_grad():
                        embeddings = resnet(combined_tensors).detach().cpu().numpy().astype('float32')

                    faiss.normalize_L2(embeddings)
                    distances, indices = index.search(embeddings, k=1)

                    for i in range(len(batch_track_ids)):
                        t_id = batch_track_ids[i]
                        
                        similarity_score = distances[i][0]
                        matched_faiss_row = indices[i][0]
                        predicted_label_index = y_real[matched_faiss_row]
                        
                        if similarity_score > CONFIDENCE_THRESHOLD:
                            predicted_name = target_names[predicted_label_index]
                        else:
                            predicted_name = "Unknown"

                        track_memory[t_id]['buffer'].append(predicted_name)
                        track_memory[t_id]['all_preds'].append(predicted_name)

                        # Rolling voting mechanism
                        if len(track_memory[t_id]['buffer']) >= FRAMES_PER_VOTE:
                            vote_counts = Counter(track_memory[t_id]['buffer'])
                            winner = vote_counts.most_common(1)[0][0]
                            
                            if winner != "Unknown":
                                global_student_votes[winner] += 1
                                
                            track_memory[t_id]['buffer'] = [] # Reset buffer after voting
                            
                            # IDENTITY MERGING
                            if t_id not in track_identities or winner != "Unknown":
                                track_identities[t_id] = winner

            # --- 3. DRAWING PHASE ---
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                t_id = track.track_id
                
                # Strictly enforce our custom visibility rule
                if track_visibility.get(t_id, False) == True:
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    
                    # Fetch the name
                    display_name = track_identities.get(t_id, "Analyzing...")
                    
                    # Color coding
                    color = (0, 255, 0) if display_name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                    
                    # Draw the box and text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{t_id} {display_name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out.write(frame)
            frame_count += 1
            pbar.update(1) 

    # Clean up threaded reader
    fvs.stop()
    out.release()
    
    print("\nCopying completed video to network drive (safe mode)...")
    try:
        shutil.copy2(temp_local_path, final_network_path)
        if os.path.exists(final_network_path):
            os.remove(temp_local_path)
    except Exception as e:
        pass

    if os.path.exists(local_input_path):
        os.remove(local_input_path)

    # ==========================================
    # 7. GENERATE REPORTS
    # ==========================================
    attendance_records = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for student in target_names:
        votes = global_student_votes.get(student, 0)
        status = 'Present' if votes >= REQUIRED_VOTES else 'Absent'
            
        attendance_records.append({
            'Name': student,
            'Time': current_time,
            'Status': status,
            'Count': votes
        })

    pd.DataFrame(attendance_records).to_csv(attendance_csv_path, index=False)
    print(f"Attendance Report: {attendance_csv_path}")

    debug_records = []
    for t_id, data in track_memory.items():
        if len(data['all_preds']) == 0:
            continue
            
        overall_counts = Counter(data['all_preds'])
        true_identity = overall_counts.most_common(1)[0][0]
        
        debug_records.append({
            'Track ID': t_id,
            'First Seen (Video Time)': data['start_time'],
            'Total Frames Tracked': len(data['all_preds']),
            'Predicted Identity': true_identity,
            'Breakdown': dict(overall_counts)
        })
        
    pd.DataFrame(debug_records).to_csv(debug_csv_path, index=False)
    print(f"Debug Tracker Report: {debug_csv_path}")
    print("-" * 50 + "\n")

print("Processing complete!")
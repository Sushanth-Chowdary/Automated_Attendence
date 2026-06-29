# 1. Imports
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import pickle
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm  
from collections import Counter
import faiss
import gc
import time 
import subprocess 
import threading
import queue

# ==========================================
# THREADED VIDEO I/O HELPER 
# ==========================================
class ThreadedVideoReader:
    def __init__(self, path, queue_size=128):
        self.cap = cv2.VideoCapture(path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Some mkv files (esp. screen-captured / re-muxed ones) report 0 fps from
        # OpenCV's metadata reader. Guard against that so frame_count // fps below
        # never throws a ZeroDivisionError.
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            while not self.stopped:
                try:
                    self.q.put(frame, timeout=0.5)
                    break 
                except queue.Full:
                    continue
        self.cap.release()

    def read(self):
        try: return self.q.get(timeout=2.0)
        except queue.Empty: return None

    def more(self): return self.q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        while not self.q.empty():
            try: self.q.get_nowait()
            except queue.Empty: break

def crop_standard(img, box):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    x1 = max(0, x1 - margin_x); y1 = max(0, y1 - margin_y)
    x2 = min(img.shape[1], x2 + margin_x); y2 = min(img.shape[0], y2 + margin_y)
    return img[y1:y2, x1:x2]

def format_timestamp(frame_count, fps):
    """HH:MM:SS position in the video, used to record when a track ID first appeared."""
    total_seconds = frame_count // fps
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# 2. Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLO('yolov8n-face.pt', task='detect')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device).half()

faiss_index_path = './face_attendance_faiss.bin'
index = faiss.read_index(faiss_index_path)
index.nprobe = 20

with open('./face_attendance_meta.pkl', 'rb') as f:
    saved_data = pickle.load(f)
target_names, y_real = saved_data['target_names'], saved_data['y_real']

# 3. Parameters
CONFIDENCE_THRESHOLD = 0.74     
FRAME_SKIP = 5                # INCREASED for massive speed boost
FRAMES_PER_VOTE = 5          

input_dir = 'VIDEOS'
output_dir = os.path.abspath('ATTENDENCE RESULTS/MINE')
os.makedirs(output_dir, exist_ok=True)

# Writing the big .mp4 directly into the nested "ATTENDANCE RESULTS/MINE"
# folder on the network share was erroring out partway through. Writing it
# here first (same folder as this script, i.e. EE23B044) and moving the
# FINISHED file over afterward is much more reliable — only the small/quick
# CSVs go straight into output_dir, since those write in a fraction of a
# second and never showed the same problem.
video_staging_dir = os.path.abspath('.')

to_tensor = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])

# 4. Processing Loop
target_videos = ['2026-04-27_10.02.44.mkv', '2026-02-25_11.23.07.mkv', '2026-03-05_11.02.28.mkv', '2026-03-09_10.03.16.mkv', '2026-04-07_09.18.02.mkv', 'video2.mkv', '2026-02-25_11.21.17.mkv', '2026-03-09_10.04.35.mkv', '2026-02-25_11.03.43.mkv', '2026-02-18_11.02.03.mkv', 'video1_uajX8qg0.mp4', '2026-02-25_11.00.04.mkv', '2026-02-25_11.15.41.mkv', '2026-03-02_09.55.37.mkv']


def save_attendance_results(video_filename, archived_tracks, active_track_memory, target_names, output_dir):
    """Phase 4: collapse all tracked identities into a Present/Absent report.

    Pulled into its own function so it can be called BOTH on a normal finish
    AND from the interrupt/error handler below — so a Ctrl+C or a crash no
    longer means losing every track this video has already collected.
    """
    final_mem = {**archived_tracks, **active_track_memory}
    debug_data = []
    student_presence = {name: False for name in target_names}

    for t_id, data in final_mem.items():
        # 'frames_alive' = real elapsed frames the track was on screen, counted
        # every frame regardless of FRAME_SKIP (see the main loop). This is what
        # the 90-frame presence gate below checks against, so raising/lowering
        # FRAME_SKIP no longer silently changes how hard it is to "Pass" — it
        # only changes how often recognition runs, not how presence is measured.
        total_frames = data.get('frames_alive', 0)
        recognition_votes = len(data['all_preds'])
        counts = Counter(data['all_preds'])
        winner = counts.most_common(1)[0][0] if data['all_preds'] else "Unknown"

        # recognition_votes can legitimately be 0 (e.g. every crop for this
        # track failed the blur check) even if total_frames >= 90, so the
        # confidence-ratio check must be guarded — otherwise this is a
        # ZeroDivisionError waiting to happen.
        status = "Passed" if (total_frames >= 90 and recognition_votes > 0
                               and (counts.get(winner, 0) / recognition_votes) >= 0.6) else "Failed"
        if status == "Passed" and winner != "Unknown":
            student_presence[winner] = True

        debug_data.append({'Track ID': t_id, 'Start Time': data.get('start_time', ''),
                            'Total Frames': total_frames, 'Recognition Votes': recognition_votes,
                            'Predicted Identity': winner, 'Gate Status': status,
                            'Breakdown': dict(counts)})

    stem = os.path.splitext(video_filename)[0]
    # These now land in ATTENDANCE RESULTS/MINE (output_dir), named after the
    # source video itself (e.g. "2026-03-09_10.03.16_DEBUG_Tracks.csv"), instead
    # of whatever folder you happened to launch the script from with a "temp_"
    # prefix — previously output_dir was created (line ~98) but never used.
    pd.DataFrame(debug_data).to_csv(os.path.join(output_dir, f"{stem}_DEBUG_Tracks.csv"))
    pd.DataFrame([{'Name': s, 'Status': 'Present' if student_presence[s] else 'Absent'}
                  for s in target_names]).to_csv(os.path.join(output_dir, f"{stem}_output.csv"))
    print(f"  -> Saved attendance + debug CSVs for {video_filename} ({len(final_mem)} tracks) to {output_dir}")


interrupted = False

for video_filename in target_videos:
    if not os.path.exists(os.path.join(input_dir, video_filename)): continue

    print(f"\nProcessing: {video_filename}")
    video_stream = ThreadedVideoReader(os.path.join(input_dir, video_filename)).start()

    video_stem = os.path.splitext(video_filename)[0]
    staging_video_path = os.path.join(video_staging_dir, f"{video_stem}_output.mp4")
    out = cv2.VideoWriter(staging_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_stream.fps, (video_stream.frame_width, video_stream.frame_height))

    active_track_memory, archived_tracks, track_identities = {}, {}, {}
    frame_count = 0

    try:
        with tqdm(total=video_stream.total_frames, unit="frame") as pbar:
            while video_stream.more():
                frame = video_stream.read()
                if frame is None: break 

                # --- BYTE TRACK ---
                results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

                # Pull boxes/ids off the GPU ONCE per frame and reuse them for
                # recognition, drawing, AND cleanup below. Previously the drawing
                # loop called results[0].boxes.id.int().cpu().numpy() fresh on
                # EVERY detection (i.e. once per face, not once per frame), which
                # means re-doing a GPU->CPU sync N times instead of once whenever
                # N faces were on screen. At any real classroom headcount that adds
                # up fast and was very likely part of why this felt slow.
                has_detections = results[0].boxes.id is not None
                if has_detections:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.int().cpu().numpy()

                # Register every visible track and bump its real on-screen frame
                # count EVERY frame — not just on FRAME_SKIP checkpoints like
                # before. Previously a track was only created/counted on frames
                # where frame_count % FRAME_SKIP == 0, so raising FRAME_SKIP
                # silently shrank "Total Frames" for every track without the
                # 90-frame Pass/Fail gate changing to match it.
                if has_detections:
                    for t_id in ids:
                        if t_id not in active_track_memory:
                            active_track_memory[t_id] = {'start_time': format_timestamp(frame_count, video_stream.fps),
                                                          'frames_alive': 0, 'buffer': [], 'all_preds': []}
                        active_track_memory[t_id]['frames_alive'] += 1

                if frame_count % FRAME_SKIP == 0 and has_detections:
                    batch_tensors, batch_track_ids = [], []

                    for i, t_id in enumerate(ids):
                        crop = crop_standard(frame, boxes[i])
                        if crop.size > 0 and cv2.Laplacian(crop, cv2.CV_64F).var() > 4.0:
                            batch_tensors.append((to_tensor(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device) - 0.5) * 2)
                            batch_track_ids.append(t_id)

                    if batch_tensors:
                        with torch.no_grad():
                            embeddings = resnet(torch.cat(batch_tensors, dim=0).half()).cpu().numpy().astype('float32')
                        faiss.normalize_L2(embeddings)
                        sims, indices = index.search(embeddings, k=1)

                        for i, t_id in enumerate(batch_track_ids):
                            name = target_names[y_real[indices[i][0]]] if sims[i][0] > CONFIDENCE_THRESHOLD else "Unknown"
                            active_track_memory[t_id]['buffer'].append(name)
                            active_track_memory[t_id]['all_preds'].append(name)

                            if len(active_track_memory[t_id]['buffer']) >= FRAMES_PER_VOTE:
                                votes = [v for v in active_track_memory[t_id]['buffer'] if v != "Unknown"]
                                top_candidate, top_count = Counter(votes).most_common(1)[0] if votes else ("Unknown", 0)
                                winner = top_candidate if top_count >= 7 else "Unknown"
                                track_identities[t_id] = winner
                                active_track_memory[t_id]['buffer'] = []

                # Drawing
                if has_detections:
                    for i in range(len(ids)):
                        t_id = ids[i]
                        box = boxes[i]
                        name = track_identities.get(t_id, "Analyzing...")
                        color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                        cv2.putText(frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                out.write(frame)

                # Memory Cleanup
                alive_ids = set(ids) if has_detections else set()
                for t_id in list(active_track_memory.keys()):
                    if t_id not in alive_ids:
                        archived_tracks[t_id] = active_track_memory.pop(t_id)

                frame_count += 1
                pbar.update(1)

    except KeyboardInterrupt:
        # Ctrl+C: stop this video cleanly instead of letting the raw traceback
        # blow past out.release() / video_stream.stop() and skip the CSV save.
        print(f"\n[Interrupted] Stopping '{video_filename}' early. "
              f"Saving whatever attendance data was collected before exiting.")
        interrupted = True
    finally:
        # Runs on a clean finish, a real exception, OR a Ctrl+C — so the reader
        # thread always gets stopped, the video file always gets a valid footer
        # written via release(), and you always get a CSV instead of nothing.
        video_stream.stop()
        out.release()

        final_video_path = os.path.join(output_dir, f"{video_stem}_output.mp4")
        try:
            subprocess.run(['mv', staging_video_path, final_video_path], check=True)
            print(f"  -> Moved video to {final_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"  -> WARNING: video finished but couldn't be moved to {output_dir} ({e}). "
                  f"It's still saved locally at {staging_video_path} — move it manually.")

        save_attendance_results(video_filename, archived_tracks, active_track_memory, target_names, output_dir)

    if interrupted:
        break

if interrupted:
    print("\nBatch stopped early by user. Remaining videos in the list were not processed.")
else:
    print("\nAll videos processed.")
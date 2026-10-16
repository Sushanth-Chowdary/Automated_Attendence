import os
import shutil
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ==========================================
# 1. GENERATE custom_ocsort.yaml TRACKER CONFIG
# ==========================================
ocsort_yaml_content = """# Custom OC-SORT / Observation-Centric Tracker Config for PTZ Attendance
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 50
match_thresh: 0.8
fuse_score: True

# Camera Motion Compensation (CMC) for PTZ preset settling
gmc_method: sparseOptFlow

# ReID disabled (handled by AdaFace EMA voting loop)
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
"""

with open("custom_ocsort.yaml", "w") as f:
    f.write(ocsort_yaml_content)
print("[1/3] Created 'custom_ocsort.yaml' in project directory.")

# Also create ocsort.yaml alias so either filename in the AI agent's code works seamlessly
shutil.copyfile("custom_ocsort.yaml", "ocsort.yaml")

# ==========================================
# 2. DOWNLOAD ADAFACE IRESNET-50 WEIGHTS
# ==========================================
adaface_target = "adaface_ir50_webface4m.ckpt"
if not os.path.exists(adaface_target):
    print("[2/3] Downloading pretrained AdaFace IResNet-50 (WebFace4M) weights...")
    try:
        # Official HuggingFace mirror of mk-minchul/AdaFace pretrained weights
        downloaded_path = hf_hub_download(
            repo_id="minchul/cvlface_adaface_ir50_webface4m",
            filename="model.pt"
        )
        shutil.copyfile(downloaded_path, adaface_target)
        print(f"      Saved AdaFace weights to '{adaface_target}'")
    except Exception as e:
        print(f"      [!] Auto-download notice: {e}")
        print("      Please download 'adaface_ir50_webface4m.ckpt' from https://github.com/mk-minchul/AdaFace")
        print(f"      and place it in: {os.path.abspath(adaface_target)}")
else:
    print(f"[2/3] Found existing '{adaface_target}'.")

# ==========================================
# 3. VERIFY / EXPORT YOLO-FACE TO TENSORRT (.engine)
# ==========================================
yolo_pt = "yolo26n-face.pt"
if not os.path.exists(yolo_pt):
    if os.path.exists("yolov8n-face.pt"):
        print("[3/3] Creating 'yolo26n-face.pt' alias from 'yolov8n-face.pt'...")
        shutil.copyfile("yolov8n-face.pt", yolo_pt)
    else:
        print("[3/3] No local 'yolo26n-face.pt' found. Place your 5-point YOLO face weights in this directory.")

if os.path.exists(yolo_pt) and torch.cuda.is_available():
    engine_path = "yolo26n-face.engine"
    if not os.path.exists(engine_path):
        print(f"      Exporting '{yolo_pt}' to TensorRT FP16 '{engine_path}'...")
        try:
            model = YOLO(yolo_pt)
            model.export(format="engine", half=True, imgsz=640, device=0)
            print(f"      Successfully built TensorRT engine: '{engine_path}'")
        except Exception as e:
            print(f"      TensorRT export skipped (PyTorch .pt will be used automatically): {e}")
    else:
        print(f"[3/3] Found existing TensorRT engine: '{engine_path}'.")
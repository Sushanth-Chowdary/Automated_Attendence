import os
import requests
import subprocess
import time
import datetime
import threading
from requests.auth import HTTPBasicAuth

# ==========================================
# CONFIGURATION 
# ==========================================
USERNAME = "admin"
PASSWORD = "admin"
BASE_OUTPUT_DIR = "Recording"

CAM1_PRESETS = [1, 2, 3, 4, 5, 6, 7, 8]  # Left Corner (10.34.0.17)
CAM2_PRESETS = [1, 2, 3, 4, 5, 6]        # Center (10.34.0.16)

# 1 = 5 minutes
# 2 = 15 minutes
# 3 = 45 minutes
SELECTED_PROTOCOL = 1 

PROTOCOLS = {
    1: {"name": "5-Minute", "cam1": 32, "cam2": 45},
    2: {"name": "15-Minute", "cam1": 107, "cam2": 145},
    3: {"name": "45-Minute", "cam1": 332, "cam2": 445}
}

os.makedirs(os.path.join(BASE_OUTPUT_DIR, "Camera_1"), exist_ok=True)
os.makedirs(os.path.join(BASE_OUTPUT_DIR, "Camera_2"), exist_ok=True)

cam1_current_files = []
cam2_current_files = []

# ==========================================
# CAMERA CONTROLS
# ==========================================

def call_preset(ip, preset_number):
    url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&poscall&{preset_number}"
    try:
        response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=5)
        if response.status_code != 200:
            print(f"    [-] {ip}: Failed to trigger preset {preset_number}. Status {response.status_code}")
    except Exception as e:
        print(f"    [-] {ip}: Error connecting - {e}")

def record_stream(ip, folder, duration_seconds, preset_number, tracker_list):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{ip}:554/stream1"
    
    filename = f"Preset{preset_number:02d}_{timestamp}.mp4"
    output_filepath = os.path.join(BASE_OUTPUT_DIR, folder, filename)
    
    command = [
        'ffmpeg',
        '-y',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-t', str(duration_seconds),
        '-c', 'copy',
        output_filepath
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[+] {ip} finished Preset {preset_number} -> {filename}")
        
        tracker_list.append(filename)
        
    except Exception as e:
        print(f"[-] {ip} failed to record Preset {preset_number}: {e}")

# ==========================================
# THREADS
# ==========================================

def run_camera_1(duration):
    ip = "10.34.0.17"
    for preset in CAM1_PRESETS:
        print(f"[*] Camera 1 moving to Preset {preset} (Recording for {duration}s)...")
        call_preset(ip, preset)
        time.sleep(5) 
        record_stream(ip, "Camera_1", duration_seconds=duration, preset_number=preset, tracker_list=cam1_current_files)

def run_camera_2(duration):
    ip = "10.34.0.16"
    for preset in CAM2_PRESETS:
        print(f"[*] Camera 2 moving to Preset {preset} (Recording for {duration}s)...")
        call_preset(ip, preset)
        time.sleep(5) 
        record_stream(ip, "Camera_2", duration_seconds=duration, preset_number=preset, tracker_list=cam2_current_files)

# ==========================================
# VIDEO MERGER
# ==========================================

def merge_videos(folder_name, output_filename, file_list):
    """Stitches ONLY the specific files passed in the file_list."""
    if not file_list:
        print(f"[-] No new videos were recorded for {folder_name} to merge.")
        return

    print(f"\n[*] Merging {len(file_list)} new clips for {folder_name}...")
    folder_path = os.path.join(BASE_OUTPUT_DIR, folder_name)
    
    file_list.sort()
    
    list_file_path = os.path.join(folder_path, "concat_list.txt")
    with open(list_file_path, "w") as f:
        for video in file_list:
            f.write(f"file '{video}'\n")
            
    final_video_path = os.path.join(BASE_OUTPUT_DIR, output_filename)
    
    command = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file_path,
        '-c', 'copy',          
        final_video_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[+] Successfully merged! Final video saved as: {final_video_path}")
    except Exception as e:
        print(f"[-] Failed to merge videos for {folder_name}: {e}")
    finally:
        if os.path.exists(list_file_path):
            os.remove(list_file_path)

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    if SELECTED_PROTOCOL not in PROTOCOLS:
        print(f"[-] Invalid protocol selected: {SELECTED_PROTOCOL}. Please choose 1, 2, or 3.")
        exit(1)

    protocol_info = PROTOCOLS[SELECTED_PROTOCOL]
    print(f"Starting Independent {protocol_info['name']} Camera Sweeps (Protocol {SELECTED_PROTOCOL})...\n")
    
    thread_cam1 = threading.Thread(target=run_camera_1, args=(protocol_info['cam1'],))
    thread_cam2 = threading.Thread(target=run_camera_2, args=(protocol_info['cam2'],))
    
    thread_cam1.start()
    thread_cam2.start()
    
    thread_cam1.join()
    thread_cam2.join()
    
    print("\n[+] Sweeps complete. Beginning video merge process...")
    
    master_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    r
    merge_videos("Camera_1", f"Camera_1_MasterSweep_{master_timestamp}.mp4", cam1_current_files)
    merge_videos("Camera_2", f"Camera_2_MasterSweep_{master_timestamp}.mp4", cam2_current_files)
    
    print("\n[+] All automation and merging completely finished.")
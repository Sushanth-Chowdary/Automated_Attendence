import os
import glob
import time
import threading
import subprocess
import requests
from requests.auth import HTTPBasicAuth
import cv2

# =====================================================================
# 1. CAMERA & GRID CONFIGURATION (Matched to main_gui.py)
# =====================================================================
CAMERA_IP = "10.34.0.17"       # Or "10.34.0.16" for Camera 2
USERNAME = "admin"
PASSWORD = "admin"

RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/stream1"

OUTPUT_DIR = "sweep_images"
FINAL_PANO_TIF = "final_mega_panorama.tif"
FINAL_PANO_JPG = "final_mega_panorama.jpg"

# Optional: Call a starting preset (e.g., Preset 1) before sweeping, or set to None
START_PRESET = None

# Grid & Timing Parameters (Tune these to keep ~30-40% overlap between adjacent shots)
GRID_ROWS = 4               # Number of vertical tilt rows
GRID_COLS = 6               # Number of horizontal pan columns per row
PAN_STEP_TIME = 0.5         # Seconds to move left/right per step
TILT_STEP_TIME = 0.4        # Seconds to move down per row
ZOOM_MAX_TIME = 5.0         # Seconds to hold 'zoomin' to reach full optical zoom
SETTLE_TIME = 2.0           # Seconds to wait after stopping for camera shake & autofocus


# =====================================================================
# 2. CONTINUOUS RTSP STREAM READER (Prevents Stale/Duplicate Frames)
# =====================================================================
class LiveRTSPStream:
    """Continuously drains the RTSP buffer in a background thread so get_frame() always returns the latest live image."""
    def __init__(self, url: str):
        self.url = url
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        print(f"[*] Connecting to RTSP stream: {self.url} ...")
        # Wait up to 10 seconds for the first valid frame
        t0 = time.perf_counter()
        while self.get_frame() is None and (time.perf_counter() - t0) < 10.0:
            time.sleep(0.1)
        if self.get_frame() is None:
            raise RuntimeError(f"Failed to receive frames from {self.url}")
        print("[*] RTSP stream active and receiving live frames.")

    def _reader_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.url)
                if not self.cap.isOpened():
                    time.sleep(2)
                    continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.cap.release()
                self.cap = None
                continue
            with self.lock:
                self.latest_frame = frame

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()


# =====================================================================
# 3. EXACT PTZ HARDWARE CONTROL (From main_gui.py lines 714-731)
# =====================================================================
def manual_ptz(command: str, ip: str = CAMERA_IP) -> bool:
    """Sends exact PTZ command matching main_gui.py: /cgi-bin/ptzctrl.cgi?ptzcmd&{command}"""
    url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&{command}"
    try:
        resp = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=3)
        return resp.status_code == 200
    except Exception as e:
        print(f"[PTZ ERROR] Failed sending '{command}' to {url}: {e}")
        return False


def call_preset(preset_num: int, ip: str = CAMERA_IP) -> bool:
    """Calls a camera preset matching main_gui.py: /cgi-bin/ptzctrl.cgi?ptzcmd&poscall&{preset}"""
    url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&poscall&{preset_num}"
    try:
        print(f"  -> Moving to Preset {preset_num}...")
        resp = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=5)
        return resp.status_code == 200
    except Exception as e:
        print(f"[PRESET ERROR] Failed calling preset {preset_num}: {e}")
        return False


def move_ptz_timed(start_cmd: str, stop_cmd: str, duration: float):
    """Starts movement, waits for `duration` seconds, and sends the exact stop command."""
    print(f"  -> PTZ '{start_cmd}' for {duration:.2f}s -> '{stop_cmd}'")
    manual_ptz(start_cmd)
    time.sleep(duration)
    manual_ptz(stop_cmd)


# =====================================================================
# 4. AUTOMATED MATRIX SWEEP & CAPTURE
# =====================================================================
def run_matrix_sweep():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for old_file in glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")):
        os.remove(old_file)

    # Start continuous RTSP reader
    stream = LiveRTSPStream(RTSP_URL)
    stream.start()

    captured_files = []
    try:
        print("\n=== STEP 1: INITIALIZING STARTING POSITION & MAX ZOOM ===")
        if START_PRESET is not None:
            call_preset(START_PRESET)
            time.sleep(5.0)  # Wait for preset mechanical movement
        else:
            # Sweep to Top-Left corner before starting grid
            print("Moving to top-left starting corner...")
            move_ptz_timed("left", "ptzstop", duration=6.0)
            move_ptz_timed("up", "ptzstop", duration=4.0)

        # Zoom in to capture fine optical detail
        print("Zooming in to maximum optical level...")
        move_ptz_timed("zoomin", "zoomstop", duration=ZOOM_MAX_TIME)
        time.sleep(SETTLE_TIME)

        print(f"\n=== STEP 2: RUNNING {GRID_ROWS}x{GRID_COLS} SNAKE GRID SWEEP ===")
        for row in range(GRID_ROWS):
            # Even rows sweep Left -> Right ("right"); Odd rows sweep Right -> Left ("left")
            direction = "right" if (row % 2 == 0) else "left"

            for col in range(GRID_COLS):
                actual_col = col if (row % 2 == 0) else (GRID_COLS - 1 - col)
                filename = os.path.join(OUTPUT_DIR, f"tile_r{row:02d}_c{actual_col:02d}.jpg")

                print(f"\n[Row {row + 1}/{GRID_ROWS}, Col {actual_col + 1}/{GRID_COLS}] Settling for {SETTLE_TIME}s...")
                time.sleep(SETTLE_TIME)

                frame = stream.get_frame()
                if frame is not None:
                    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    print(f"  [SAVED] {filename} ({frame.shape[1]}x{frame.shape[0]})")
                    captured_files.append(filename)
                else:
                    print(f"  [WARNING] No frame available at Row {row + 1}, Col {actual_col + 1}")

                # Pan horizontally to the next column if not at the end of the row
                if col < GRID_COLS - 1:
                    move_ptz_timed(direction, "ptzstop", duration=PAN_STEP_TIME)

            # Tilt down to the next row if not on the final row
            if row < GRID_ROWS - 1:
                print(f"\n[Row {row + 1} Done] Tilting down to Row {row + 2}...")
                move_ptz_timed("down", "ptzstop", duration=TILT_STEP_TIME)

        print("\n=== SWEEP COMPLETE! Resetting zoom... ===")
        move_ptz_timed("zoomout", "zoomstop", duration=ZOOM_MAX_TIME)

    finally:
        # Guarantee the camera stops moving and RTSP closes cleanly even if interrupted
        manual_ptz("ptzstop")
        manual_ptz("zoomstop")
        stream.stop()

    return sorted(captured_files)


# =====================================================================
# 5. GIGAPIXEL STITCHING ENGINE (HUGIN CLI + OPENCV FALLBACK)
# =====================================================================
def stitch_with_hugin(image_files):
    """Aligns, optimizes, and blends captured tiles using Hugin CLI tools."""
    print("\n=== STEP 3: RUNNING HUGIN GIGAPIXEL STITCHING PIPELINE ===")
    pto_file = "project.pto"

    try:
        print("1/5 Generating Hugin project file (pto_gen)...")
        subprocess.run(["pto_gen", "-o", pto_file] + image_files, check=True)

        print("2/5 Finding multi-row control points (cpfind)...")
        subprocess.run(["cpfind", "-o", pto_file, "--multirow", pto_file], check=True)

        print("3/5 Optimizing positions and lens distortion (autooptimiser)...")
        subprocess.run(["autooptimiser", "-a", "-m", "-l", "-s", "-o", pto_file, pto_file], check=True)

        print("4/5 Remapping images (nona)...")
        subprocess.run(["nona", "-o", "output_pano", "-m", "TIFF_m", pto_file], check=True)

        remapped_tiffs = sorted(glob.glob("output_pano*.tif"))
        if not remapped_tiffs:
            raise RuntimeError("nona did not produce any output_pano*.tif files.")

        print(f"5/5 Blending {len(remapped_tiffs)} tiles into final mega-panorama (enblend)...")
        subprocess.run(["enblend", "-o", FINAL_PANO_TIF] + remapped_tiffs, check=True)

        pano_img = cv2.imread(FINAL_PANO_TIF)
        if pano_img is not None:
            cv2.imwrite(FINAL_PANO_JPG, pano_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"\n[SUCCESS] Hugin Mega-Panorama saved to '{FINAL_PANO_TIF}' and '{FINAL_PANO_JPG}'!")
        return True

    except FileNotFoundError:
        print("[WARNING] Hugin CLI tools (pto_gen, cpfind, nona, enblend) are not installed.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Hugin CLI step failed: {e}")
        return False


def stitch_with_opencv(image_files):
    """Fallback panorama stitcher using OpenCV if Hugin CLI is not installed."""
    print("\n=== RUNNING OPENCV BUILT-IN STITCHER (FALLBACK) ===")
    images = [cv2.imread(f) for f in image_files if cv2.imread(f) is not None]

    if len(images) < 2:
        print("[ERROR] Need at least 2 valid images to stitch.")
        return

    # Use SCANS mode or PANORAMA mode
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        cv2.imwrite(FINAL_PANO_JPG, pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(FINAL_PANO_TIF, pano)
        print(f"\n[SUCCESS] OpenCV Mega-Panorama saved to '{FINAL_PANO_JPG}' and '{FINAL_PANO_TIF}'!")
    else:
        print(f"\n[ERROR] OpenCV stitching failed with status code {status}.")
        print("Tip: Reduce PAN_STEP_TIME and TILT_STEP_TIME so adjacent frames overlap by ~35%.")


if __name__ == "__main__":
    captured_tiles = run_matrix_sweep()
    if len(captured_tiles) >= 2:
        if not stitch_with_hugin(captured_tiles):
            stitch_with_opencv(captured_tiles)
    else:
        print("[ABORT] Not enough images captured to stitch.")
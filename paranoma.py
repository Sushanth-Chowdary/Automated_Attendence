import os
import time
import glob
import subprocess
import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
import cv2

# =====================================================================
# 1. CAMERA & GRID CONFIGURATION
# =====================================================================
CAMERA_IP = "10.34.0.17"
USERNAME = "admin"
PASSWORD = "admin"

# Common RTSP stream path (Update if your camera model uses a different path, e.g., /Streaming/Channels/101 or /h264Preview_01_main)
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/stream1"

# Common CGI URL template for cameras using "up/ptzstop" command pairs (e.g., HiSilicon / CamHi / Foscam / Amcrest)
# Adjust this endpoint to match the URL used in your button GUI script
CGI_URL_TEMPLATE = f"http://{CAMERA_IP}/cgi-bin/hi3510/ptzctrl.cgi?-step=0&-act={{cmd}}&-speed=20"

OUTPUT_DIR = "sweep_images"
FINAL_PANO_TIF = "final_mega_panorama.tif"
FINAL_PANO_JPG = "final_mega_panorama.jpg"

# Grid & Timing Parameters (Calibrate these for your room and lens FOV)
GRID_ROWS = 4               # Number of vertical tilt rows
GRID_COLS = 6               # Number of horizontal pan columns per row
PAN_STEP_TIME = 0.6         # Seconds to move left/right per step (keep ~30% image overlap!)
TILT_STEP_TIME = 0.5        # Seconds to move down per row (keep ~30% image overlap!)
ZOOM_MAX_TIME = 6.0         # Seconds to hold 'zoomin' to reach full optical zoom
SETTLE_TIME = 2.0           # Seconds to wait after stopping for vibration & autofocus to settle


# =====================================================================
# 2. HARDWARE CONTROL FUNCTIONS
# =====================================================================
def send_ptz_command(cmd: str):
    """Sends a PTZ start/stop command to the camera via HTTP."""
    url = CGI_URL_TEMPLATE.format(cmd=cmd)
    try:
        # Try Basic Auth first; fall back to Digest Auth if needed
        response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=5)
        if response.status_code == 401:
            response = requests.get(url, auth=HTTPDigestAuth(USERNAME, PASSWORD), timeout=5)
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"[ERROR] Failed to send PTZ command '{cmd}': {e}")
        return False


def move_ptz_timed(start_cmd: str, stop_cmd: str, duration: float):
    """Starts a PTZ action, waits for `duration` seconds, and sends the stop command."""
    print(f"  -> Executing '{start_cmd}' for {duration:.2f}s...")
    send_ptz_command(start_cmd)
    time.sleep(duration)
    send_ptz_command(stop_cmd)


def capture_fresh_frame(rtsp_url: str, save_path: str) -> bool:
    """Connects to the RTSP stream, flushes buffered frames, and saves a sharp high-res image."""
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[ERROR] Could not open RTSP stream: {rtsp_url}")
        return False

    # Flush internal OpenCV buffer (grab 5 frames) so we don't capture a blurry mid-movement frame
    for _ in range(5):
        cap.grab()

    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"  [SAVED] {save_path} ({frame.shape[1]}x{frame.shape[0]})")
        return True
    else:
        print(f"  [ERROR] Failed to decode frame for {save_path}")
        return False


# =====================================================================
# 3. AUTOMATED MATRIX SWEEP (BOUSTROPHEDON / SNAKE PATTERN)
# =====================================================================
def run_matrix_sweep():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear old sweep images
    for old_file in glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")):
        os.remove(old_file)

    print("=== STEP 1: HOMING TO TOP-LEFT & INITIALIZING MAX ZOOM ===")
    # Optional: Move to top-left corner before starting
    move_ptz_timed("left", "ptzstop", duration=8.0)
    move_ptz_timed("up", "ptzstop", duration=5.0)

    # Zoom in completely to maximize optical detail
    move_ptz_timed("zoomin", "zoomstop", duration=ZOOM_MAX_TIME)
    time.sleep(SETTLE_TIME)

    print(f"\n=== STEP 2: STARTING {GRID_ROWS}x{GRID_COLS} GRID SWEEP ===")
    captured_files = []

    for row in range(GRID_ROWS):
        # Even rows sweep Left -> Right; Odd rows sweep Right -> Left
        direction = "right" if (row % 2 == 0) else "left"

        for col in range(GRID_COLS):
            # Calculate true column index in the grid
            actual_col = col if (row % 2 == 0) else (GRID_COLS - 1 - col)
            filename = os.path.join(OUTPUT_DIR, f"tile_r{row:02d}_c{actual_col:02d}.jpg")

            print(f"\n[Row {row + 1}/{GRID_ROWS}, Col {actual_col + 1}/{GRID_COLS}] Waiting {SETTLE_TIME}s to stabilize & focus...")
            time.sleep(SETTLE_TIME)

            # Capture high-res frame
            if capture_fresh_frame(RTSP_URL, filename):
                captured_files.append(filename)

            # Move horizontally to the next column (unless we are at the end of the row)
            if col < GRID_COLS - 1:
                move_ptz_timed(direction, "ptzstop", duration=PAN_STEP_TIME)

        # At the end of a row, step down to the next row (unless we finished the last row)
        if row < GRID_ROWS - 1:
            print(f"\n[Row {row + 1} Complete] Stepping down to Row {row + 2}...")
            move_ptz_timed("down", "ptzstop", duration=TILT_STEP_TIME)

    # Reset zoom after sweep completes
    print("\n=== SWEEP COMPLETE! Resetting zoom... ===")
    move_ptz_timed("zoomout", "zoomstop", duration=ZOOM_MAX_TIME)
    return sorted(captured_files)


# =====================================================================
# 4. GIGAPIXEL STITCHING ENGINE (HUGIN CLI + OPENCV FALLBACK)
# =====================================================================
def stitch_with_hugin(image_files):
    """Runs the Hugin CLI toolchain to align, optimize, and blend the panorama."""
    print("\n=== STEP 3: RUNNING HUGIN GIGAPIXEL STITCHING PIPELINE ===")
    pto_file = "project.pto"

    try:
        # Note: Pass image_files list directly because subprocess.run(..., shell=False) does not expand '*.jpg' wildcards!
        print("1/5 Generating Hugin project file (pto_gen)...")
        subprocess.run(["pto_gen", "-o", pto_file] + image_files, check=True)

        print("2/5 Finding multi-row control points (cpfind)...")
        subprocess.run(["cpfind", "-o", pto_file, "--multirow", pto_file], check=True)

        print("3/5 Optimizing positions, lens distortion, and exposure (autooptimiser)...")
        subprocess.run(["autooptimiser", "-a", "-m", "-l", "-s", "-o", pto_file, pto_file], check=True)

        print("4/5 Remapping tiles (nona)...")
        subprocess.run(["nona", "-o", "output_pano", "-m", "TIFF_m", pto_file], check=True)

        remapped_tiffs = sorted(glob.glob("output_pano*.tif"))
        if not remapped_tiffs:
            raise RuntimeError("nona did not produce any output_pano*.tif files.")

        print(f"5/5 Blending {len(remapped_tiffs)} remapped tiles into final mega-image (enblend)...")
        subprocess.run(["enblend", "-o", FINAL_PANO_TIF] + remapped_tiffs, check=True)

        # Also save a compressed JPG version for easy viewing
        pano_img = cv2.imread(FINAL_PANO_TIF)
        if pano_img is not None:
            cv2.imwrite(FINAL_PANO_JPG, pano_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"\n[SUCCESS] Mega-image saved to '{FINAL_PANO_TIF}' and '{FINAL_PANO_JPG}'!")
        return True

    except FileNotFoundError:
        print("[WARNING] Hugin CLI tools (pto_gen/cpfind/nona/enblend) not found in system PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Hugin pipeline failed during execution: {e}")
        return False


def stitch_with_opencv_fallback(image_files):
    """Fallback stitcher using OpenCV's built-in Stitcher class if Hugin is not installed."""
    print("\n=== RUNNING OPENCV BUILT-IN STITCHER (FALLBACK) ===")
    images = [cv2.imread(f) for f in image_files if cv2.imread(f) is not None]

    if len(images) < 2:
        print("[ERROR] Need at least 2 valid images to stitch.")
        return

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        cv2.imwrite(FINAL_PANO_JPG, pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(FINAL_PANO_TIF, pano)
        print(f"[SUCCESS] OpenCV stitched mega-image saved to '{FINAL_PANO_JPG}' and '{FINAL_PANO_TIF}'!")
    else:
        print(f"[ERROR] OpenCV stitching failed with status code {status}. (Ensure at least 30% overlap between steps).")


if __name__ == "__main__":
    tiles = run_matrix_sweep()
    if len(tiles) >= 2:
        if not stitch_with_hugin(tiles):
            stitch_with_opencv_fallback(tiles)
    else:
        print("[ABORT] Not enough frames captured to generate a panorama.")
"""
File: camera_calibrator.py
Project: Coin Detection and Value Decoding System

Author: Suraj Karki
Affiliation: Master's Programme – AI and Automation, Högskolan Väst
Date: 2026-01-12

Description:
    Calibrates the camera to determine metric scale (pixels per mm). 
    Determines the pixel-to-mm ratio using a known square marker of fixed size.
"""
import cv2
import numpy as np
import json
from camera import Camera
from utility import utility

# ------------------------
# Parameters
# ------------------------
KNOWN_SQUARE_MM = 5
MIN_CONTOUR_AREA = 1000

DISPLAY_WINDOW_NAME = "Live Camera Calibration"
DISPLAY_SCALE = 1.3

STABLE_FRAMES = 30
STABILITY_EPS_PXMM = 0.2

CALIB_EXPOSURE = 1500
CALIB_GAIN = 12

COLOR_RANGES = utility.COLOR_RANGES


class CameraCalibration:
    def __init__(self):
        self.pixels_per_mm_x: float | None = None
        self.pixels_per_mm_y: float | None = None
        self.known_square_mm: float | None = None

    # ========================
    # Live calibration
    # ========================
    def run_live_calibration(self, save_path="sensorTechnologyAndImageAnalysis/metric_camera.json") -> bool:
        camera = Camera(
            exposure=CALIB_EXPOSURE,
            gain=CALIB_GAIN,
            target_fps=15
        )

        print(f"[INFO] Native resolution: {camera.get_resolution()}")
        print("[INFO] Waiting for stable calibration...")

        px_mm_x: list[float] = []
        px_mm_y: list[float] = []

        cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            frame = camera.read()
            if frame is None:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            detected = False

            for ranges in COLOR_RANGES.values():
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)

                for lo, hi in ranges:
                    mask |= cv2.inRange(hsv, lo, hi)

                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
                )
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
                )

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = [
                    c for c in contours
                    if cv2.contourArea(c) >= MIN_CONTOUR_AREA
                ]

                if not contours:
                    continue

                square = min(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(square)

                px_mm_x.append(w / KNOWN_SQUARE_MM)
                px_mm_y.append(h / KNOWN_SQUARE_MM)

                detected = True
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                )
                break

            # ------------------------
            # Stability check
            # ------------------------
            stable = False
            mean_x = mean_y = std_x = std_y = None

            if len(px_mm_x) >= STABLE_FRAMES:
                recent_x = px_mm_x[-STABLE_FRAMES:]
                recent_y = px_mm_y[-STABLE_FRAMES:]

                std_x = np.std(recent_x)
                std_y = np.std(recent_y)

                mean_x = float(np.mean(recent_x))
                mean_y = float(np.mean(recent_y))

                if std_x < STABILITY_EPS_PXMM and std_y < STABILITY_EPS_PXMM:
                    stable = True

            # ------------------------
            # Overlay
            # ------------------------
            if mean_x is not None:
                colour = (0, 255, 0) if stable else (0, 165, 255)

                cv2.putText(
                    frame,
                    f"Pixels/mm  X:{mean_x:.2f}  Y:{mean_y:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    colour,
                    2
                )

                cv2.putText(
                    frame,
                    f"STD  X:{std_x:.3f}  Y:{std_y:.3f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colour,
                    2
                )

            display = cv2.resize(
                frame, (0, 0),
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE
            )
            cv2.imshow(DISPLAY_WINDOW_NAME, display)

            if stable:
                print("[INFO] Calibration stable — saving")
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Calibration aborted by user")
                break

        camera.release()
        cv2.destroyAllWindows()

        if len(px_mm_x) < STABLE_FRAMES:
            print("[ERROR] Calibration failed: insufficient stable frames")
            return False

        self.pixels_per_mm_x = float(np.mean(px_mm_x[-STABLE_FRAMES:]))
        self.pixels_per_mm_y = float(np.mean(px_mm_y[-STABLE_FRAMES:]))
        self.known_square_mm = KNOWN_SQUARE_MM

        self.save(save_path)
        print(f" Calibration saved to {save_path}")
        return True

    # ========================
    # Save / Load Utility 
    # ========================
    def save(self, path: str) -> None:
        if self.pixels_per_mm_x is None or self.pixels_per_mm_y is None:
            raise RuntimeError("Calibration not available")
        
        utility.PIXELS_PER_MM_X = self.pixels_per_mm_x
        utility.PIXELS_PER_MM_Y = self.pixels_per_mm_y

        data = {
            "pixels_per_mm_x": self.pixels_per_mm_x,
            "pixels_per_mm_y": self.pixels_per_mm_y
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)

        self.pixels_per_mm_x = float(data["pixels_per_mm_x"])
        self.pixels_per_mm_y = float(data["pixels_per_mm_y"])

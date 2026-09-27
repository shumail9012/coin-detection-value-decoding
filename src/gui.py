import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import time
import sys
from collections import Counter
import numpy as np
import cv2

from camera import Camera
from camera_calibrator import CameraCalibration
from object_detector import ObjectAndSymbolDetector
from object_validator import ObjectValidator
from value_decoder import ValueDecoder
from camera_producer import CameraProducer
from frame_procesor import FrameProcessor


class ScannerGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Galactic Credit Scanner")
        self.window.geometry("1200x700")
        self.window.minsize(800, 500)
        self.target_fps = 60
        self.detector = ObjectAndSymbolDetector()
        self.validator = ObjectValidator()
        self.decoder = ValueDecoder()
        self.camera = Camera(target_fps=self.target_fps)
        self.camera_producer = CameraProducer(self.camera)

        self.processor = FrameProcessor(
            self.camera_producer,
            self.detector,
            self.validator,
            self.decoder
)

        # ------------------------
        # Calibration
        # ------------------------
        self.calibrator = CameraCalibration()
        self.calibrator.load("metric_camera.json")

        # ------------------------
        # Camera
        # ------------------------
        

        # ------------------------
        # Processing stack
        # ------------------------
        self.detector = ObjectAndSymbolDetector()
        self.validator = ObjectValidator()
        self.decoder = ValueDecoder()

        # ------------------------
        # FPS tracking
        # ------------------------
        self.last_frame_time = time.time()
        self.current_fps = 0

        # ------------------------
        # Layout
        # ------------------------
        main_frame = ttk.Frame(window)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)

        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        control_frame = ttk.Frame(main_frame, width=220)
        control_frame.grid(row=0, column=1, sticky="ns", padx=10)
        control_frame.grid_propagate(False)

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=0)

        self.status_label = ttk.Label(
            control_frame,
            text="LIVE",
            font=("Helvetica", 16),
            foreground="green"
        )
        self.status_label.pack(pady=20)

        self.capture_btn = ttk.Button(
            control_frame, text="Take Picture", command=self.capture_image
        )
        self.capture_btn.pack(pady=10, fill=tk.X)

        self.record_btn = ttk.Button(
            control_frame, text="Start Recording", command=self.toggle_recording
        )
        self.record_btn.pack(pady=10, fill=tk.X)

        self.calibrate_btn = ttk.Button(
            control_frame, text="Calibrate Camera", command=self.calibrate_camera
        )
        self.calibrate_btn.pack(pady=20, fill=tk.X)

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start live loop
        self.update_frame()

    # -------------------------------------------------
    # Core live update
    # -------------------------------------------------
    def update_frame(self):
        frame = self.processor.get_latest()
        if frame is not None:
            lbl_w = self.video_label.winfo_width()
            lbl_h = self.video_label.winfo_height()

            h, w, _ = frame.shape
            scale = min(lbl_w / w, lbl_h / h) if lbl_w > 1 else 1

            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR
            )

            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

        self.window.after(1, self.update_frame)

    # -------------------------------------------------
    # Frame processing (reused from offline GUI)
    # -------------------------------------------------
    def process_frame(self, frame_bgr):
        result = self.detector.detect(frame_bgr, visualise=False)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()

        for obj in result["objects"]:
            self.validator.validate_object(obj)

            symbols = obj.get("symbols", [])
            if symbols:
                color = Counter([s["color"] for s in symbols]).most_common(1)[0][0]
                value = self.decoder.decode_object(
                    symbols, color, obj["bbox_rect"], img_debug=frame_bgr
                )
            else:
                color = "-"
                value = "-"

            obj_pts = np.array(obj["inner_area"], dtype=np.int32)
            draw.polygon(
                [tuple(p[0]) for p in obj_pts],
                outline=(0, 255, 0) if obj["real_or_fake"] == "Real" else (255, 0, 0),
                width=3
            )

            cx, cy = obj["centroid"]

            for s in symbols:
                s_pts = np.array(s["contour"], dtype=np.int32)
                draw.polygon(
                    [tuple(p[0]) for p in s_pts],
                    outline={
                        "red": (0, 255, 0),
                        "blue": (255, 0, 255),
                        "yellow": (255, 0, 0)
                    }.get(s["color"], (255, 255, 255)),
                    width=2
                )

            text = f"{obj['real_or_fake']} | {value} | {color}"
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            x0 = int(cx - tw / 2)
            y0 = int(cy - obj_pts[:, 0, 1].min() - th - 10)

            draw.rectangle(
                [x0 - 4, y0 - 4, x0 + tw + 4, y0 + th + 4],
                fill=(0, 0, 0)
            )
            draw.text(
                (x0, y0),
                text,
                fill=(0, 255, 0) if obj["real_or_fake"] == "Real" else (255, 0, 0),
                font=font
            )

        return img_pil

    # -------------------------------------------------
    # Controls
    # -------------------------------------------------
    def capture_image(self):
        self.camera.capture_image()

    def toggle_recording(self):
        if self.camera.recording:
            self.camera.stop_recording()
            self.record_btn.config(text="⏺ Start Recording")
        else:
            if self.camera.start_recording():
                self.record_btn.config(text="⏹ Stop Recording")

    def calibrate_camera(self):
        if self.camera.recording:
            self.status_label.config(text="STOP RECORDING", foreground="red")
            return

        self.camera.release()
        self.status_label.config(text="CALIBRATING...", foreground="orange")
        self.window.update_idletasks()

        success = CameraCalibration().run_live_calibration()
        self.camera = Camera(target_fps=self.target_fps)

        if success:
            self.calibrator.load("metric_camera.json")
            self.status_label.config(text="CALIBRATION OK", foreground="green")
        else:
            self.status_label.config(text="CALIBRATION FAILED", foreground="red")

    def on_close(self):
        self.camera_producer.stop()
        self.processor.stop()
        self.camera.release()
        self.window.destroy()
        sys.exit()


# =========================
# Run
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = ScannerGUI(root)
    root.mainloop()

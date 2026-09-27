"""
File: static_gui.py
Project: Coin Detection and Value Decoding System

Author: Suraj Karki
Affiliation: Master's Programme – AI and Automation, Högskolan Väst
Date: 2026-01-12

Description:
    Main script for the Coin Detection and Value Decoding System.
    GUI for Galactic Credit Scanner integrating camera calibration, object detection, validation, and value decoding.
    Starts live camera feed or processes images/videos from disk.
"""
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
from collections import Counter
import numpy as np
import os
import threading
import time
from queue import Queue

from camera_calibrator import CameraCalibration
from value_decoder import ValueDecoder
from object_detector import ObjectAndSymbolDetector
from object_validator import ObjectValidator
from camera import Camera

# =========================
# Scanner GUI
# =========================
class ScannerGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Galactic Credit Scanner")
        self.window.geometry("1200x700")
        self.window.minsize(800, 500)

        # Calibration
        self.calibrator = CameraCalibration()
        self.calibrator.load("metric_camera.json")

        # Layout-
        main_frame = ttk.Frame(window)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)

        # Display panel
        self.display_label = ttk.Label(main_frame)
        self.display_label.grid(row=0, column=0, sticky="nsew")

        # Control panel
        control_frame = ttk.Frame(main_frame, width=220)
        control_frame.grid(row=0, column=1, sticky="ns", padx=10)
        control_frame.grid_propagate(False)

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=0)

        # Status label
        self.status_label = ttk.Label(control_frame, text="No Media Loaded", font=("Helvetica", 10), foreground="blue")
        self.status_label.pack(pady=20)

        # Buttons
        self.load_folder_btn = ttk.Button(control_frame, text="Load Image Folder", command=self.load_folder)
        self.load_folder_btn.pack(pady=10, fill=tk.X)
        self.prev_btn = ttk.Button(control_frame, text="Previous Image", command=self.prev_image)
        self.prev_btn.pack(pady=5, fill=tk.X)
        self.next_btn = ttk.Button(control_frame, text="Next Image", command=self.next_image)
        self.next_btn.pack(pady=5, fill=tk.X)
        self.load_video_btn = ttk.Button(control_frame, text="Load Video", command=self.load_video)
        self.load_video_btn.pack(pady=10, fill=tk.X)
        self.live_btn = ttk.Button(control_frame, text="Start Live Camera", command=self.start_live_camera)
        self.live_btn.pack(pady=10, fill=tk.X)
        self.calibrate_btn = ttk.Button(control_frame, text="Calibrate Camera", command=self.calibrate_camera)
        self.calibrate_btn.pack(pady=20, fill=tk.X)

        # ------------------------
        # Detector, Validator, Decoder
        # ------------------------
        self.detector = ObjectAndSymbolDetector()
        self.validator = ObjectValidator()
        self.decoder = ValueDecoder()

        # ------------------------
        # State
        # ------------------------
        self.current_image = None
        self.video_capture = None
        self.video_playing = False
        self.live_camera = False
        self.image_list = []
        self.image_index = 0
        self.target_fps = 30

        # Threading
        self.frame_queue = Queue(maxsize=1)  # latest raw frame
        self.processed_frame = None
        self.stop_threads = False

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start GUI update loop
        self.update_gui_loop()

    # ------------------------
    # Load image folder
    # ------------------------
    def load_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.image_list = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)])
        if not self.image_list:
            self.status_label.config(text="No images found", foreground="red")
            return
        self.image_index = 0
        self.load_current_image()

    def load_current_image(self):
        img_path = self.image_list[self.image_index]
        img = cv2.imread(img_path)
        if img is None:
            self.status_label.config(text=f"Failed to load: {img_path}", foreground="red")
            return
        self.current_image = img
        self.video_playing = False
        self.video_capture = None
        self.status_label.config(text=f"{self.image_index+1}/{len(self.image_list)}: {os.path.basename(img_path)}", foreground="green")
        self.processed_frame = self.process_frame(img)

    def next_image(self):
        if not self.image_list:
            return
        self.image_index = (self.image_index + 1) % len(self.image_list)
        self.load_current_image()

    def prev_image(self):
        if not self.image_list:
            return
        self.image_index = (self.image_index - 1) % len(self.image_list)
        self.load_current_image()

    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video", 
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not file_path:
            return

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            self.status_label.config(text="Failed to open video", foreground="red")
            return

        self.video_capture = cap
        self.video_playing = True
        self.live_camera = False
        self.status_label.config(text=f"Playing video: {os.path.basename(file_path)}", foreground="green")

        # Start threads
        threading.Thread(target=self.video_loop, daemon=True).start()
        threading.Thread(target=self.processing_loop, daemon=True).start() 



    def video_loop(self):
        while self.video_playing and self.video_capture and not self.stop_threads:
            ret, frame = self.video_capture.read()
            if not ret:
                self.video_capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
                continue
            self.enqueue_frame(frame)

    # ------------------------
    # Live camera
    # ------------------------
    def start_live_camera(self):
        if getattr(self, "camera", None) is None:
            self.camera = Camera(target_fps=self.target_fps)
        self.video_playing = False
        self.live_camera = True
        self.status_label.config(text="LIVE CAMERA", foreground="green")
        threading.Thread(target=self.camera_loop, daemon=True).start()
        threading.Thread(target=self.processing_loop, daemon=True).start()

    def camera_loop(self):
        while self.live_camera and not self.stop_threads:
            frame = self.camera.read()
            if frame is not None:
                self.enqueue_frame(frame)
            time.sleep(0.001)

    # ------------------------
    # Frame queue
    # ------------------------
    def enqueue_frame(self, frame):
        if not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()  # discard old frame
            except:
                pass
        self.frame_queue.put(frame)

    # ------------------------
    # Processing thread
    # ------------------------
    def processing_loop(self):
        while (self.live_camera or self.video_playing or self.current_image is not None) and not self.stop_threads:
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue
            frame = self.frame_queue.get()
            self.processed_frame = self.process_frame(frame)

    # ------------------------
    # Process a single frame/image
    # ------------------------
    def process_frame(self, frame):
        # Detect objects
        result = self.detector.detect(frame, visualise=False)

        # Convert frame to Pillow RGB image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 52)
        except:
            font = ImageFont.load_default()

        for obj in result['objects']:
            # Validate object
            self.validator.validate_object(obj)

            cx, cy = obj['centroid']

            # -------------------
            # If object is Fake, draw only "FAKE" and skip further processing
            # -------------------
            if obj['real_or_fake'] != "Real":
                text = "FAKE"
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                x0, y0 = int(cx - tw / 2), int(cy - th / 2)
                draw.rectangle([x0-4, y0-4, x0+tw+4, y0+th+4], fill=(0,0,0))
                draw.text((x0, y0), text, fill=(255, 50, 25), font=font)
                continue  

            # Only process Real objects
            symbols = obj.get('symbols', [])
            color = Counter([s['color'] for s in symbols]).most_common(1)[0][0]
            value = self.decoder.decode_object(symbols, color, obj['bbox_rect'], img_debug=frame)

            # Draw object contour
            obj_pts = np.array(obj['inner_area'], dtype=np.int32)
            draw.polygon([tuple(p[0]) for p in obj_pts], outline=(60, 255, 80), width=2)

            # Draw label
            text = f"{obj['real_or_fake']}:- {value}"
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x0, y0 = int(cx - tw / 2), int(cy - th / 2)
            draw.rectangle([x0-4, y0-4, x0+tw+4, y0+th+4], fill=(0,0,0))
            draw.text((x0, y0), text, fill=(60, 255, 80), font=font)

        return img_pil

    # ------------------------
    # GUI update loop
    # ------------------------
    def update_gui_loop(self):
        if self.processed_frame is not None:
            lbl_w = self.display_label.winfo_width()
            lbl_h = self.display_label.winfo_height()
            img_w, img_h = self.processed_frame.size
            scale = min(lbl_w/img_w, lbl_h/img_h) if lbl_w>1 else 1
            new_w, new_h = int(img_w*scale), int(img_h*scale)
            img_resized = self.processed_frame.resize((new_w,new_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (lbl_w,lbl_h), (0,0,0))
            x_off = (lbl_w - new_w) // 2
            y_off = (lbl_h - new_h) // 2

            canvas.paste(img_resized, (x_off, y_off))
            imgtk = ImageTk.PhotoImage(canvas)
            self.display_label.imgtk = imgtk
            self.display_label.configure(image=imgtk)
        self.window.after(10, self.update_gui_loop)

    # ------------------------
    # Calibrate camera
    # ------------------------
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

    # ------------------------
    # Close
    # ------------------------
    def on_close(self):
        # Stop threads
        self.stop_threads = True
        self.live_camera = False
        self.video_playing = False

        # Release resources
        if getattr(self, "camera", None):
            self.camera.release()
        if self.video_capture:
            self.video_capture.release()

        # Wait a moment for threads to exit
        time.sleep(0.1)

        self.window.destroy()


# =========================
# Run
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = ScannerGUI(root)
    root.mainloop()

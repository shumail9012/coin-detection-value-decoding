import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import sys
import cv2
import numpy as np
import csv
import datetime

from camera import Camera
from camera_calibrator import CameraCalibration
from object_detector import ObjectAndSymbolDetector
from value_decoder import ValueDecoder
from object_validator import ObjectValidator
from utility import utility

class ScannerGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Galactic Credit Scanner")
        self.window.geometry("1200x700")
        self.window.minsize(800, 500)
        self.log_rows = []

        # Calibration
        self.calibrator = CameraCalibration()
        self.calibrator.load("metric_camera.json")

        # Detector + Decoder + Validator
        self.detector = ObjectAndSymbolDetector()
        self.decoder = ValueDecoder()
        self.validator = ObjectValidator()

        # Camera
        self.camera = Camera(target_fps=15)

        # FPS
        self.last_frame_time = time.time()
        self.current_fps = 0
        self.target_fps = 15

        # Tracker
        self.trackers = {}  # oid -> cv2 tracker
        self.tracked_objects = {}  # oid -> object dict
        self.next_id = 1  # unique ID counter

        # Detection frequency
        self.detect_every_n_frames = 5
        self.frame_counter = 0

        # GUI
        self._build_gui()
        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_gui(self):
        main_frame = ttk.Frame(self.window)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        control_frame = ttk.Frame(main_frame, width=220)
        control_frame.grid(row=0, column=1, sticky="ns", padx=10)
        control_frame.grid_propagate(False)
        main_frame.grid_columnconfigure(1, weight=0)

        self.status_label = ttk.Label(control_frame, text="LIVE", font=("Helvetica", 16), foreground="green")
        self.status_label.pack(pady=20)
        self.capture_btn = ttk.Button(control_frame, text="Take Picture", command=self.capture_image)
        self.capture_btn.pack(pady=10, fill=tk.X)
        self.record_btn = ttk.Button(control_frame, text="Start Recording", command=self.toggle_recording)
        self.record_btn.pack(pady=10, fill=tk.X)
        self.calibrate_btn = ttk.Button(control_frame, text="Calibrate Camera", command=self.calibrate_camera)
        self.calibrate_btn.pack(pady=20, fill=tk.X)
        ttk.Button(control_frame, text="Export CSV", command=self.export_csv).pack(pady=10, fill=tk.X)

    def update_frame(self):
        frame = self.camera.read()
        if frame is None:
            self.window.after(10, self.update_frame)
            return
        self.frame_counter += 1

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------------------
        # Update existing trackers
        # -------------------
        remove_ids = []
        for oid, tracker in self.trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                self.tracked_objects[oid]["bbox_rect"] = (x, y, w, h)
            else:
                remove_ids.append(oid)

        # Remove lost trackers
        for oid in remove_ids:
            self.trackers.pop(oid)
            self.tracked_objects.pop(oid)

        # -------------------
        # Detect new objects every n frames
        # -------------------
        if self.frame_counter % self.detect_every_n_frames == 0:
            result = self.detector.detect(frame, visualise=False)
            for obj in result["objects"]:
                # Check if already tracked (based on overlap)
                x, y, w, h = utility.rect_to_bbox(obj["bbox_rect"])
                matched = False
                for tracked_obj in self.tracked_objects.values():
                    tx, ty, tw, th = utility.rect_to_bbox(tracked_obj["bbox_rect"])
                    iou = self._iou((x, y, w, h), (tx, ty, tw, th))
                    if iou > 0.3:
                        matched = True
                        break
                if not matched:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (x, y, w, h))
                    oid = self.next_id
                    self.next_id += 1
                    self.trackers[oid] = tracker
                    obj["id"] = oid
                    self.tracked_objects[oid] = obj

        # -------------------
        # Process and draw tracked objects
        # -------------------
        for oid, obj in self.tracked_objects.items():
            x, y, w, h = utility.rect_to_bbox(obj["bbox_rect"])

            if not obj.get("decode_locked", False):
                self.validator.validate_object(obj)
                if obj.get("real_or_fake") == "Real":
                    symbols = obj.get("symbols", [])
                    colour = symbols[0]["color"] if symbols else "-"
                    value = self.decoder.decode_object(symbols, colour, obj["bbox_rect"], img_debug=frame)
                else:
                    value = "FAKE"
                    colour = "FAKE"
                obj["decoded_value"] = value
                obj["decode_locked"] = True
            else:
                value = obj.get("decoded_value", "missing")
                colour = obj.get("decoded_value", "missing")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            self.log_rows.append([timestamp, oid, value, f"{obj['width']:.1f}", f"{obj['length']:.1f}", colour, obj.get("real_or_fake")])

            colour_box = (0, 255, 0) if obj.get("real_or_fake") == "Real" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour_box, 3)
            cv2.putText(frame, f"ID:{oid}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour_box, 2)

        # Render
        self._render_frame(frame)

        # FPS
        now = time.time()
        self.current_fps = 1 / max(now - self.last_frame_time, 1e-6)
        self.last_frame_time = now
        self.status_label.config(text=f"{'RECORDING' if self.camera.recording else 'LIVE'}\nFPS: {self.current_fps:.1f}",
                                 foreground="red" if self.camera.recording else "green")
        self.window.after(int(1000 / self.target_fps), self.update_frame)

    def _iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        return inter_area / float(bbox1_area + bbox2_area - inter_area + 1e-6)

    def _render_frame(self, frame):
        lbl_w, lbl_h = self.video_label.winfo_width(), self.video_label.winfo_height()
        img = Image.fromarray(frame[:, :, ::-1])
        if lbl_w > 1 and lbl_h > 1:
            scale = min(lbl_w / img.width, lbl_h / img.height)
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (lbl_w, lbl_h), (0, 0, 0))
            canvas.paste(img, ((lbl_w - img.width) // 2, (lbl_h - img.height) // 2))
        else:
            canvas = img
        imgtk = ImageTk.PhotoImage(canvas)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    # Controls
    def capture_image(self):
        filename = self.camera.capture_image()
        if filename:
            self.status_label.config(text=f"Saved\n{filename.split('/')[-1]}", foreground="blue")

    def toggle_recording(self):
        if self.camera.recording:
            self.camera.stop_recording()
            self.record_btn.config(text="Start Recording")
        else:
            if self.camera.start_recording():
                self.record_btn.config(text="Stop Recording")

    def calibrate_camera(self):
        if self.camera.recording:
            self.status_label.config(text="STOP RECORDING\nTO CALIBRATE", foreground="red")
            return
        self.status_label.config(text="CALIBRATING...\nOpenCV window", foreground="orange")
        self.window.update_idletasks()
        self.camera.release()
        success = CameraCalibration().run_live_calibration()
        self.camera = Camera(target_fps=self.target_fps)
        if success:
            self.calibrator.load("metric_camera.json")
            self.status_label.config(text="CALIBRATION OK", foreground="green")
        else:
            self.status_label.config(text="CALIBRATION FAILED", foreground="red")

    def export_csv(self):
        if not self.log_rows:
            self.status_label.config(text="No data to export", foreground="orange")
            return
        filename = f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "id", "value", "width_mm", "length_mm", "colour", "real_or_fake"])
            writer.writerows(self.log_rows)
        self.status_label.config(text=f"CSV Exported", foreground="blue")

    def on_close(self):
        self.camera.stop_recording()
        self.camera.release()
        self.window.destroy()
        sys.exit()

# Run
if __name__ == "__main__":
    root = tk.Tk()
    app = ScannerGUI(root)
    root.mainloop()

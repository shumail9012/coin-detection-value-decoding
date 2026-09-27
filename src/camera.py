"""
File: camera.py
Project: Coin Detection and Value Decoding System

Author: Suraj Karki
Affiliation: Master's Programme – AI and Automation, Högskolan Väst
Date: 2026-01-12

Description:
    Camera interface module to capture images and videos using Basler cameras or fallback to OpenCV.
"""
import cv2
from datetime import datetime
import os
from pypylon import pylon
import numpy as np

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

class Camera:
    def __init__(self, exposure=800, gain=6, device_index=0, target_fps=15, image_path=None):
        """
        If image_path is provided, the camera will read from that image instead of a live camera.
        """
        self.backend = None
        self.cap = None
        self.converter = None
        self.image_path = image_path
        self.static_image = None

        self.exposure = exposure
        self.gain = gain
        self.recording = False
        self.video_writer = None
        self.target_fps = target_fps

        self.pixels_per_mm_x = None
        self.pixels_per_mm_y = None

        if self.image_path:
            self.static_image = cv2.imread(self.image_path)
            if self.static_image is None:
                raise FileNotFoundError(f"Static image not found: {self.image_path}")
            self.width = self.static_image.shape[1]
            self.height = self.static_image.shape[0]
            print(f"[INFO] Using static image: {self.image_path}")
            return

        # Try Basler first
        try:
            factory = pylon.TlFactory.GetInstance()
            devices = factory.EnumerateDevices()
            if len(devices) == 0:
                raise RuntimeError("No Basler camera found")

            self.cap = pylon.InstantCamera(factory.CreateFirstDevice())
            self.cap.Open()
            self.cap.ExposureTime.SetValue(self.exposure)
            self.cap.Gain.SetValue(self.gain)
            self.cap.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.backend = "basler"
            self.width = self.cap.Width.GetValue()
            self.height = self.cap.Height.GetValue()
            print(f"[INFO] Basler camera resolution: {self.width} x {self.height}")

        except Exception as e:
            print(f"[WARN] Basler not available ({e})")
            print("[INFO] Using OpenCV VideoCapture")

            self.cap = cv2.VideoCapture(device_index)
            if not self.cap.isOpened():
                raise RuntimeError("No camera available")

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.backend = "opencv"
            print(f"[INFO] OpenCV camera resolution: {self.width} x {self.height}")

    # =========================
    # Read a frame (live or static)
    # =========================
    def read(self):
        if self.static_image is not None:
            return self.static_image.copy()

        frame = None
        if self.backend == "basler":
            grab = self.cap.RetrieveResult(1500, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                image = self.converter.Convert(grab)
                frame = image.GetArray()
            grab.Release()
        else:
            ret, frame = self.cap.read()
            if not ret:
                frame = None

        if frame is not None and self.recording:
            self.write_frame(frame)
        return frame

    # =========================
    # Capture single image
    # =========================
    def capture_image(self):
        frame = self.read()
        if frame is None:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(DATASET_DIR, f"img_{timestamp}.bmp")
        cv2.imwrite(filename, frame)
        return filename

    # =========================
    # Start/stop recording
    # =========================
    def start_recording(self):
        if self.recording:
            return None
        frame = self.read()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(DATASET_DIR, f"video_{timestamp}.avi")
        self.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            self.target_fps,
            (w, h)
        )
        self.recording = True
        return video_path

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

    def write_frame(self, frame):
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def release(self):
        if self.recording:
            self.stop_recording()
        if self.static_image is not None:
            return
        if self.backend == "basler":
            if self.cap.IsGrabbing():
                self.cap.StopGrabbing()
            self.cap.Close()
        else:
            self.cap.release()

    # =========================
    # Get native resolution
    # =========================
    def get_resolution(self):
        return self.width, self.height

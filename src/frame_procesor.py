import cv2
import threading
import time
from collections import Counter
import numpy as np
from adaptive_threshold import AdaptiveResolutionController


class FrameProcessor:
    def __init__(self, producer, detector, validator, decoder):
        self.producer = producer
        self.detector = detector
        self.validator = validator
        self.decoder = decoder

        self.output = None
        self.running = True
        self.lock = threading.Lock()

        self.res_controller = AdaptiveResolutionController(
            target_fps=15,
            min_width=640,
            max_width=1280
        )

        self.last_time = time.time()
        self.fps = 0.0

        threading.Thread(target=self._loop, daemon=True).start()

    # -------------------------------------------------
    # Main processing loop
    # -------------------------------------------------
    def _loop(self):
        while self.running:
            frame = self.producer.get_latest()
            if frame is None:
                time.sleep(0.001)
                continue

            start = time.time()

            # --- Adaptive resolution ---
            max_width = self.res_controller.update(self.fps)
            frame_small, scale = self.resize_for_processing(frame, max_width)

            processed = self.process_frame(frame_small)

            # --- FPS measurement (end-to-end) ---
            dt = time.time() - start
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

            with self.lock:
                self.output = processed

    def get_latest(self):
        with self.lock:
            return self.output

    def stop(self):
        self.running = False

    # -------------------------------------------------
    # Frame processing (resized domain ONLY)
    # -------------------------------------------------
    def process_frame(self, frame_bgr):
        frame = frame_bgr.copy()  # safety

        result = self.detector.detect(frame, visualise=False)

        for obj in result.get("objects", []):
            self.validator.validate_object(obj)

            color, value = "-", "-"
            symbols = obj.get("symbols", [])

            if symbols:
                color = Counter(
                    s["color"] for s in symbols
                ).most_common(1)[0][0]

                value = self.decoder.decode_object(
                    symbols,
                    color,
                    obj["bbox_rect"],
                    img_debug=frame
                )

            # --- Draw contours ---
            pts = np.asarray(obj["inner_area"], dtype=np.int32)
            cv2.polylines(
                frame,
                [pts],
                True,
                (0, 255, 0) if obj["real_or_fake"] == "Real" else (0, 0, 255),
                2
            )

            cx, cy = map(int, obj["centroid"])
            label = f"{obj['real_or_fake']} | {value} | {color}"

            cv2.putText(
                frame,
                label,
                (cx - 80, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------
    # Resize utility
    # -------------------------------------------------
    def resize_for_processing(self, frame, max_width):
        h, w = frame.shape[:2]

        if w <= max_width:
            return frame, 1.0

        scale = max_width / w
        new_size = (int(w * scale), int(h * scale))

        resized = cv2.resize(
            frame,
            new_size,
            interpolation=cv2.INTER_AREA
        )

        return resized, scale

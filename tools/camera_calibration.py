import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CalibrationResult:
    pixels_per_mm: float
    square_sizes_px: List[float]
    num_squares: int


class CalibrationDetector:
    """
    Calibration detector using known red reference squares.
    Designed for industrial machine vision.
    """

    def __init__(
        self,
        square_size_mm: float = 5.0,
        min_area_px: int = 150
    ):
        self.square_size_mm = square_size_mm
        self.min_area_px = min_area_px

        # Red HSV ranges (OpenCV scale)
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

        # ---- Internal state ----
        self.image = None
        self.clean_mask = None
        self.contours = None

    # =========================
    # Public API
    # =========================
    def calibrate(self, image_bgr: np.ndarray, debug: bool = False) -> CalibrationResult:
        self.image = image_bgr.copy()

        mask = self._create_red_mask(self.image)
        self.clean_mask = self._clean_mask(mask)

        self.contours = self._extract_valid_squares(self.clean_mask)
        if len(self.contours) == 0:
            raise RuntimeError("No valid calibration squares detected")

        square_sizes_px = self._measure_squares(self.contours)

        pixels_per_mm = np.median(square_sizes_px) / self.square_size_mm

        if debug:
            self.visualise_static()

        return CalibrationResult(
            pixels_per_mm=float(pixels_per_mm),
            square_sizes_px=square_sizes_px,
            num_squares=len(square_sizes_px)
        )

    # =========================
    # Masking
    # =========================
    def _create_red_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        sv_mask = cv2.inRange(s, 120, 255) & cv2.inRange(v, 80, 255)

        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)

        return (mask1 | mask2) & sv_mask

    # =========================
    # Mask cleaning
    # =========================
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.medianBlur(mask, 5)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area_px:
                clean_mask[labels == i] = 255

        return clean_mask

    # =========================
    # Shape extraction
    # =========================
    def _extract_valid_squares(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0

            if 0.85 <= aspect <= 1.15:
                valid.append(cnt)

        return sorted(valid, key=cv2.contourArea, reverse=True)

    # =========================
    # Measurement
    # =========================
    def _measure_squares(self, contours):
        return [(cv2.boundingRect(c)[2] + cv2.boundingRect(c)[3]) / 2 for c in contours]

    # =========================
    # Visualisation (STATIC)
    # =========================
    def visualise_static(self):
        if self.image is None or self.clean_mask is None or self.contours is None:
            raise RuntimeError("visualise_static() called before calibrate()")

        vis = self.image.copy()

        for i, cnt in enumerate(self.contours):
            x, y, w, h = cv2.boundingRect(cnt)
            size_px = (w + h) / 2

            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{i+1}: {size_px:.1f}px",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        plt.figure(figsize=(16, 5))

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Clean Red Mask")
        plt.imshow(self.clean_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Detected Calibration Squares")
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.title("Clean Red Mask with Contours")
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

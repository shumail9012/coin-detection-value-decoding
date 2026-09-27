"""
File: object_detector.py
Project: Coin Detection and Value Decoding System

Author: Suraj Karki
Affiliation: Master's Programme – AI and Automation, Högskolan Väst
Date: 2026-01-12

Description:
    Detects coin objects in the frame:

    1. Outer contour of each coin
    2. Symbols (yellow, blue, red squares/rectangles) inside each coin
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utility import utility
from object_validator import ObjectValidator

class ObjectAndSymbolDetector:
    """Detects objects and symbols in an image."""

    def __init__(self, min_area=2000, min_symbol_area=500):
        self.min_area = min_area
        self.min_symbol_area = min_symbol_area
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    def detect(self, img_bgr, visualise=False):
        out = {}
        img_h, img_w = img_bgr.shape[:2]

        # --------------------------
        # 1. Remove green background (HSV)
        # --------------------------
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

         # Adaptive median green hue
        green_pixels = h[(h > 35) & (h < 85)]
        if len(green_pixels) > 0:
            green_hue = int(np.median(green_pixels))
        else:
            green_hue = 60
        lower = np.array([green_hue-15, 40, 40])
        upper = np.array([green_hue+15, 255, 255])
        green_mask = cv2.inRange(hsv, lower, upper)

        # Invert to get foreground mask
        fg_mask = cv2.bitwise_not(green_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        fg_mask = cv2.medianBlur(fg_mask, 3)
        # --------------------------
        # 2. Outer contour detection
        # --------------------------
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        objects = []
        final_mask = np.zeros_like(fg_mask)

        for cnt in contours:
            # Reject partially visible objects
            touches_edge = utility.touches_border(cnt, img_w, img_h)
            if touches_edge:
                continue

            # Rotated rectangle
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w_px, h_px), _ = rect

            # Rotated box points
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            box_pts = box.reshape((-1, 1, 2))

            # Axis-aligned bounding box
            x, y, w, h = cv2.boundingRect(box)

            # Object mask
            mask = np.zeros(fg_mask.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [box_pts], 255)
            cv2.fillPoly(final_mask, [box_pts], 255)

            width_mm, length_mm = utility.px_to_mm(w_px, h_px)
            obj = {
                "centroid": (cx, cy),
                "bbox_rect": rect,
                "rotated_rect": [int(x), int(y), int(w), int(h)],
                "inner_area": box_pts,
                "width_px": float(w_px),
                "length_px": float(h_px),
                "width": float(width_mm),
                "length": float(length_mm),
                "mask": mask,
                "symbols": []
            }

            # --------------------------
            # 3. Symbol detection
            # --------------------------
            color_ranges = utility.COLOR_RANGES
            for color, ranges in color_ranges.items():
                color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for low, high in ranges:
                    color_mask |= cv2.inRange(hsv, low, high)
                color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)

                # Morph cleaning
                color_mask = cv2.medianBlur(color_mask, 11)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,
                                              cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

                cnts_sym, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts_sym:
                    if cv2.contourArea(c) < self.min_symbol_area:
                        continue

                    rrect = cv2.minAreaRect(c)
                    (_, (w_px_sym, h_px_sym), _) = rrect
                    box_sym = cv2.boxPoints(rrect).astype(np.int32)
                    approx = utility._sort_corners_clockwise(box_sym.reshape((-1, 1, 2)))
                    sym_width, sym_length = utility.px_to_mm(w_px_sym, h_px_sym)
                    centroid = tuple(np.int64(rrect[0]))
                    rel_centroid = (centroid[0] - cx, centroid[1] - cy)
                    obj['symbols'].append({
                        "contour": approx,
                        "s_width": sym_width,
                        "s_length": sym_length,
                        "width_px": w_px_sym,
                        "length_px": h_px_sym,
                        "centroid": centroid,
                        "color": color,
                        "rel_centroid": rel_centroid,
                        "orientation": rrect[2]
                    })

            objects.append(obj)

        # --------------------------
        # 4. Output
        # --------------------------
        out.update({
            "objects": objects,
            "metric_mask": fg_mask,
            "final_mask": final_mask
        })

        if visualise:
            self._visualise(img_bgr, out)

        return out

#     def _visualise(self, img, out):
#         overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         for obj in out['objects']:
#             cv2.polylines(overlay, [obj['inner_area']], True, (0, 0, 255), 2)
#             for s in obj['symbols']:
#                 cv2.polylines(
#                     overlay, [s['contour']], True,
#                     {"red": (0, 255, 0), "blue": (255, 0, 255), "yellow": (255, 0, 0)}.get(s['color'], (255, 255, 255)), 2
#                 )
#                 # draw symbol centroid
#                 cx, cy = obj['centroid']
#                 sx, sy = int(cx + s['rel_centroid'][0]), int(cy + s['rel_centroid'][1])
#                 cv2.circle(overlay, (sx, sy), 4, (255, 255, 0), -1)
#         plt.figure(figsize=(8, 8))
#         plt.imshow(overlay)
#         plt.title("Objects & Symbols")
#         plt.show()


# # =========================
# # dEBG PIPE
# # =========================
# if __name__ == "__main__":
#     img = cv2.imread("sensorTechnologyAndImageAnalysis/dataset/blue_test.bmp")
#     detector = ObjectAndSymbolDetector()
#     validator = ObjectValidator()
#     result = detector.detect(img, visualise=True)

#     for i, obj in enumerate(result['objects']):
#         validator.validate_object(obj)
#         print(f"\nObject {i+1}: Width={obj['width']:.1f}, Length={obj['length']:.1f}")
#         print(f"  Symbols detected: {len(obj['symbols'])}")
#         print(f"  Real or fake: {obj['real_or_fake']}")
#         for s in obj['symbols']:
#             print(f"    Color={s['color']}, Width={s['s_width']:.1f} mm, Length={s['s_length']:.1f} mm, Centroid={s['centroid']}")

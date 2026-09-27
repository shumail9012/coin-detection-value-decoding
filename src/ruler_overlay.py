import cv2
import numpy as np

def draw_mm_rulers(
    image: np.ndarray,
    pixels_per_mm: float,
    major_mm: int = 10,
    minor_mm: int = 1
) -> np.ndarray:
    """
    Draw metric rulers (mm) on top and left of a rectified image.
    """
    overlay = image.copy()
    h, w = overlay.shape[:2]

    px_per_mm = pixels_per_mm

    major_px = int(major_mm * px_per_mm)
    minor_px = int(minor_mm * px_per_mm)

    colour_major = (0, 255, 0)
    colour_minor = (0, 180, 0)
    thickness_major = 2
    thickness_minor = 1

    # -------------------------
    # Horizontal ruler (top)
    # -------------------------
    for x in range(0, w, minor_px):
        if x % major_px == 0:
            cv2.line(overlay, (x, 0), (x, 20), colour_major, thickness_major)
            cv2.putText(
                overlay,
                f"{x // px_per_mm:.0f}",
                (x + 2, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                colour_major,
                1
            )
        else:
            cv2.line(overlay, (x, 0), (x, 10), colour_minor, thickness_minor)

    # -------------------------
    # Vertical ruler (left)
    # -------------------------
    for y in range(0, h, minor_px):
        if y % major_px == 0:
            cv2.line(overlay, (0, y), (20, y), colour_major, thickness_major)
            cv2.putText(
                overlay,
                f"{y // px_per_mm:.0f}",
                (25, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                colour_major,
                1
            )
        else:
            cv2.line(overlay, (0, y), (10, y), colour_minor, thickness_minor)

    return overlay

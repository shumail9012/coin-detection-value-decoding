import numpy as np
import cv2
import matplotlib.pyplot as plt

class utility:
    PIXELS_PER_MM_X = 8.25
    PIXELS_PER_MM_Y = 8.25

    COLOR_RANGES = {
        "yellow": [((20, 70, 70), (25, 255, 255)),
                   ((25, 70, 70), (35, 255, 255)),
                   ((35, 70, 70), (40, 255, 255))],
        "blue": [((95, 60, 60), (110, 255, 255)),
                 ((110, 60, 60), (125, 255, 255)),
                 ((125, 60, 60), (140, 255, 255))],
        "red": [((0, 100, 80), (10, 255, 255)),
                ((160, 100, 80), (180, 255, 255))]
    }

    @staticmethod
    def get_center_roi(frame, roi_size):
        h, w = frame.shape[:2]
        rw, rh = roi_size
        x1 = max(w // 2 - rw // 2, 0)
        y1 = max(h // 2 - rh // 2, 0)
        x2 = min(w // 2 + rw // 2, w)
        y2 = min(h // 2 + rh // 2, h)
        roi = frame[y1:y2, x1:x2].copy()
        return roi, (x1, y1, x2, y2)

    @staticmethod
    def _sort_corners_clockwise(corners):
        corners = corners.reshape(4, 2)
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1).flatten()
        top_left = corners[np.argmin(s)]
        bottom_right = corners[np.argmax(s)]
        top_right = corners[np.argmin(diff)]
        bottom_left = corners[np.argmax(diff)]
        return np.array([top_left, top_right, bottom_right, bottom_left]).reshape(-1, 1, 2)

    @staticmethod
    def px_to_mm(width_px: float, height_px: float) -> tuple[float, float]:

        w_mm = width_px / utility.PIXELS_PER_MM_X
        h_mm = height_px / utility.PIXELS_PER_MM_Y
        # Ensure width <= length convention
        width_mm, length_mm = sorted([w_mm, h_mm])
        return width_mm, length_mm

    @staticmethod
    def _compute_short_edges_mm(bbox_rect):
        pts = np.int32(cv2.boxPoints(bbox_rect))
        edges = [pts[1]-pts[0], pts[2]-pts[1], pts[3]-pts[2], pts[0]-pts[3]]
        edges_lengths = [np.linalg.norm(e) for e in edges]
        short_indices = np.argsort(edges_lengths)[:2]

        short_edges_mm = []
        for idx in short_indices:
            start_pt = pts[idx]
            end_pt = pts[(idx+1)%4]
            short_edges_mm.append([
                np.array([start_pt[0]/utility.PIXELS_PER_MM_X, start_pt[1]/utility.PIXELS_PER_MM_Y]),
                np.array([end_pt[0]/utility.PIXELS_PER_MM_X, end_pt[1]/utility.PIXELS_PER_MM_Y])
            ])
        return short_edges_mm

    @staticmethod
    def draw_debug(img, bbox_rect, value_rows):
        """Draw the short edges and row centroids for debugging."""
        _, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Draw bbox
        pts = cv2.boxPoints(bbox_rect)
        pts = np.int32(pts)
        poly = plt.Polygon(pts, fill=None, edgecolor='blue', linewidth=2)
        ax.add_patch(poly)

        # Draw short edges
        short_edges_mm = utility._compute_short_edges_mm(bbox_rect)
        for edge_mm in short_edges_mm:
            start_px = edge_mm[0] * np.array([utility.PIXELS_PER_MM_X, utility.PIXELS_PER_MM_Y])
            end_px = edge_mm[1] * np.array([utility.PIXELS_PER_MM_X, utility.PIXELS_PER_MM_Y])
            ax.plot([start_px[0], end_px[0]], [start_px[1], end_px[1]], 'w-', linewidth=2)

        # Draw symbol centroids
        for i, row in enumerate(value_rows):
            for s in row:
                cx, cy = s['centroid']
                ax.plot(cx, cy, 'ro')
                ax.text(cx + 2, cy - 2, str(i), color='w', fontsize=8)

        ax.set_title("Orientation Debug with Short Edges")
        ax.axis('off')
        plt.show()
    
    def point_inside_polygon(point, polygon):
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    @staticmethod
    def touches_border(contour, img_w, img_h, margin=2):
        """
        Returns True if the contour touches any image border
        within a given pixel margin.
        """

        # Flatten contour to Nx2 safely
        pts = contour.reshape(-1, 2)

        # Left border
        if np.any(pts[:, 0] <= margin):
            return True

        # Right border
        if np.any(pts[:, 0] >= img_w - 1 - margin):
            return True

        # Top border
        if np.any(pts[:, 1] <= margin):
            return True

        # Bottom border
        if np.any(pts[:, 1] >= img_h - 1 - margin):
            return True

        return False
    
    @staticmethod
    def rect_to_bbox(rect):
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        x, y, w, h = cv2.boundingRect(box)
        return x, y, w, h



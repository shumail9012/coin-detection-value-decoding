"""
File: value_decoder.py
Project: Coin Detection and Value Decoding System

Author: Suraj Karki
Affiliation: Master's Programme – AI and Automation, Högskolan Väst
Date: 2026-01-12

Description:
    Decodes the numeric value of detected coin objects using
    rule-based geometric measurements.
"""
import numpy as np
from utility import utility
from symbol_classifier import SymbolClassifier

class ValueDecoder:

    VALUE_MAP = {
        (0, 0, 1): 0,
        (0, 0, 2): 1,
        (0, 0, 3): 2,
        (0, 0, 4): 3,
        (0, 1, 1): 4,
        (0, 1, 2): 5,
        (0, 1, 3): 6,
        (1, 0, 1): 7,
        (1, 0, 2): 8,
        (1, 0, 3): 9,
    }

    def __init__(self):
        self.cls = SymbolClassifier()

    def row_signature(self, row):
        rect2 = rect3 = square = 0
        for s in row:
            if s['type'] == 'square':
                square += 1
            elif s['type'] == 'rect2':
                rect2 = 1
            elif s['type'] == 'rect3':
                rect3 = 1
        return (rect3, rect2, square)
    
    def _row_distance_to_edge(self, row, edge_mm):
        """
        row point → short edge segment
        """
        P = row[-1]['centroid_mm']          # (x, y)
        A, B = edge_mm                      # each (x, y)

        return self.point_to_segment_distance(P, A, B)

    def _row_distance_to_gold(self, row, gold_centroid_mm):
        """
        row centroid → gold centroid
        """
        row_c = row[-1]['centroid_mm']     # (x, y)
        gold_c = np.array(gold_centroid_mm, dtype=float)

        return self.point_to_point_distance(row_c, gold_c)

    
    def point_to_segment_distance(self, P, A, B):
        """
        Distance from point P to segment AB (all in mm).
        P, A, B are (x, y)
        """
        P = np.array(P, dtype=float)
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)

        AB = B - A
        AP = P - A

        denom = np.dot(AB, AB)
        if denom == 0:
            return np.linalg.norm(AP)

        t = np.dot(AP, AB) / denom
        t = np.clip(t, 0.0, 1.0)

        closest = A + t * AB
        return np.linalg.norm(P - closest)
    
    def point_to_point_distance(self, P, Q):
        """
        Euclidean distance between two points (x, y) in mm
        """
        P = np.array(P, dtype=float)
        Q = np.array(Q, dtype=float)
        return np.linalg.norm(P - Q)

    
    def decode_object(self, symbols, color, bbox_rect, img_debug=None):
        if not symbols:
            return None
        
        gold_symbols = []
        normal_symbols = []

        # -----------------------------
        # 1. Classify symbols
        # -----------------------------
        for s in symbols:
            w_mm, h_mm = s['s_width'], s['s_length']
            s['type'] = self.cls.classify_from_mm(w_mm, h_mm)
            c_px = np.array(s['centroid'], dtype=np.float32)
            s['centroid_mm'] = np.array([c_px[0]/utility.PIXELS_PER_MM_X, c_px[1]/utility.PIXELS_PER_MM_Y])
            if s['type'] == 'gold':
                gold_symbols.append(s)
            else:
                normal_symbols.append(s)

        if not normal_symbols:
            return None

        # -----------------------------
        # 2. Compute object axes
        # -----------------------------
        (_, _), (w, h), angle = bbox_rect
        theta = np.deg2rad(angle)
        #Hough transform gives angle wrt horizontal axis
        ux = np.array([np.cos(theta), np.sin(theta)])
        uy = np.array([-np.sin(theta), np.cos(theta)])
        long_axis, short_axis = (ux, uy) if w >= h else (uy, ux)

        # -----------------------------
        # 3. Project symbols into object frame
        # -----------------------------
        for s in normal_symbols:
            c = s['centroid_mm']
            s['u'] = np.dot(c, long_axis)
            s['v'] = np.dot(c, short_axis)

        symbols_sorted = sorted(normal_symbols, key=lambda s: s['u'])

        # -----------------------------
        # 5. Row formation by pixel
        # -----------------------------
        rows = []
        for s in symbols_sorted:
            placed = False
            for row in rows:
                # Compute distance along long axis (u) to all symbols in this row
                distances = [abs(s['u'] - sym['u']) for sym in row]
                if  min(distances) <= 9:  # threshold, self width (5)+ space (3)+ 1 mm margin
                    row.append(s)
                    placed = True
                    break
            if not placed:
                # Start a new row
                rows.append([s])

        # ----------------------------- 
        #  7. Determine row order 
        # # # ----------------------------- 
        if gold_symbols: 
            # Compute distances from each row to the gold symbol 
            row_distances = [(row, self._row_distance_to_gold(row, gold_symbols[0]['centroid_mm'])) for row in rows] 
            # Sort rows based on distance 
            row_distances.sort(key=lambda x: x[1]) 
            # Unpack sorted rows and distances
            sorted_rows, sorted_distances = zip(*row_distances) 
            #print(f"[DEBUG] Sorted distances: {sorted_distances}") 
            rows = list(sorted_rows) 
            # Merge rows of length 1 into the nearest previous row 
            #remove this later
            merged_rows = [] 
            for row in rows: 
                if len(row[0]) == 1 and merged_rows: merged_rows[-1].extend(row) 
                else: merged_rows.append(row) 
                rows = merged_rows
        else:
            short_edges_mm = utility._compute_short_edges_mm(bbox_rect) 
            distances = [min([self._row_distance_to_edge(row, edge) for edge in short_edges_mm]) 
                        for row in rows] 
            if distances[0] > distances[-1]: 
                rows = list(reversed(rows)) 
            else: rows = rows.copy()

        # -----------------------------
        # 8. Decode rows using VALUE_MAP
        # -----------------------------
        row_values = []
        for idx, row in enumerate(rows):
                    # Compute distance to each short edge
            #edge_distances = []
            # for e_idx, edge in enumerate(short_edges_mm):
            #     d = self._row_distance_to_edge(row, edge)
            #     edge_distances.append(d)
            #     print(f"[DEBUG] Row {idx} → Short edge {e_idx}: {d:.2f} mm")

            # min_dist = min(edge_distances)
            # print(f"[DEBUG] Row {idx} → Min edge distance: {min_dist:.2f} mm")
            row_no_gold = [s for s in row if s['type'] != 'gold']
            if not row_no_gold:
                continue
            sig = self.row_signature(row)
            digit = self.VALUE_MAP.get(sig)
            if digit is not None:
                row_values.append(digit)

        if not row_values:
            return None
        
        # if img_debug is not None: 
        #     utility.draw_debug(img_debug.copy(), bbox_rect, rows)

        # -----------------------------
        # 9. Compute final value by color
        # -----------------------------
        if color=='red':
            value = 1
            for d in row_values:
                value *= d
            return value
        if color=='blue':
            return int("".join(map(str,row_values)))
        if color=='yellow':
            return int("".join(map(str,row_values))) * 10

        return None


# ------------------------
# Debug pipe
# ------------------------

# # Imort for debugging purposes
# import cv2
# import matplotlib.pyplot as plt
# from collections import Counter
# from object_detector import ObjectAndSymbolDetector
# from object_validator import ObjectValidator
"""if __name__ == "__main__":

    decoder = ValueDecoder()
    detector = ObjectAndSymbolDetector()
    validator = ObjectValidator()

    img = cv2.imread("sensorTechnologyAndImageAnalysis/dataset/blue_test.bmp")
    result = detector.detect(img, visualise=False)
    #validator.validate_object(result)


    for i, obj in enumerate(result['objects']):
        validator.validate_object(obj)
        print(f"\nObject {i+1}: Width={obj['width']:.1f}, Length={obj['length']:.1f}")
        print(f"  Symbols detected: {len(obj['symbols'])}")
        print(f"  Real or fake: {obj['real_or_fake']}")
    
        symbols = obj['symbols']
        color = Counter([s['color'] for s in symbols]).most_common(1)[0][0]
        value = decoder.decode_object(symbols, color, obj['bbox_rect'], img_debug=img)
        print(f"Object {i+1}: Value={value}, Color={color}")"""
        
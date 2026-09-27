import numpy as np
from scipy.optimize import linear_sum_assignment

class ObjectTracker:
    """
    Deterministic tracker for fully visible conveyor objects.
    No prediction, no Kalman, no correction.
    """

    def __init__(self,
                 max_distance_px=120,
                 backward_gate_px=20,
                 min_age=2,
                 max_missed=3):
        self.tracks = {}      # id -> track dict
        self.next_id = 1

        self.max_distance_px = max_distance_px
        self.backward_gate_px = backward_gate_px
        self.min_age = min_age
        self.max_missed = max_missed

    # ------------------------
    # Create new track
    # ------------------------
    def _create_track(self, det):
        oid = self.next_id
        self.next_id += 1

        cx, cy = det["centroid"]

        self.tracks[oid] = {
            "centroid": (cx, cy),
            "age": 1,
            "missed": 0,

            # Freeze geometry at creation
            "width": det.get("width"),
            "length": det.get("length"),

            # Accumulated symbols
            "symbols": [
                {**s, "rel_centroid": (s["centroid"][0] - cx,
                                       s["centroid"][1] - cy)}
                for s in det.get("symbols", [])
            ],

            "decode_locked": False
        }

        det["id"] = oid

    # ------------------------
    # Update tracker
    # ------------------------
    def update(self, detections):

        # Age all tracks
        for track in self.tracks.values():
            track["age"] += 1
            track["missed"] += 1

        # Bootstrap
        if not self.tracks:
            for d in detections:
                self._create_track(d)
            return []

        track_ids = list(self.tracks.keys())
        cost = np.full((len(track_ids), len(detections)), np.inf, dtype=np.float32)

        # ------------------------
        # Cost matrix
        # ------------------------
        for i, oid in enumerate(track_ids):
            px, py = self.tracks[oid]["centroid"]

            for j, det in enumerate(detections):
                cx, cy = det["centroid"]

                # Directional gating
                if cx < px - self.backward_gate_px:
                    continue

                dist = np.hypot(px - cx, py - cy)
                if dist <= self.max_distance_px:
                    cost[i, j] = dist

        matched_tracks = set()
        matched_dets = set()

        # ------------------------
        # Assignment
        # ------------------------
        if not np.isinf(cost).all():
            rows, cols = linear_sum_assignment(cost)

            for r, c in zip(rows, cols):
                if cost[r, c] == np.inf:
                    continue

                oid = track_ids[r]
                det = detections[c]
                cx, cy = det["centroid"]

                track = self.tracks[oid]

                # Update kinematics
                track["centroid"] = (cx, cy)
                track["missed"] = 0

                # Accumulate symbols until locked
                if not track["decode_locked"]:
                    for s in det.get("symbols", []):
                        track["symbols"].append({
                            **s,
                            "rel_centroid": (s["centroid"][0] - cx,
                                             s["centroid"][1] - cy)
                        })

                det["id"] = oid
                matched_tracks.add(oid)
                matched_dets.add(c)

        # ------------------------
        # New tracks
        # ------------------------
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._create_track(det)

        # ------------------------
        # Remove dead tracks
        # ------------------------
        dead = [
            oid for oid, t in self.tracks.items()
            if t["missed"] > self.max_missed
        ]
        for oid in dead:
            del self.tracks[oid]

        # ------------------------
        # Output mature tracks
        # ------------------------
        results = []
        for det in detections:
            oid = det.get("id")
            if oid is None or oid not in self.tracks:
                continue

            track = self.tracks[oid]
            if track["age"] >= self.min_age:
                results.append({
                    **det,
                    "symbols": track["symbols"],
                    "width": track["width"],
                    "length": track["length"],
                    "decode_locked": track["decode_locked"],
                    "age": track["age"]
                })

        return results

"""
File: object_detector.py
Project: Coin Detection and Value Decoding System

Author: Suraj Karki
Affiliation: Master's Programme – AI and Automation, Högskolan Väst
Date: 2026-01-12

Description:
    Validates objects as real or fake based on physical dimensions
    and color.
"""

class ObjectValidator:
    """Validates detected objects based on expected sizes."""
    
    # Expected sizes per symbol color (width x length in mm)
    EXPECTED_SIZES_MM = {
        "red": (38, 76),     # width, length
        "blue": (38, 150),
        "yellow": (38, 114)
    }

    TOLERANCE_MM = 5  # ±5 mm tolerance


    def validate_object(self, obj: dict) -> dict:
        """
        Validate a detected object. Adds a 'real_or_fake' flag to the object.
        """
        # If the object has symbols, use first one to determine color
        if obj['symbols']:
            symbol = obj['symbols'][0] 
            color = symbol['color']

            expected_w_mm, expected_l_mm = self.EXPECTED_SIZES_MM[color]
            w, l = obj['width'], obj['length']

            print(f"Expected size: {expected_w_mm:.2f} mm x {expected_l_mm:.2f} mm")
            print(f"Measured size: {w:.2f} mm x {l:.2f} mm")

        else:
            # No symbol detected, mark as fake
            obj['real_or_fake'] = "Fake"
            return obj

        # Check tolerance
        if (expected_w_mm - self.TOLERANCE_MM <= w <= expected_w_mm + self.TOLERANCE_MM and
            expected_l_mm - self.TOLERANCE_MM <= l <= expected_l_mm + self.TOLERANCE_MM):
            obj['real_or_fake'] = "Real"
        else:
            obj['real_or_fake'] = "Fake"

        return obj

    def validate_all(self, objects: list[dict]) -> list[dict]:
        """
        Validate a list of detected objects.
        """
        return [self.validate_object(obj) for obj in objects]

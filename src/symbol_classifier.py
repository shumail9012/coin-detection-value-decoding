"""
File: symbol_classifier.py
Project: Coin Detection and Value Decoding System

Author: Suraj Karki
Affiliation: Master's Programme – AI and Automation, Högskolan Väst
Date: 2026-01-12

Description:
Classifies detected symbols based on size.

"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    width_mm: float
    height_mm: float

class SymbolClassifier:
    """
    Classifies detected symbols based on size.
    """

    SYMBOLS = [
        SymbolSpec("square", 5.0, 5.0),
        SymbolSpec("rect2", 5.0, 13.0),
        SymbolSpec("rect3", 5.0, 21.0),
        SymbolSpec("gold", 13.0, 17.0),
    ]

    def __init__(
        self,
        max_tol_mm: int = 3
    ):
        self.tol_mm = max_tol_mm

    def classify_from_mm(
        self,
        width_mm: float,
        height_mm: float
    ) -> str:
        """
        Classify symbol .
        """
        # Ensure width <= height
        w_mm, h_mm = sorted((width_mm, height_mm))

        for spec in self.SYMBOLS:
            if (
                abs(w_mm - spec.width_mm) <= self.tol_mm
                and abs(h_mm - spec.height_mm) <= self.tol_mm
            ):
                #print(f"[CLASSIFY] width={w_mm:.2f}mm, height={h_mm:.2f}mm → {spec.name}")
                return spec.name

        #print(f"[CLASSIFY] width={w_mm:.2f}mm, height={h_mm:.2f}mm → unknown")
        return "unknown"


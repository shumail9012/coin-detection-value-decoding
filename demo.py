"""
Headless demo: run the detection -> validation -> decoding pipeline on images
without the GUI or a camera, and save annotated results.

Usage:
    python demo.py                      # processes samples/ -> results/
    python demo.py path/to/images out/  # custom input and output folders
"""
import glob
import os
import sys
from collections import Counter

import cv2
import matplotlib

matplotlib.use("Agg")  # no display needed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from object_detector import ObjectAndSymbolDetector  # noqa: E402
from object_validator import ObjectValidator  # noqa: E402
from value_decoder import ValueDecoder  # noqa: E402


def annotate(img, objects, decoder, validator):
    """Detect, validate and decode every object; draw the result on a copy of img."""
    out = img.copy()
    labels = []
    for obj in objects:
        validator.validate_object(obj)
        cx, cy = map(int, obj["centroid"])
        if obj["real_or_fake"] != "Real":
            text, colour = "FAKE", (25, 50, 255)
        else:
            symbols = obj.get("symbols", [])
            coin_colour = Counter(s["color"] for s in symbols).most_common(1)[0][0]
            value = decoder.decode_object(symbols, coin_colour, obj["bbox_rect"], img_debug=img)
            text, colour = f"{coin_colour}: {value}", (80, 255, 60)
            pts = obj["inner_area"].astype("int32") if hasattr(obj["inner_area"], "astype") else obj["inner_area"]
            cv2.polylines(out, [pts], True, colour, 6)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
        x0, y0 = cx - tw // 2, cy + th // 2
        cv2.rectangle(out, (x0 - 10, y0 - th - 10), (x0 + tw + 10, y0 + 10), (0, 0, 0), -1)
        cv2.putText(out, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 3, colour, 6)
        labels.append(text)
    return out, labels


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "samples"
    dst = sys.argv[2] if len(sys.argv) > 2 else "results"
    os.makedirs(dst, exist_ok=True)

    detector, validator, decoder = ObjectAndSymbolDetector(), ObjectValidator(), ValueDecoder()
    files = sorted(f for ext in ("png", "bmp", "jpg", "jpeg") for f in glob.glob(os.path.join(src, f"*.{ext}")))
    for path in files:
        img = cv2.imread(path)
        result = detector.detect(img, visualise=False)
        annotated, labels = annotate(img, result["objects"], decoder, validator)
        name = os.path.splitext(os.path.basename(path))[0] + "_result.jpg"
        cv2.imwrite(os.path.join(dst, name), annotated)
        print(f"{os.path.basename(path)}: {labels or 'no objects found'}")


if __name__ == "__main__":
    main()

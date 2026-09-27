import cv2
import numpy as np
import matplotlib.pyplot as plt


class ShapeDetector:
    """
    Detects all significant shapes (rectangles, circles, or irregular shapes)
    on a surface and measures their bounding boxes using a pixel-to-mm scale.
    """

    def __init__(
        self,
        pixels_per_mm_x: float,
        pixels_per_mm_y: float,
        min_area_px: int = 500,
        bg_kernel_size: int = 51
    ):
        self.px_per_mm_x = pixels_per_mm_x
        self.px_per_mm_y = pixels_per_mm_y
        self.min_area_px = min_area_px
        self.bg_kernel_size = bg_kernel_size

    # Remove green background
    def _remove_green_background(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        img_no_green = img_bgr.copy()
        img_no_green[green_mask > 0] = 0
        return img_no_green, green_mask

    # Preprocessing
    def _preprocess(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.bg_kernel_size, self.bg_kernel_size))
        background = cv2.morphologyEx(gray_blur, cv2.MORPH_OPEN, kernel)
        detail = cv2.subtract(gray_blur, background)
        # CLAHE for local histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        detail = clahe.apply(detail)

        # Gradient magnitude
        gx = cv2.Sobel(detail, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(detail, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return gray, background, detail, grad


    def _clean_binary(self, grad):
        # Use bilateral filter to reduce salt-and-pepper noise
        grad_filtered = cv2.bilateralFilter(grad, d=5, sigmaColor=75, sigmaSpace=75)

        # Otsu thresholding (better than adaptive for gradient)
        _, binary = cv2.threshold(grad_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel)

        # Remove small blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
        clean = np.zeros_like(binary_clean)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area_px:
                clean[labels == i] = 255

        return binary, clean


    # Detect all shapes
    def _detect_shapes(self, binary_clean):
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_px:
                continue

            rect = cv2.minAreaRect(cnt)
            width_mm = rect[1][0] / self.px_per_mm_x
            height_mm = rect[1][1] / self.px_per_mm_y

            shapes.append({
                "contour": cnt,
                "rect": rect,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "area": area
            })

        return shapes

    # Public API
    def detect(self, img_bgr, visualise=False):
        outputs = {}
        img_no_green, green_mask = self._remove_green_background(img_bgr)
        gray, background, detail, grad = self._preprocess(img_no_green)
        binary, binary_clean = self._clean_binary(grad)
        shapes = self._detect_shapes(binary_clean)

        outputs.update({
            "no_green": img_no_green,
            "gray": gray,
            "background": background,
            "detail": detail,
            "gradient": grad,
            "binary": binary,
            "binary_clean": binary_clean,
            "shapes": shapes
        })

        if visualise:
            self._visualise(img_bgr, outputs)
            # Display histogram of gradient magnitude
            plt.figure(figsize=(6, 4))
            plt.hist(grad.ravel(), bins=256, range=(0, 255), color='blue')
            plt.title("Histogram of Gradient Magnitude")
            plt.xlabel("Gradient Value")
            plt.ylabel("Pixel Count")
            plt.show()

            # Display green mask
            plt.figure(figsize=(6, 4))
            plt.hist(green_mask.ravel(), bins=2, range=(0, 255), color='green')
            plt.title("Green Mask")
            plt.xlabel("Mask Value")
            plt.ylabel("Pixel Count")
            plt.show()

            # Display histogram of background
            plt.figure(figsize=(6, 4))
            plt.hist(background.ravel(), bins=256, range=(0, 255), color='blue')
            plt.title("Histogram of Background")
            plt.xlabel("Background Value")
            plt.ylabel("Pixel Count")
            plt.show()

            # Display histogram of detail
            plt.figure(figsize=(6, 4))
            plt.hist(detail.ravel(), bins=256, range=(0, 255), color='blue')
            plt.title("Histogram of Detail")
            plt.xlabel("Detail Value")
            plt.ylabel("Pixel Count")
            plt.show()

            # Display histogram of binary clean
            plt.figure(figsize=(6, 4))
            plt.hist(binary_clean.ravel(), bins=2, range=(0, 255), color='black')
            plt.title("Binary Clean Mask")
            plt.xlabel("Mask Value")
            plt.ylabel("Pixel Count")
            plt.show()

        

            # Display histogram of gray
            plt.figure(figsize=(6, 4))  
            plt.hist(gray.ravel(), bins=256, range=(0, 255), color='blue')
            plt.title("Histogram of Gray")  
            plt.xlabel("Gray Value")  
            plt.ylabel("Pixel Count")   
            plt.show()  

            # Display histogram of Adaptive Threshold
            plt.figure(figsize=(6, 4))
            plt.hist(binary.ravel(), bins=2, range=(0, 255), color='black')
            plt.title("Adaptive Threshold Mask")
            plt.xlabel("Mask Value")
            plt.ylabel("Pixel Count")
            plt.show()  

        return outputs

    # Visualization
    def _visualise(self, img_bgr, out):
        images = [
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            out["no_green"],
            out["detail"],
            out["gradient"],
            out["binary"],
            out["binary_clean"]
        ]
        titles = ["Original", "Green removed", "High-pass detail", "Gradient magnitude", "Adaptive threshold", "Clean binary"]
        plt.figure(figsize=(14, 8))
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(2, 3, i + 1)
            plt.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
            plt.title(title)
            plt.axis("off")

        if out["shapes"]:
            overlay = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            for s in out["shapes"]:
                cv2.drawContours(overlay, [s["contour"]], 0, (0, 255, 0), 2)
            plt.subplot(2, 3, 6)
            plt.imshow(overlay)
            plt.title(f"Detected {len(out['shapes'])} shapes")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


# =====================
# Main runner
# =====================
if __name__ == "__main__":
    img = cv2.imread("dataset/img_20251219_125327_644554.bmp")
    detector = ShapeDetector(pixels_per_mm_x=9.0, pixels_per_mm_y=9.4)
    result = detector.detect(img, visualise=True)

    if result["shapes"]:
        for i, s in enumerate(result["shapes"], start=1):
            print(f"Shape {i}: Width={s['width_mm']:.2f} mm, Height={s['height_mm']:.2f} mm, Area={s['area']:.0f}")
    else:
        print("No shapes detected")

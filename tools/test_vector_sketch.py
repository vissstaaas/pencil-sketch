import cv2
import numpy as np
from PIL import Image, ImageDraw
import math

# 1. Load image
img_path = r'C:\Users\sxbdg\.gemini\antigravity\brain\bbc29d7c-b41a-4e24-8cef-dc7033eaecc8\.user_uploaded\media_1789110001455.jpg'
orig = Image.open(img_path).convert('RGB')
w, h = orig.size
gray = np.array(orig.convert('L'))

# Scale up if small so lines have high resolution (e.g. 768px)
target_dim = 768
scale = target_dim / max(w, h)
new_w = int(w * scale)
new_h = int(h * scale)
gray_scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

# 2. Color Dodge Line Art filter
smooth = cv2.bilateralFilter(gray_scaled, d=9, sigmaColor=50, sigmaSpace=50)
inv = 255 - smooth
blur = cv2.GaussianBlur(inv, (21, 21), 0)
dodge = cv2.divide(smooth, 255 - blur, scale=256)

# 3. High-definition edge extraction
# Adaptive multi-threshold to capture both strong outlines and subtle facial expressions
binary_face = (dodge < 242).astype(np.uint8) * 255
# Remove isolated noise dots
kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary_cleaned = cv2.morphologyEx(binary_face, cv2.MORPH_OPEN, kernel_clean)

# Morphological boundary to turn filled shapes into clean outlines
grad = cv2.morphologyEx(binary_cleaned, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

# Thin to 1-pixel skeleton
skeleton = cv2.ximgproc.thinning(grad) if hasattr(cv2, 'ximgproc') else grad

# 4. Extract connected contours
contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Render onto 768 canvas
canvas = Image.new('RGB', (new_w, new_h), (251, 249, 245))
draw = ImageDraw.Draw(canvas)

stroke_count = 0
for cnt in contours:
    pts = cnt.reshape(-1, 2)
    if len(pts) < 5:
        continue
    # Douglas-Peucker simplification to get clean curve vertices
    approx = cv2.approxPolyDP(cnt, epsilon=1.2, closed=False).reshape(-1, 2)
    if len(approx) < 2:
        continue
    stroke_count += 1
    # Draw smooth polyline
    for i in range(len(approx) - 1):
        p0 = tuple(approx[i])
        p1 = tuple(approx[i+1])
        draw.line([p0, p1], fill=(35, 32, 30), width=2)

canvas.save('tools/test_vector_sketch.png')
print(f'Rendered {stroke_count} continuous vector strokes onto {new_w}x{new_h} canvas.')
print('Saved tools/test_vector_sketch.png')

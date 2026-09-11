import cv2
import numpy as np
from PIL import Image, ImageDraw

orig = Image.open(r'C:\Users\sxbdg\.gemini\antigravity\brain\bbc29d7c-b41a-4e24-8cef-dc7033eaecc8\.user_uploaded\media_1789110001455.jpg').convert('RGB')
w, h = orig.size
target_dim = 768
scale = target_dim / max(w, h)
nw, nh = int(w * scale), int(h * scale)
gray = cv2.resize(np.array(orig.convert('L')), (nw, nh), interpolation=cv2.INTER_LANCZOS4)

# 1. Bilateral filter
smooth = cv2.bilateralFilter(gray, 7, 45, 45).astype(np.float32) / 255.0

# 2. Color Dodge
inv = 1.0 - smooth
blur = cv2.GaussianBlur(inv, (21, 21), 0)
denom = np.maximum(0.005, 1.0 - blur)
dodge = np.clip(smooth / denom, 0.0, 1.0)
edge_data = np.maximum(0.0, (1.0 - dodge - 0.03) / 0.97)

# 3. Local Feature Contrast Enhancement (captures true facial likeness)
local_mean = cv2.GaussianBlur(smooth, (15, 15), 0)
contrast = np.maximum(0.0, local_mean - smooth)
feature_gain = np.clip(contrast * 10.0, 0.0, 1.0)

# Blend
final_edge = np.maximum(edge_data, feature_gain)

# Binary for contour tracing
binary = (final_edge > 0.18).astype(np.uint8) * 255
kernel = np.ones((2, 2), np.uint8)
binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
grad = cv2.morphologyEx(binary_clean, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
skel = cv2.ximgproc.thinning(grad) if hasattr(cv2, 'ximgproc') else grad

cnts, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

canvas = Image.new('RGB', (nw, nh), (251, 249, 245))
draw = ImageDraw.Draw(canvas)

count = 0
for cnt in cnts:
    length = cv2.arcLength(cnt, False)
    if length < 10:
        continue
    pts = cnt.reshape(-1, 2)
    cx, cy = pts.mean(axis=0)
    is_face = (240 < cx < 530) and (120 < cy < 340)
    eps = 0.9 if is_face else 1.8
    approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=False).reshape(-1, 2)
    if len(approx) < 2:
        continue
    count += 1
    width = 2 if is_face else 1
    for i in range(len(approx) - 1):
        draw.line([tuple(approx[i]), tuple(approx[i+1])], fill=(35, 32, 30), width=width)

canvas.save('tools/test_full_enhanced_portrait.png')
print(f'Rendered {count} strokes to tools/test_full_enhanced_portrait.png')

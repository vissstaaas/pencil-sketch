import cv2
import numpy as np
from PIL import Image, ImageDraw

orig = Image.open(r'C:\Users\sxbdg\.gemini\antigravity\brain\bbc29d7c-b41a-4e24-8cef-dc7033eaecc8\.user_uploaded\media_1789110001455.jpg').convert('RGB')
w, h = orig.size
target_dim = 768
scale = target_dim / max(w, h)
nw, nh = int(w * scale), int(h * scale)
gray = cv2.resize(np.array(orig.convert('L')), (nw, nh), interpolation=cv2.INTER_LANCZOS4)

# Face box: x in [240, 530], y in [120, 340]
face_gray = gray[120:340, 240:530]

# 1. Bilateral filter tuned for facial skin (preserves sharp eyes/lips, removes noise)
face_smooth = cv2.bilateralFilter(face_gray, d=5, sigmaColor=35, sigmaSpace=35)

# 2. Extract multi-threshold facial features
# EYES & EYEBROWS & MOUTH:
# Adaptive threshold / local contrast captures the EXACT eye shape, iris, and smile!
adapt = cv2.adaptiveThreshold(
    face_smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6
)

# 3. Morphological cleanup
kernel = np.ones((2, 2), np.uint8)
adapt_clean = cv2.morphologyEx(255 - adapt, cv2.MORPH_OPEN, kernel)

# Skeletonize to 1px lines
skel = cv2.ximgproc.thinning(adapt_clean) if hasattr(cv2, 'ximgproc') else adapt_clean

# Extract contours
cnts, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

canvas = Image.new('RGB', (face_gray.shape[1], face_gray.shape[0]), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

for cnt in cnts:
    if cv2.arcLength(cnt, False) < 6:
        continue
    approx = cv2.approxPolyDP(cnt, epsilon=0.8, closed=False).reshape(-1, 2)
    if len(approx) < 2:
        continue
    for i in range(len(approx) - 1):
        draw.line([tuple(approx[i]), tuple(approx[i+1])], fill=(25, 22, 20), width=2)

canvas.save('tools/test_face_adaptive.png')
print('Saved test_face_adaptive.png')

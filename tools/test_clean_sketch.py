import cv2
import numpy as np
from PIL import Image, ImageDraw
import math

# 1. Load image
img_path = r'C:\Users\sxbdg\.gemini\antigravity\brain\bbc29d7c-b41a-4e24-8cef-dc7033eaecc8\.user_uploaded\media_1789110001455.jpg'
orig = Image.open(img_path).convert('RGB')
w, h = orig.size
gray = np.array(orig.convert('L'))

# 2. Color Dodge / Artistic Pencil Line Map
# Bilateral filter to smooth texture while keeping real boundaries sharp
smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)
inv = 255 - smooth
blur = cv2.GaussianBlur(inv, (15, 15), 0)
dodge = cv2.divide(smooth, 255 - blur, scale=256)

# Line strength: inverted dodge (0 = paper white, 255 = deepest line)
line_map = (255 - dodge).astype(np.float32) / 255.0

# Boost contrast of facial features and contours
line_map = np.clip((line_map - 0.12) / 0.45, 0.0, 1.0)

# Save line map for visual check
Image.fromarray((line_map * 255).astype(np.uint8)).save('tools/test_clean_linemap.png')

# 3. Extract continuous vector chains
binary = (line_map > 0.25).astype(np.uint8) * 255

# Thin to 1px skeleton using morphological thinning or distance ridge
# cv2 ximgproc or standard thinning:
# Let's find contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

# 4. Render on white canvas with nice pencil strokes
canvas = Image.new('RGB', (w, h), (251, 249, 245))
draw = ImageDraw.Draw(canvas)

total_strokes = 0
for cnt in contours:
    # cnt is (N, 1, 2)
    pts = cnt.reshape(-1, 2)
    if len(pts) < 4:
        continue
    total_strokes += 1
    # Draw as continuous polyline
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i+1]
        draw.line([(x0, y0), (x1, y1)], fill=(38, 34, 32), width=1)

canvas.save('tools/test_clean_canvas.png')
print(f'Total extracted clean contours: {total_strokes}')
print('Saved test_clean_canvas.png')

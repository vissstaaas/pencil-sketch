import cv2
import numpy as np
from PIL import Image, ImageDraw

img_path = r'C:\Users\sxbdg\.gemini\antigravity\brain\bbc29d7c-b41a-4e24-8cef-dc7033eaecc8\.user_uploaded\media_1789110001455.jpg'
orig = Image.open(img_path).convert('RGB')
w, h = orig.size
target_dim = 768
scale = target_dim / max(w, h)
nw, nh = int(w * scale), int(h * scale)
gray = cv2.resize(np.array(orig.convert('L')), (nw, nh), interpolation=cv2.INTER_LANCZOS4)

# 1. Bilateral filter + Color Dodge
smooth = cv2.bilateralFilter(gray, 9, 45, 45)
inv = 255 - smooth
blur = cv2.GaussianBlur(inv, (19, 19), 0)
dodge = cv2.divide(smooth, 255 - blur, scale=256)

# Line strength
line_map = 1.0 - (dodge.astype(np.float32) / 255.0)

# Crop to face region: y in [120, 320], x in [250, 520]
face_crop = (np.clip(dodge, 0, 255)).astype(np.uint8)
Image.fromarray(face_crop[120:320, 250:520]).save('tools/test_face_crop_original.png')
print('Face crop saved: tools/test_face_crop_original.png')

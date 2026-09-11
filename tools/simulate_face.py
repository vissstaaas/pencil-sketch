import cv2
import numpy as np
from PIL import Image, ImageDraw

# Let's inspect what happens to the face features
dodge = cv2.imread('tools/test_face_crop_original.png', 0)
h, w = dodge.shape

# Threshold for feature anchors: edge > 0.12
edge_map = (255 - dodge).astype(np.float32) / 255.0

# If we trace with the current detectFeatureAnchors logic:
# It steps greedily forward only, capped at 70px, without bidirectional expansion
visited = np.zeros_like(edge_map, dtype=bool)
chains = []

neighbors = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1)
]

for y in range(h):
    for x in range(w):
        if visited[y, x] or edge_map[y, x] < 0.15:
            continue
        
        # Greedy 1-way trace
        pts = [(x, y)]
        visited[y, x] = True
        cx, cy = x, y
        
        while len(pts) < 70:
            best_edge = -1
            bx, by = -1, -1
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    if edge_map[ny, nx] > best_edge and edge_map[ny, nx] > 0.10:
                        best_edge = edge_map[ny, nx]
                        bx, by = nx, ny
            if bx != -1:
                visited[by, bx] = True
                pts.append((bx, by))
                cx, cy = bx, by
            else:
                break
        
        if len(pts) >= 4:
            chains.append(pts)

# Draw on canvas
canvas = Image.new('RGB', (w, h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

for pts in chains:
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i+1]], fill=(30, 30, 30), width=1)

canvas.save('tools/test_face_simulation_current.png')
print(f'Current logic generated {len(chains)} chains on face.')

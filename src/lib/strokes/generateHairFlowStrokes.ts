import type { PreprocessedImage, StrokeGenerationOptions } from '../../types/image-processing';
import type { Stroke, StrokePoint } from '../../types/stroke';
import type { SeededRandom } from '../../utils/random';

/**
 * Universal Hair & Fur Flow Streamlines:
 * Generates delicate, silky, uninterrupted hair and fur curves along the smoothed Edge Tangent Flow.
 * Features:
 * - Fine pencil width (0.65-0.85px) and soft opacity to prevent blocky black blobs.
 * - Coherent downward and natural flow without zig-zag noodle distortion.
 * - Natural pressure tapering at both ends.
 */
export function generateHairFlowStrokes(
  preprocessed: PreprocessedImage,
  options: StrokeGenerationOptions,
  rng: SeededRandom
): Stroke[] {
  const { width, height, toneMap, gradientMap, foregroundMap } = preprocessed;
  const strokes: Stroke[] = [];
  let strokeIdCounter = 0;

  // Grid step for hair lines (~ 10-14px)
  const baseStep = Math.max(9, Math.min(14, Math.round(width / 60)));
  const visited = new Uint8Array(width * height);

  const markVisited = (x: number, y: number, r: number) => {
    const minX = Math.max(0, Math.floor(x - r));
    const maxX = Math.min(width - 1, Math.ceil(x + r));
    const minY = Math.max(0, Math.floor(y - r));
    const maxY = Math.min(height - 1, Math.ceil(y + r));
    const rSq = r * r;
    for (let cy = minY; cy <= maxY; cy++) {
      const row = cy * width;
      for (let cx = minX; cx <= maxX; cx++) {
        if ((cx - x) * (cx - x) + (cy - y) * (cy - y) <= rSq) {
          visited[row + cx] = 1;
        }
      }
    }
  };

  const maxHairStrokes = Math.round(options.targetStrokeCount * 0.4);

  for (let y = baseStep; y < height - baseStep; y += baseStep) {
    for (let x = baseStep; x < width - baseStep; x += baseStep) {
      if (strokes.length >= maxHairStrokes) break;

      const idx = y * width + x;
      const fg = foregroundMap?.data[idx] ?? 1.0;
      if (fg < 0.2) continue;

      const tone = toneMap.data[idx];
      // Hair and fur areas
      if (tone < 0.38 || visited[idx]) continue;

      let curX = x + rng.range(-1.5, 1.5);
      let curY = y + rng.range(-1.5, 1.5);

      const strandPoints: StrokePoint[] = [];
      const targetLength = rng.range(70, 160);
      const stepSize = 4.0;
      const maxSteps = Math.round(targetLength / stepSize);

      let prevAngle = -999;

      for (let s = 0; s < maxSteps; s++) {
        const px = Math.round(curX);
        const py = Math.round(curY);
        if (px < 2 || px >= width - 2 || py < 2 || py >= height - 2) break;

        const pIdx = py * width + px;
        const pTone = toneMap.data[pIdx];
        if (pTone < 0.22) break; // Exited into bright skin or background

        markVisited(px, py, 2.2);

        // Vector field tangent
        const gradAng = gradientMap.angle[pIdx];
        let flowAng = gradAng + Math.PI * 0.5;

        // Ensure overall downward trend for long hair
        if (Math.sin(flowAng) < 0) {
          flowAng += Math.PI;
        }

        // Smooth angular transition (prevent abrupt turns)
        if (prevAngle !== -999) {
          let diff = flowAng - prevAngle;
          while (diff > Math.PI) diff -= 2 * Math.PI;
          while (diff < -Math.PI) diff += 2 * Math.PI;
          flowAng = prevAngle + diff * 0.4;
        }
        prevAngle = flowAng;

        // Pressure tapering
        const progress = s / maxSteps;
        const taper = Math.sin(progress * Math.PI);
        const pressure = Math.max(0.15, Math.min(0.85, (0.3 + pTone * 0.45) * taper));

        strandPoints.push({
          x: Math.round(curX * 10) / 10,
          y: Math.round(curY * 10) / 10,
          pressure,
        });

        curX += Math.cos(flowAng) * stepSize;
        curY += Math.sin(flowAng) * stepSize;
      }

      if (strandPoints.length >= 6) {
        const midIdx = Math.floor(strandPoints.length / 2);
        const midPt = strandPoints[midIdx];
        const midTone = toneMap.data[Math.floor(midPt.y) * width + Math.floor(midPt.x)] || 0.65;
        const isDarkCore = midTone > 0.68;

        strokes.push({
          id: `hair-strand-${++strokeIdCounter}`,
          kind: 'hatching',
          layer: isDarkCore ? 'darkTone' : 'midTone',
          order: 0,
          points: strandPoints,
          brush: {
            width: isDarkCore ? 0.85 : 0.68, // Delicate fine pencil line
            opacity: isDarkCore ? 0.28 : 0.18, // Transparent, buildable graphite
            hardness: 0.6,
            grain: 0.0,
            jitter: 0.02,
          },
          durationMs: Math.max(60, Math.min(150, strandPoints.length * 6)),
          priority: isDarkCore ? 55 : 45,
          sourceConfidence: midTone,
        });
      }
    }
  }

  return strokes;
}

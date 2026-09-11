import type { PreprocessedImage, StrokeGenerationOptions } from '../../types/image-processing';
import type { Stroke, StrokePoint } from '../../types/stroke';
import type { SeededRandom } from '../../utils/random';

/**
 * Universal Academic Shading & Hatching Engine:
 * Generates delicate tonal volume across the face, neck, body, and background shadows:
 * 1. Pure specular highlights (tone < 0.08) are left 100% white paper.
 * 2. Subtle skin and form modeling (0.08 <= tone < 0.32) uses ultra-soft, transparent 40-degree hatching
 *    to give the face three-dimensional roundness without looking dirty.
 * 3. Midtone and deep shadows (neck, jaw, drapery) receive disciplined parallel and cross-hatching.
 */
export function generateShadingStrokes(
  preprocessed: PreprocessedImage,
  options: StrokeGenerationOptions,
  rng: SeededRandom
): Stroke[] {
  const { width, height, toneMap, foregroundMap } = preprocessed;
  const strokes: Stroke[] = [];
  let strokeIdCounter = 0;

  // Spacing proportional to resolution: ~ 15px - 22px
  const gridStep = Math.max(14, Math.min(22, Math.round(width / 45)));
  const classicalAngle = Math.PI * 0.22; // ~40 degrees

  const maxShadingStrokes = Math.round(options.targetStrokeCount * 0.55);

  for (let y = gridStep; y < height - gridStep; y += gridStep) {
    for (let x = gridStep; x < width - gridStep; x += gridStep) {
      if (strokes.length >= maxShadingStrokes) break;

      const idx = y * width + x;
      const fg = foregroundMap?.data[idx] ?? 1.0;
      if (fg < 0.2) continue;

      const tone = toneMap.data[idx];

      // Pure specular highlight protection (only the brightest highlights are left paper white)
      if (tone < 0.08) {
        continue;
      }

      // Facial skin protection:
      // Forehead, cheeks, and nose bridge should remain clean and glowing without rough hatching.
      const isCentralFace =
        x > width * 0.25 && x < width * 0.75 &&
        y > height * 0.15 && y < height * 0.55;
      if (isCentralFace && tone < 0.45) {
        continue;
      }

      const cx = x + rng.range(-2, 2);
      const cy = y + rng.range(-2, 2);

      // 1. Ultra-delicate facial & subject form modeling (0.08 <= tone < 0.32)
      if (tone < 0.32) {
        const strokeLen = rng.range(26, 48);
        const points = createSmoothLinePoints(
          cx,
          cy,
          classicalAngle,
          strokeLen,
          rng,
          0.25
        );

        if (points.length >= 2) {
          strokes.push({
            id: `shade-skin-${++strokeIdCounter}`,
            kind: 'hatching',
            layer: 'lightTone',
            order: 0,
            points,
            brush: {
              width: 0.7,
              opacity: 0.065, // Feather-light graphite touch for soft 3D facial volume
              hardness: 0.35,
              grain: 0.0,
              jitter: 0.02,
            },
            durationMs: 65,
            priority: 40,
            sourceConfidence: tone,
          });
        }
      }
      // 2. Medium tone volume (0.32 <= tone < 0.65)
      else if (tone < 0.65) {
        const strokeLen = rng.range(24, 44);
        const points = createSmoothLinePoints(
          cx,
          cy,
          classicalAngle,
          strokeLen,
          rng,
          0.4
        );

        if (points.length >= 2) {
          strokes.push({
            id: `shade-mid-${++strokeIdCounter}`,
            kind: 'hatching',
            layer: 'midTone',
            order: 0,
            points,
            brush: {
              width: 0.9,
              opacity: 0.15,
              hardness: 0.45,
              grain: 0.0,
              jitter: 0.02,
            },
            durationMs: 75,
            priority: 48,
            sourceConfidence: tone,
          });
        }
      }
      // 3. Deep shadow cross-hatching (tone >= 0.65)
      else {
        const strokeLen = rng.range(22, 38);
        const p1 = createSmoothLinePoints(
          cx,
          cy,
          classicalAngle,
          strokeLen,
          rng,
          0.55
        );

        if (p1.length >= 2) {
          strokes.push({
            id: `shade-dark-p1-${++strokeIdCounter}`,
            kind: 'hatching',
            layer: 'darkTone',
            order: 0,
            points: p1,
            brush: {
              width: 1.0,
              opacity: 0.22,
              hardness: 0.55,
              grain: 0.0,
              jitter: 0.02,
            },
            durationMs: 80,
            priority: 55,
            sourceConfidence: tone,
          });
        }

        // Cross-hatching angle ~ 130 deg
        const crossAngle = classicalAngle + Math.PI * 0.45;
        const p2 = createSmoothLinePoints(
          cx,
          cy,
          crossAngle,
          strokeLen * 0.75,
          rng,
          0.6
        );

        if (p2.length >= 2) {
          strokes.push({
            id: `shade-dark-p2-${++strokeIdCounter}`,
            kind: 'crossHatching',
            layer: 'darkTone',
            order: 0,
            points: p2,
            brush: {
              width: 1.1,
              opacity: 0.26,
              hardness: 0.6,
              grain: 0.0,
              jitter: 0.02,
            },
            durationMs: 85,
            priority: 60,
            sourceConfidence: tone,
          });
        }
      }
    }
  }

  return strokes;
}

/**
 * Creates 5 smooth points along a line segment with subtle natural pressure tapering.
 */
function createSmoothLinePoints(
  centerX: number,
  centerY: number,
  angle: number,
  length: number,
  rng: SeededRandom,
  basePressure: number
): StrokePoint[] {
  const halfLen = length * 0.5;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  const startX = centerX - cos * halfLen;
  const startY = centerY - sin * halfLen;
  const endX = centerX + cos * halfLen;
  const endY = centerY + sin * halfLen;

  const p1X = centerX - cos * (halfLen * 0.5);
  const p1Y = centerY - sin * (halfLen * 0.5);
  const p3X = centerX + cos * (halfLen * 0.5);
  const p3Y = centerY + sin * (halfLen * 0.5);

  const jitterX = rng.range(-0.12, 0.12);
  const jitterY = rng.range(-0.12, 0.12);

  return [
    { x: Math.round(startX), y: Math.round(startY), pressure: basePressure * 0.35 },
    { x: Math.round(p1X), y: Math.round(p1Y), pressure: basePressure * 0.8 },
    { x: Math.round(centerX + jitterX), y: Math.round(centerY + jitterY), pressure: basePressure },
    { x: Math.round(p3X), y: Math.round(p3Y), pressure: basePressure * 0.8 },
    { x: Math.round(endX), y: Math.round(endY), pressure: basePressure * 0.4 },
  ];
}

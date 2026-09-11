import type { PreprocessedImage, StrokeGenerationOptions } from '../../types/image-processing';
import type { Stroke, StrokePoint, StrokeKind, StrokeLayer } from '../../types/stroke';
import type { SeededRandom } from '../../utils/random';
import { simplifyPoints, resamplePoints } from './pathGeometry';

/**
 * Universal Focal Feature Extractor:
 * Traces delicate, complete facial and focal feature contours (eyes, eyebrows, pupils, nostrils, lips)
 * directly from the image's line response map with bidirectional continuity.
 * All strokes are fine, sharp, and elegant (0.75-0.95px) with zero jitter, perfectly capturing likeness.
 */
export function detectFeatureAnchors(
  preprocessed: PreprocessedImage,
  options: StrokeGenerationOptions,
  _rng: SeededRandom,
  claimed?: Uint8Array
): Stroke[] {
  const { width, height, edgeMap, toneMap, saliencyMap, foregroundMap } = preprocessed;
  const strokes: Stroke[] = [];
  let strokeIdCounter = 0;

  // Search in the focal zone: central 65% width, 15% to 75% height
  const minX = Math.floor(width * 0.18);
  const maxX = Math.floor(width * 0.82);
  const minY = Math.floor(height * 0.12);
  const maxY = Math.floor(height * 0.75);

  const visited = new Uint8Array(width * height);
  const featureThreshold = 0.12 * (1.2 - options.contourSensitivity * 0.3);

  const neighbors = [
    [1, 0], [1, 1], [0, 1], [-1, 1],
    [-1, 0], [-1, -1], [0, -1], [1, -1],
  ];

  // Helper to trace one direction from a point
  const traceBranch = (startX: number, startY: number, maxSteps = 55): StrokePoint[] => {
    const branch: StrokePoint[] = [];
    let currX = startX;
    let currY = startY;

    for (let s = 0; s < maxSteps; s++) {
      let bestNextX = -1;
      let bestNextY = -1;
      let bestEdge = -1;

      for (const [dx, dy] of neighbors) {
        const nx = currX + dx;
        const ny = currY + dy;
        if (nx < minX || nx >= maxX || ny < minY || ny >= maxY) continue;

        const nIdx = ny * width + nx;
        if (visited[nIdx]) continue;

        const nEdge = edgeMap.data[nIdx];
        if (nEdge > featureThreshold * 0.6 && nEdge > bestEdge) {
          bestEdge = nEdge;
          bestNextX = nx;
          bestNextY = ny;
        }
      }

      if (bestNextX !== -1) {
        visited[bestNextY * width + bestNextX] = 1;
        const pTone = toneMap.data[bestNextY * width + bestNextX];
        branch.push({
          x: bestNextX,
          y: bestNextY,
          pressure: Math.min(0.85, 0.40 + bestEdge * 0.35 + pTone * 0.2),
        });
        currX = bestNextX;
        currY = bestNextY;
      } else {
        break;
      }
    }
    return branch;
  };

  for (let y = minY; y < maxY; y++) {
    for (let x = minX; x < maxX; x++) {
      const idx = y * width + x;
      const fg = foregroundMap?.data[idx] ?? 1.0;
      if (fg < 0.25 || visited[idx] || (claimed && claimed[idx])) continue;

      const edge = edgeMap.data[idx];
      const saliency = saliencyMap.data[idx];
      const tone = toneMap.data[idx];

      // High edge response in focal region or dark feature (eyebrow, pupil, lip)
      if (edge > featureThreshold && (saliency > 0.32 || tone > 0.38)) {
        visited[idx] = 1;

        const seedPoint: StrokePoint = {
          x,
          y,
          pressure: Math.min(0.85, 0.40 + edge * 0.35 + tone * 0.2),
        };

        // Bidirectional tracing: forward & backward branches
        const forward = traceBranch(x, y, 50);
        const backward = traceBranch(x, y, 50);

        const rawPoints: StrokePoint[] = [
          ...backward.reverse(),
          seedPoint,
          ...forward,
        ];

        if (rawPoints.length >= 4) {
          const simplified = simplifyPoints(rawPoints, 1.1);
          const resampled = resamplePoints(simplified, 2.2);
          if (resampled.length < 2) continue;

          const midIdx = Math.floor(resampled.length / 2);
          const midPt = resampled[midIdx];
          const midMapIdx = Math.floor(midPt.y) * width + Math.floor(midPt.x);
          const localTone = toneMap.data[midMapIdx] || 0.5;

          const isDarkAccent = localTone > 0.68;
          const kind: StrokeKind = isDarkAccent ? 'accent' : 'feature';
          const layer: StrokeLayer = isDarkAccent ? 'finalAccent' : 'features';
          const priority = isDarkAccent ? 80 : 35;

          // Gentle pressure tapering at endpoints
          resampled[0].pressure *= 0.6;
          resampled[resampled.length - 1].pressure *= 0.6;

          // Claim these pixels so general contour generator won't double-trace
          if (claimed) {
            for (const pt of resampled) {
              const px = Math.round(pt.x);
              const py = Math.round(pt.y);
              if (px >= 0 && px < width && py >= 0 && py < height) {
                claimed[py * width + px] = 1;
                // Claim immediate 3x3 neighborhood
                for (let dy = -1; dy <= 1; dy++) {
                  for (let dx = -1; dx <= 1; dx++) {
                    const cIdx = (py + dy) * width + (px + dx);
                    if (cIdx >= 0 && cIdx < width * height) {
                      claimed[cIdx] = 1;
                    }
                  }
                }
              }
            }
          }

          strokes.push({
            id: `feat-${++strokeIdCounter}`,
            kind,
            layer,
            order: 0,
            points: resampled,
            brush: {
              width: isDarkAccent ? 0.95 : 0.75, // Silky, delicate pencil
              opacity: isDarkAccent ? 0.75 : 0.55, // Transparent, clean graphite
              hardness: 0.85,
              grain: 0.0,
              jitter: 0.0, // Zero jitter for pure smooth facial lines
            },
            durationMs: Math.max(50, Math.min(130, resampled.length * 6)),
            priority,
            sourceConfidence: edge,
          });
        }
      }
    }
  }

  return strokes;
}

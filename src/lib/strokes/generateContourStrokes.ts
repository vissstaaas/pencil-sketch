import type { PreprocessedImage, StrokeGenerationOptions } from '../../types/image-processing';
import type { Stroke, StrokePoint, StrokeKind, StrokeLayer } from '../../types/stroke';
import type { SeededRandom } from '../../utils/random';

function simplifyPath(points: StrokePoint[], epsilon: number): StrokePoint[] {
  if (points.length < 3) return points;
  const sqEpsilon = epsilon * epsilon;

  function pointLineDistanceSq(p: StrokePoint, p1: StrokePoint, p2: StrokePoint) {
    let x = p1.x, y = p1.y;
    let dx = p2.x - x, dy = p2.y - y;
    if (dx !== 0 || dy !== 0) {
      const t = ((p.x - x) * dx + (p.y - y) * dy) / (dx * dx + dy * dy);
      if (t > 1) {
        x = p2.x; y = p2.y;
      } else if (t > 0) {
        x += dx * t; y += dy * t;
      }
    }
    dx = p.x - x; dy = p.y - y;
    return dx * dx + dy * dy;
  }

  function simplifyDPStep(points: StrokePoint[], first: number, last: number, sqTolerance: number, simplified: StrokePoint[]) {
    let maxSqDist = sqTolerance;
    let index = -1;
    for (let i = first + 1; i < last; i++) {
      const sqDist = pointLineDistanceSq(points[i], points[first], points[last]);
      if (sqDist > maxSqDist) {
        index = i;
        maxSqDist = sqDist;
      }
    }
    if (maxSqDist > sqTolerance) {
      if (index - first > 1) simplifyDPStep(points, first, index, sqTolerance, simplified);
      simplified.push(points[index]);
      if (last - index > 1) simplifyDPStep(points, index, last, sqTolerance, simplified);
    }
  }

  const simplified: StrokePoint[] = [points[0]];
  simplifyDPStep(points, 0, points.length - 1, sqEpsilon, simplified);
  simplified.push(points[points.length - 1]);
  return simplified;
}

function catmullRomResample(points: StrokePoint[], spacing: number): StrokePoint[] {
  if (points.length < 2) return points;
  const result: StrokePoint[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];
    const segLen = Math.hypot(p2.x - p1.x, p2.y - p1.y);
    const steps = Math.max(1, Math.round(segLen / spacing));
    for (let s = 0; s < steps; s++) {
      const t = s / steps;
      const t2 = t * t;
      const t3 = t2 * t;
      const x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3);
      const y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3);
      const pressure = 0.5 * ((2 * p1.pressure) + (-p0.pressure + p2.pressure) * t + (2 * p0.pressure - 5 * p1.pressure + 4 * p2.pressure - p3.pressure) * t2 + (-p0.pressure + 3 * p1.pressure - 3 * p2.pressure + p3.pressure) * t3);
      result.push({ x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10, pressure: Math.max(0.15, Math.min(1, pressure)) });
    }
  }
  result.push(points[points.length - 1]);
  return result;
}

export function generateContourStrokes(
  preprocessed: PreprocessedImage,
  options: StrokeGenerationOptions,
  rng: SeededRandom,
  claimed?: Uint8Array
): Stroke[] {
  const { width, height, edgeMap, saliencyMap, toneMap, foregroundMap } = preprocessed;

  // Sensitive edge threshold to catch subtle features (eyes, nose, jawline, clothing folds)
  const edgeThreshold = 0.065 * (1.2 - options.contourSensitivity * 0.3);
  const visited = new Uint8Array(width * height);

  const getEffectiveEdge = (x: number, y: number): number => {
    const idx = y * width + x;
    let base = edgeMap.data[idx];
    // If near foreground boundary, boost slightly so white clothes and light arms are not lost
    if (foregroundMap && x > 1 && x < width - 2 && y > 1 && y < height - 2) {
      const fgDiff =
        Math.abs(foregroundMap.data[idx + 1] - foregroundMap.data[idx - 1]) +
        Math.abs(foregroundMap.data[idx + width] - foregroundMap.data[idx - width]);
      if (fgDiff > 0.2) {
        base = Math.max(base, 0.25);
      }
    }
    return base;
  };

  const isEdge = (x: number, y: number) => {
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return false;
    const idx = y * width + x;
    const fg = foregroundMap?.data[idx] ?? 1.0;
    if (fg < 0.12) return false;
    if (claimed && claimed[idx]) return false;
    return getEffectiveEdge(x, y) > edgeThreshold && !visited[idx];
  };

  const getPressure = (x: number, y: number) => {
    const idx = y * width + x;
    const val = getEffectiveEdge(x, y);
    const tone = toneMap.data[idx];
    return Math.min(0.85, Math.max(0.25, 0.3 + val * 0.4 + tone * 0.15));
  };

  const neighbors = [
    [1, 0], [1, 1], [0, 1], [-1, 1],
    [-1, 0], [-1, -1], [0, -1], [1, -1],
  ];

  const rawChains: { points: StrokePoint[]; length: number; avgEdge: number }[] = [];

  // Extract connected edge chains
  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      if (!isEdge(x, y)) continue;

      const idx = y * width + x;
      visited[idx] = 1;

      const forward: StrokePoint[] = [{ x, y, pressure: getPressure(x, y) }];
      const backward: StrokePoint[] = [];

      // Forward trace
      let currX = x;
      let currY = y;
      let tracing = true;
      while (tracing && forward.length < 250) {
        tracing = false;
        for (const [dx, dy] of neighbors) {
          const nx = currX + dx;
          const ny = currY + dy;
          if (isEdge(nx, ny)) {
            visited[ny * width + nx] = 1;
            forward.push({ x: nx, y: ny, pressure: getPressure(nx, ny) });
            currX = nx;
            currY = ny;
            tracing = true;
            break;
          }
        }
      }

      // Backward trace
      currX = x;
      currY = y;
      tracing = true;
      while (tracing && backward.length < 250) {
        tracing = false;
        for (const [dx, dy] of neighbors) {
          const nx = currX + dx;
          const ny = currY + dy;
          if (isEdge(nx, ny)) {
            visited[ny * width + nx] = 1;
            backward.push({ x: nx, y: ny, pressure: getPressure(nx, ny) });
            currX = nx;
            currY = ny;
            tracing = true;
            break;
          }
        }
      }

      const chain: StrokePoint[] = [];
      for (let i = backward.length - 1; i >= 0; i--) chain.push(backward[i]);
      for (let i = 0; i < forward.length; i++) chain.push(forward[i]);

      let length = 0;
      let edgeSum = 0;
      let darkCount = 0;

      for (let i = 1; i < chain.length; i++) {
        length += Math.hypot(chain[i].x - chain[i - 1].x, chain[i].y - chain[i - 1].y);
        const pIdx = Math.floor(chain[i].y) * width + Math.floor(chain[i].x);
        edgeSum += getEffectiveEdge(chain[i].x, chain[i].y);
        if (toneMap.data[pIdx] > 0.75) {
          darkCount++;
        }
      }

      // Suppress small closed loops inside solid dark hair (prevents topographic camouflage lines)
      const isInternalHairLoop = length < 60 && (darkCount / Math.max(1, chain.length)) > 0.75;
      if (isInternalHairLoop) continue;

      // Keep lines with length >= 8px so facial features and clothing contours are NOT lost
      if (length >= 8) {
        rawChains.push({
          points: chain,
          length,
          avgEdge: chain.length > 1 ? edgeSum / (chain.length - 1) : 0.5,
        });
      }
    }
  }

  // Sort: longest outer silhouettes first
  rawChains.sort((a, b) => b.length - a.length);

  const strokes: Stroke[] = [];
  let strokeIdCounter = 0;

  // Add light preliminary construction strokes (initial 2H pencil layout)
  const numStructureStrokes = Math.min(8, Math.floor(rawChains.length * 0.08));
  for (let i = 0; i < numStructureStrokes; i++) {
    const chain = rawChains[i];
    const simplified = simplifyPath(chain.points, 5.0);
    const resampled = catmullRomResample(simplified, 7.0);
    if (resampled.length >= 2) {
      strokes.push({
        id: `struct-${++strokeIdCounter}`,
        kind: 'construction',
        layer: 'structure',
        order: 0,
        points: resampled.map((p) => ({ ...p, pressure: 0.15 })),
        brush: {
          width: 0.75,
          opacity: 0.12,
          hardness: 0.4,
          grain: 0.0,
          jitter: 0.02,
        },
        durationMs: 70,
        priority: 5,
        sourceConfidence: 0.8,
      });
    }
  }

  // Process all valid chains
  for (const item of rawChains) {
    const midInitial = item.points[Math.floor(item.points.length / 2)];
    const isFaceZone =
      midInitial.x > width * 0.24 && midInitial.x < width * 0.76 &&
      midInitial.y > height * 0.14 && midInitial.y < height * 0.58;

    const eps = isFaceZone ? 1.2 : 2.2;
    const spacing = isFaceZone ? 2.4 : 3.6;
    const simplified = simplifyPath(item.points, eps);
    const resampled = catmullRomResample(simplified, spacing);

    if (resampled.length < 2) continue;

    // Taper ends
    const taperCount = Math.min(isFaceZone ? 3 : 5, Math.floor(resampled.length / 3));
    for (let i = 0; i < resampled.length; i++) {
      let taper = 1.0;
      if (i < taperCount) taper = (i + 1) / taperCount;
      else if (i > resampled.length - 1 - taperCount) taper = (resampled.length - i) / taperCount;
      resampled[i].pressure = Math.max(0.2, resampled[i].pressure * taper);
    }

    const midIdx = Math.floor(resampled.length / 2);
    const midPt = resampled[midIdx];
    const mapIdx = Math.floor(midPt.y) * width + Math.floor(midPt.x);

    const midTone = toneMap.data[mapIdx] || 0.5;
    const midSaliency = saliencyMap.data[mapIdx] || 0.5;
    const midEdge = edgeMap.data[mapIdx] || 0.5;

    let kind: StrokeKind = 'contour';
    let layer: StrokeLayer = 'outline';
    let priority = 20;

    if (midTone > 0.80 && midSaliency > 0.55) {
      kind = 'accent';
      layer = 'finalAccent';
      priority = 75;
    } else if (midSaliency > 0.45 && midEdge > 0.22) {
      kind = 'feature';
      layer = 'features';
      priority = 35;
    }

    // Delicate line width: 0.72 - 0.85px (fine sharpened pencil!)
    const widthVal = isFaceZone ? 0.80 : (0.75 + rng.range(-0.06, 0.08));
    const opacityVal = isFaceZone ? 0.45 : (0.38 + rng.range(-0.04, 0.08));

    strokes.push({
      id: `contour-${++strokeIdCounter}`,
      kind,
      layer,
      order: 0,
      points: resampled,
      brush: {
        width: widthVal,
        opacity: opacityVal,
        hardness: isFaceZone ? 0.8 : 0.7,
        grain: 0.0,
        jitter: isFaceZone ? 0.0 : 0.015,
      },
      durationMs: Math.max(50, Math.min(160, resampled.length * 6)),
      priority,
      sourceConfidence: item.avgEdge,
    });
  }

  return strokes;
}

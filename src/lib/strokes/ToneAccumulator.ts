import type { StrokePoint } from '../../types/stroke';
import { strokeWidthAt } from '../canvas/brushMath';

/**
 * Tonal planning buffer.
 *
 * The old generator decided "this grid cell is dark, emit one hatch line" and
 * never asked how dark the *drawing* would actually become. That is why deep
 * shadows rendered at ~0.24 darkness against a 0.82 target: the ink simply was
 * not there. This accumulator mirrors the brush compositing maths so generators
 * can measure the tone they have already laid down and keep adding strokes only
 * where the paper is still too light.
 */
export class ToneAccumulator {
  readonly width: number;
  readonly height: number;
  readonly target: Float32Array;
  /** Ink reached so far (source-over accumulation, identical maths to the renderer). */
  readonly achieved: Float32Array;
  /** How much ink this accumulator may still spend, in stroke count. */
  budget: number;
  strokesApplied = 0;

  constructor(width: number, height: number, target: Float32Array, budget = Infinity) {
    this.width = width;
    this.height = height;
    this.target = target;
    this.achieved = new Float32Array(width * height);
    this.budget = budget;
  }

  deficitAt(x: number, y: number): number {
    const px = Math.round(x);
    const py = Math.round(y);
    if (px < 0 || py < 0 || px >= this.width || py >= this.height) return 0;
    const index = py * this.width + px;
    const deficit = this.target[index] - this.achieved[index];
    return deficit > 0 ? deficit : 0;
  }

  /** Average remaining deficit over a small neighbourhood, clipped at 1. */
  meanDeficitAround(x: number, y: number, radius = 2): number {
    let sum = 0;
    let count = 0;
    const minX = Math.max(0, Math.round(x) - radius);
    const maxX = Math.min(this.width - 1, Math.round(x) + radius);
    const minY = Math.max(0, Math.round(y) - radius);
    const maxY = Math.min(this.height - 1, Math.round(y) + radius);
    for (let py = minY; py <= maxY; py++) {
      const row = py * this.width;
      for (let px = minX; px <= maxX; px++) {
        const deficit = this.target[row + px] - this.achieved[row + px];
        sum += deficit > 0 ? deficit : 0;
        count++;
      }
    }
    return count > 0 ? sum / count : 0;
  }

  get remainingBudget(): number {
    return this.budget - this.strokesApplied;
  }

  /**
   * Record one stroke. Coverage is modelled as a capsule per segment with a
   * 1px anti-aliased edge, exactly like the ribbon brush does when rendering.
   */
  applyStroke(points: StrokePoint[], brushWidth: number, opacity: number, efficiency = 1): void {
    if (points.length === 0 || opacity <= 0) return;
    const halfAt = (pressure: number) => Math.max(0.2, (strokeWidthAt(brushWidth, pressure) * efficiency) / 2);
    if (points.length === 1) {
      const half = halfAt(points[0].pressure ?? 0.6);
      this.applySegment(points[0].x, points[0].y, points[0].x, points[0].y, half, opacity);
    } else {
      for (let i = 0; i < points.length - 1; i++) {
        const pressure = ((points[i].pressure ?? 0.6) + (points[i + 1].pressure ?? 0.6)) / 2;
        this.applySegment(points[i].x, points[i].y, points[i + 1].x, points[i + 1].y, halfAt(pressure), opacity);
      }
    }
    this.strokesApplied++;
  }

  /** Overall completion: 1 means every pixel reached its target tone. */
  progress(): number {
    let targetSum = 0;
    let achievedSum = 0;
    for (let i = 0; i < this.achieved.length; i++) {
      targetSum += this.target[i];
      achievedSum += Math.min(this.achieved[i], this.target[i]);
    }
    return targetSum > 0 ? achievedSum / targetSum : 1;
  }

  private applySegment(x0: number, y0: number, x1: number, y1: number, half: number, opacity: number): void {
    const minX = Math.max(0, Math.floor(Math.min(x0, x1) - half - 1));
    const maxX = Math.min(this.width - 1, Math.ceil(Math.max(x0, x1) + half + 1));
    const minY = Math.max(0, Math.floor(Math.min(y0, y1) - half - 1));
    const maxY = Math.min(this.height - 1, Math.ceil(Math.max(y0, y1) + half + 1));
    if (minX > maxX || minY > maxY) return;

    const dx = x1 - x0;
    const dy = y1 - y0;
    const lengthSq = dx * dx + dy * dy;

    for (let py = minY; py <= maxY; py++) {
      const cy = py + 0.5;
      const row = py * this.width;
      for (let px = minX; px <= maxX; px++) {
        const cx = px + 0.5;
        let t = lengthSq > 0 ? ((cx - x0) * dx + (cy - y0) * dy) / lengthSq : 0;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const distance = Math.hypot(cx - (x0 + dx * t), cy - (y0 + dy * t));
        let coverage = half + 0.5 - distance;
        if (coverage <= 0) continue;
        if (coverage > 1) coverage = 1;
        const index = row + px;
        const alpha = opacity * coverage;
        this.achieved[index] += (1 - this.achieved[index]) * alpha;
      }
    }
  }
}

import { registerHooks } from 'node:module';
import { readFileSync } from 'node:fs';
import { basename } from 'node:path';

registerHooks({ resolve(s, c, n) { if (s.startsWith('.') && !/\.(ts|js|mjs|json)$/.test(s)) { try { return n(s + '.ts', c); } catch {} } return n(s, c); } });

const { decodePng, resizeRaster } = await import('./harness/png.ts');
const { FakeCanvas } = await import('./harness/canvas2d.ts');
const { preprocessImage } = await import('../src/lib/image/preprocessImage.ts');
const { generateContourStrokes } = await import('../src/lib/strokes/generateContourStrokes.ts');
const { detectFeatureAnchors } = await import('../src/lib/strokes/detectFeatureAnchors.ts');
const { generateHairFlowStrokes } = await import('../src/lib/strokes/generateHairFlowStrokes.ts');
const { generateShadingStrokes } = await import('../src/lib/strokes/generateShadingStrokes.ts');
const { orderStrokes } = await import('../src/lib/strokes/orderStrokes.ts');
const { ToneAccumulator } = await import('../src/lib/strokes/ToneAccumulator.ts');
const { SketchCanvasEngine } = await import('../src/lib/canvas/SketchCanvasEngine.ts');
const { SeededRandom } = await import('../src/utils/random.ts');

const SAMPLE = process.argv[2] || 'samples/pasted-2.png';
const MAX_DIM = Number(process.env.SKETCH_MAX_DIM || 768);
const SEED = Number(process.env.SKETCH_SEED || 12345);
// Mirrors useStrokeGeneration.DEFAULT_OPTIONS
const OPTIONS = {
  maxProcessingSize: MAX_DIM, targetStrokeCount: 8000, contourSensitivity: 0.5,
  shadingDensity: 0.6, backgroundSuppression: 0.5, minStrokeLength: 4, maxStrokeLength: 120, speedPreset: 'normal',
};

const pct = (n, d) => ((n / Math.max(1, d)) * 100).toFixed(1) + '%';
const source = resizeRaster(decodePng(readFileSync(SAMPLE)), MAX_DIM);
let t = Date.now();
const pre = preprocessImage({ width: source.width, height: source.height, data: source.data });
const preprocessMs = Date.now() - t;
const { width: W, height: H, edgeMap, toneMap, foregroundMap, flowMap } = pre;
console.log('=== INPUT ===', basename(SAMPLE), W + 'x' + H, ' preprocess=' + preprocessMs + 'ms');

// A. line map
const EDGE_THRESHOLD = 0.08 * (1.3 - OPTIONS.contourSensitivity * 0.3);
let fgPixels = 0, bgPixels = 0, edgeInFg = 0, edgeInBg = 0, edgeAll = 0;
for (let i = 0; i < W * H; i++) {
  const isFg = (foregroundMap.data[i] ?? 1) >= 0.15;
  const isEdge = edgeMap.data[i] > EDGE_THRESHOLD;
  if (isEdge) edgeAll++;
  if (isFg) { fgPixels++; if (isEdge) edgeInFg++; } else { bgPixels++; if (isEdge) edgeInBg++; }
}
console.log('\n=== A. LINE MAP vs the new subject mask ===');
console.log('  subject mask covers ' + pct(fgPixels, W * H) + ' of the frame (background ' + pct(bgPixels, W * H) + ')');
console.log('  line pixels inside subject:    ' + pct(edgeInFg, fgPixels) + '   (clean line art: 2-6%)');
console.log('  line pixels inside background: ' + pct(edgeInBg, bgPixels) + '   (must be near 0)');
console.log('  total line pixels: ' + pct(edgeAll, W * H));

// B. tone distribution
const bands = [
  { name: 'highlight   (<0.08)', lo: 0, hi: 0.08, n: 0 },
  { name: 'light skin  (0.08-0.32)', lo: 0.08, hi: 0.32, n: 0 },
  { name: 'mid volume  (0.32-0.65)', lo: 0.32, hi: 0.65, n: 0 },
  { name: 'deep shadow (>0.65)', lo: 0.65, hi: 1.01, n: 0 },
];
for (let i = 0; i < W * H; i++) {
  if ((foregroundMap.data[i] ?? 1) < 0.2) continue;
  const tone = toneMap.data[i];
  for (const band of bands) if (tone >= band.lo && tone < band.hi) { band.n++; break; }
}
const fgTotal = bands.reduce((s, b) => s + b.n, 0);
console.log('\n=== B. SUBJECT TONE DISTRIBUTION ===');
for (const band of bands) console.log('  ' + band.name.padEnd(24) + pct(band.n, fgTotal).padStart(6));

// C. generation, exactly like the worker
const rng = new SeededRandom(SEED);
const accumulator = new ToneAccumulator(W, H, toneMap.data, OPTIONS.targetStrokeCount);
const claimed = new Uint8Array(W * H);
const budget = { remaining: OPTIONS.targetStrokeCount };
const context = { accumulator, claimed, budget };
t = Date.now();
const featureStrokes = detectFeatureAnchors(pre, OPTIONS, rng, context);
const contourStrokes = generateContourStrokes(pre, OPTIONS, rng, context);
const hairStrokes = generateHairFlowStrokes(pre, OPTIONS, rng, context);
const shadingStrokes = generateShadingStrokes(pre, OPTIONS, rng, context);
const generateMs = Date.now() - t;
const raw = [...contourStrokes, ...featureStrokes, ...hairStrokes, ...shadingStrokes];
const ordered = orderStrokes(raw, OPTIONS.targetStrokeCount);
console.log('\n=== C. STROKE BUDGET (' + generateMs + 'ms of generation) ===');
console.log('  contours=' + contourStrokes.length + ' features=' + featureStrokes.length +
  ' hair=' + hairStrokes.length + ' shading=' + shadingStrokes.length + ' total=' + raw.length + ' kept=' + ordered.length);
const byLayer = {};
for (const s of shadingStrokes) byLayer[s.layer] = (byLayer[s.layer] || 0) + 1;
console.log('  shading by band: light=' + (byLayer.lightTone || 0) + ' mid=' + (byLayer.midTone || 0) + ' dark=' + (byLayer.darkTone || 0));
console.log('  planner progress (ink delivered / ink required): ' + (accumulator.progress() * 100).toFixed(1) + '%');

// D. path coherence
function pathStats(strokes) {
  const turns = [];
  let totalLength = 0, zigzag = 0, rightAngle = 0, samples = 0;
  for (const s of strokes) {
    const p = s.points;
    for (let i = 1; i < p.length; i++) totalLength += Math.hypot(p[i].x - p[i - 1].x, p[i].y - p[i - 1].y);
    for (let i = 2; i < p.length; i++) {
      const a1 = Math.atan2(p[i - 1].y - p[i - 2].y, p[i - 1].x - p[i - 2].x);
      const a2 = Math.atan2(p[i].y - p[i - 1].y, p[i].x - p[i - 1].x);
      let d = Math.abs(a2 - a1); if (d > Math.PI) d = 2 * Math.PI - d;
      turns.push(d); samples++;
      if (d > Math.PI / 4) zigzag++;
      if (d > Math.PI / 2) rightAngle++;
    }
  }
  turns.sort((x, y) => x - y);
  return { count: strokes.length, meanLength: totalLength / Math.max(1, strokes.length), medianTurn: turns.length ? turns[Math.floor(turns.length / 2)] * 180 / Math.PI : 0, pctOver45: 100 * zigzag / Math.max(1, samples), pctOver90: 100 * rightAngle / Math.max(1, samples) };
}
console.log('\n=== D. PATH COHERENCE (confident pencil line: < 15 deg per vertex) ===');
for (const [label, list] of [['contours', contourStrokes], ['features', featureStrokes], ['hair', hairStrokes], ['shading', shadingStrokes]]) {
  const st = pathStats(list);
  console.log('  ' + label.padEnd(9) + ' n=' + String(st.count).padStart(5) + '  avg length=' + st.meanLength.toFixed(1).padStart(6) + 'px  median turn=' + st.medianTurn.toFixed(1).padStart(5) + ' deg  >45deg ' + st.pctOver45.toFixed(1).padStart(4) + '%  >90deg ' + st.pctOver90.toFixed(1).padStart(4) + '%');
}

// E. hatching vs form (measured against the real structure tensor)
let alignSum = 0, alignWeight = 0;
for (const s of shadingStrokes) {
  for (let i = 1; i < s.points.length; i++) {
    const ax = s.points[i].x - s.points[i - 1].x;
    const ay = s.points[i].y - s.points[i - 1].y;
    const len = Math.hypot(ax, ay); if (len < 0.01) continue;
    const mx = Math.round((s.points[i].x + s.points[i - 1].x) / 2);
    const my = Math.round((s.points[i].y + s.points[i - 1].y) / 2);
    if (mx < 0 || my < 0 || mx >= W || my >= H) continue;
    const flow = flowMap.angle[my * W + mx];
    alignSum += Math.abs(Math.cos(Math.atan2(ay, ax) - flow));
    alignWeight++;
  }
}
console.log('\n=== E. HATCHING vs SURFACE DIRECTION ===');
console.log('  mean |cos| between each hatch segment and the local form tangent: ' + (alignSum / Math.max(1, alignWeight)).toFixed(3));
console.log('  (a fixed 40 degree grid scored 0.635 = random; form following approaches 1)');

// F. rendered vs target, plus planner model accuracy
function render(strokes) {
  const canvas = new FakeCanvas();
  const engine = new SketchCanvasEngine({ canvas, width: W, height: H, pixelRatio: 1, background: '#fbf9f5' });
  engine.loadScript({
    version: '1.0',
    sourceImage: { width: W, height: H, processedWidth: W, processedHeight: H, mimeType: 'image/png' },
    canvas: { width: W, height: H, background: '#fbf9f5', padding: 0 },
    generation: { seed: SEED, createdAt: new Date().toISOString(), options: OPTIONS, strokeCount: strokes.length },
    strokes: strokes.map((s, i) => ({ ...s, order: i })),
  });
  if (strokes.length) engine.drawUntil(strokes.length - 1);
  return canvas;
}
t = Date.now();
const canvas = render(ordered);
const renderMs = Date.now() - t;
let sumErr = 0, n = 0, sT = 0, sD = 0, sT2 = 0, sD2 = 0, sTD = 0, neverInked = 0, muddy = 0, fgCount = 0;
const bandDrawn = bands.map(() => ({ sum: 0, n: 0, target: 0 }));
let modelErr = 0;
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if ((foregroundMap.data[i] ?? 1) < 0.2) continue;
    fgCount++;
    const lum = 0.299 * canvas.px[i * 4] + 0.587 * canvas.px[i * 4 + 1] + 0.114 * canvas.px[i * 4 + 2];
    const drawn = Math.max(0, Math.min(1, 1 - lum));
    const target = toneMap.data[i];
    sumErr += Math.abs(drawn - target); n++;
    sT += target; sD += drawn; sT2 += target * target; sD2 += drawn * drawn; sTD += target * drawn;
    modelErr += Math.abs(drawn - Math.min(1, accumulator.achieved[i]));
    if (target > 0.45 && drawn < 0.06) neverInked++;
    if (target < 0.25 && drawn > 0.45) muddy++;
    for (let b = 0; b < bands.length; b++) if (target >= bands[b].lo && target < bands[b].hi) { bandDrawn[b].sum += drawn; bandDrawn[b].n++; bandDrawn[b].target += target; break; }
  }
}
const corr = (sTD / n - (sT / n) * (sD / n)) / Math.sqrt((sT2 / n - (sT / n) ** 2) * (sD2 / n - (sD / n) ** 2));
console.log('\n=== F. RENDERED RESULT vs THE PHOTO (' + renderMs + 'ms to render) ===');
console.log('  correlation(target tone, drawn darkness): ' + corr.toFixed(3) + '   (was 0.444; good > 0.8)');
console.log('  mean absolute tone error: ' + (100 * sumErr / n).toFixed(1) + '% of full range   (was 12.1%)');
console.log('  planner model error (predicted ink vs real render): ' + (100 * modelErr / n).toFixed(1) + '%');
for (let b = 0; b < bands.length; b++) {
  const avgTarget = bandDrawn[b].target / Math.max(1, bandDrawn[b].n);
  const avgDrawn = bandDrawn[b].sum / Math.max(1, bandDrawn[b].n);
  console.log('  ' + bands[b].name.padEnd(24) + ' target ' + avgTarget.toFixed(2) + '  ->  drawn ' + avgDrawn.toFixed(3));
}
console.log('  dark subjects left as blank paper: ' + pct(neverInked, fgCount) + '   (was 5.4%)');
console.log('  light subjects smudged too dark:   ' + pct(muddy, fgCount) + '   (was 2.5%)');

// G. cap behaviour
console.log('\n=== G. STROKE CAP ===');
for (const cap of [1500, 3000, 6000]) {
  const kept = orderStrokes(raw, cap);
  const l = {};
  for (const s of kept) l[s.layer] = (l[s.layer] || 0) + 1;
  console.log('  cap=' + String(cap).padStart(5) + ' -> kept ' + String(kept.length).padStart(5) +
    ' | structure=' + String(l.structure || 0).padStart(3) + ' outline=' + String(l.outline || 0).padStart(4) +
    ' features=' + String(l.features || 0).padStart(4) + ' light=' + String(l.lightTone || 0).padStart(4) +
    ' mid=' + String(l.midTone || 0).padStart(4) + ' dark=' + String(l.darkTone || 0).padStart(5) +
    ' accent=' + String(l.finalAccent || 0).padStart(4));
}

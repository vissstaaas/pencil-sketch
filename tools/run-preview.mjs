import { registerHooks } from 'node:module';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';

// Node strips TS types natively; this hook only teaches it the extensionless
// relative imports the app source uses.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith('.') && !/\.(ts|js|mjs|json)$/.test(specifier)) {
      try { return nextResolve(specifier + '.ts', context); } catch { /* fall through */ }
    }
    return nextResolve(specifier, context);
  },
});

const { decodePng, encodePng, resizeRaster } = await import('./harness/png.ts');
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
const OUT_DIR = process.argv[3] || 'tools/out';
const MAX_DIM = Number(process.env.SKETCH_MAX_DIM || 768);
const SEED = Number(process.env.SKETCH_SEED || 12345);
const BACKGROUND = '#fbf9f5';

// Mirrors useStrokeGeneration.DEFAULT_OPTIONS
const OPTIONS = {
  maxProcessingSize: MAX_DIM, targetStrokeCount: 8000, contourSensitivity: 0.5,
  shadingDensity: 0.6, backgroundSuppression: 0.5, minStrokeLength: 4, maxStrokeLength: 120, speedPreset: 'normal',
};

function mapToPng(map, scale = 1, gamma = 1) {
  const data = new Uint8ClampedArray(map.width * map.height * 4);
  for (let i = 0; i < map.data.length; i++) {
    const value = Math.max(0, Math.min(1, map.data[i] * scale));
    const g = Math.pow(value, gamma) * 255;
    data[i * 4] = g; data[i * 4 + 1] = g; data[i * 4 + 2] = g; data[i * 4 + 3] = 255;
  }
  return encodePng({ width: map.width, height: map.height, data });
}

function renderStrokes(strokes, width, height, background = BACKGROUND) {
  const canvas = new FakeCanvas();
  const engine = new SketchCanvasEngine({ canvas, width, height, pixelRatio: 1, background });
  engine.loadScript({
    version: '1.0',
    sourceImage: { width, height, processedWidth: width, processedHeight: height, mimeType: 'image/png' },
    canvas: { width, height, background, padding: 0 },
    generation: { seed: SEED, createdAt: new Date().toISOString(), options: OPTIONS, strokeCount: strokes.length },
    strokes: strokes.map((stroke, index) => ({ ...stroke, order: index })),
  });
  if (strokes.length > 0) engine.drawUntil(strokes.length - 1);
  return canvas.toPng();
}

mkdirSync(OUT_DIR, { recursive: true });

const source = decodePng(readFileSync(SAMPLE));
const resized = resizeRaster(source, MAX_DIM);
console.log('[input]', basename(SAMPLE), source.width + 'x' + source.height, '->', resized.width + 'x' + resized.height);

let start = Date.now();
const preprocessed = preprocessImage({ width: resized.width, height: resized.height, data: resized.data });
console.log('[preprocess]', Date.now() - start + 'ms');

const { width: W, height: H, toneMap, foregroundMap } = preprocessed;
const rng = new SeededRandom(SEED);
const accumulator = new ToneAccumulator(W, H, toneMap.data, OPTIONS.targetStrokeCount);
const claimed = new Uint8Array(W * H);
const budget = { remaining: OPTIONS.targetStrokeCount };
const context = { accumulator, claimed, budget };

start = Date.now();
const featureStrokes = detectFeatureAnchors(preprocessed, OPTIONS, rng, context);
const contourStrokes = generateContourStrokes(preprocessed, OPTIONS, rng, context);
const hairStrokes = generateHairFlowStrokes(preprocessed, OPTIONS, rng, context);
const shadingStrokes = generateShadingStrokes(preprocessed, OPTIONS, rng, context);
console.log('[generate]', Date.now() - start + 'ms');

const raw = [...contourStrokes, ...featureStrokes, ...hairStrokes, ...shadingStrokes];
const ordered = orderStrokes(raw, OPTIONS.targetStrokeCount);

const LAYERS = ['structure', 'outline', 'features', 'lightTone', 'midTone', 'darkTone', 'finalAccent'];
console.log('[groups] contours=' + contourStrokes.length, 'features=' + featureStrokes.length,
  'hair=' + hairStrokes.length, 'shading=' + shadingStrokes.length, 'raw=' + raw.length, 'kept=' + ordered.length);
for (const layer of LAYERS) {
  console.log('  ' + layer.padEnd(12), 'raw=' + String(raw.filter((s) => s.layer === layer).length).padStart(5),
    'kept=' + String(ordered.filter((s) => s.layer === layer).length).padStart(5));
}
console.log('[planner] ink delivered / ink required = ' + (accumulator.progress() * 100).toFixed(1) + '%');

writeFileSync(join(OUT_DIR, '00-source.png'), encodePng(resized));
writeFileSync(join(OUT_DIR, '01-edge-map.png'), mapToPng(preprocessed.edgeMap));
writeFileSync(join(OUT_DIR, '02-tone-target.png'), mapToPng(toneMap));
if (foregroundMap) writeFileSync(join(OUT_DIR, '03-subject-mask.png'), mapToPng(foregroundMap));

start = Date.now();
writeFileSync(join(OUT_DIR, '10-full.png'), renderStrokes(ordered, W, H));
console.log('[render]', Date.now() - start + 'ms');
writeFileSync(join(OUT_DIR, '11-contours.png'), renderStrokes(contourStrokes, W, H));
writeFileSync(join(OUT_DIR, '12-features.png'), renderStrokes(featureStrokes, W, H));
writeFileSync(join(OUT_DIR, '13-hair.png'), renderStrokes(hairStrokes, W, H));
writeFileSync(join(OUT_DIR, '14-shading.png'), renderStrokes(shadingStrokes, W, H));
console.log('[output]', OUT_DIR);

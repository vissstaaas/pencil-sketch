import type {
  StrokeWorkerPhase,
  StrokeWorkerRequest,
  StrokeWorkerResponse,
} from '../types/image-processing';
import type { StrokeScript } from '../types/stroke';
import { preprocessImage } from '../lib/image/preprocessImage';
import { generateContourStrokes } from '../lib/strokes/generateContourStrokes';
import { detectFeatureAnchors } from '../lib/strokes/detectFeatureAnchors';
import { generateHairFlowStrokes } from '../lib/strokes/generateHairFlowStrokes';
import { generateShadingStrokes } from '../lib/strokes/generateShadingStrokes';
import { orderStrokes } from '../lib/strokes/orderStrokes';
import { SeededRandom } from '../utils/random';

let currentRequestId: string | null = null;
let isCancelled = false;

self.onmessage = (e: MessageEvent<StrokeWorkerRequest>) => {
  const request = e.data;

  if (request.type === 'cancel') {
    if (request.requestId === currentRequestId) {
      isCancelled = true;
    }
    return;
  }

  if (request.type === 'generate') {
    currentRequestId = request.requestId;
    isCancelled = false;

    try {
      const { imageData, meta, options, requestId } = request;
      const seed = Math.floor(Math.random() * 1000000);
      const rng = new SeededRandom(seed);

      // Phase 1: Preprocessing & ETF Flow Field
      postProgress(requestId, 'preprocessing', 0.15, '正在分析全局结构与边缘流场...');
      if (isCancelled) return;

      const preprocessed = preprocessImage(imageData);

      // Dedicated feature claim mask to prevent double-tracing eyes, lips and nose
      const claimed = new Uint8Array(preprocessed.width * preprocessed.height);

      // Phase 2: Feature Anchors (Eyes, Pupils, Muzzle, Lips, Focal Accents)
      postProgress(requestId, 'contours', 0.35, '正在提取核心五官与神态焦点...');
      if (isCancelled) return;

      const featureAnchorStrokes = detectFeatureAnchors(preprocessed, options, rng, claimed);

      // Phase 3: Major Structural Contours & Outer Silhouettes
      postProgress(requestId, 'contours', 0.50, '正在生成多尺度长轮廓与骨架线...');
      if (isCancelled) return;

      const contourStrokes = generateContourStrokes(preprocessed, options, rng, claimed);

      // Phase 4: Fur & Hair Flow Curves (ETF Streamlines)
      postProgress(requestId, 'shading', 0.65, '正在计算流场毛发与发丝走向...');
      if (isCancelled) return;

      const hairFlowStrokes = generateHairFlowStrokes(preprocessed, options, rng);

      // Phase 5: Academic Tonal Shading & Hatching (Highlight Protected)
      postProgress(requestId, 'shading', 0.80, '正在铺设学院派体块明暗排线...');
      if (isCancelled) return;

      const shadingStrokes = generateShadingStrokes(preprocessed, options, rng);

      // Phase 6: Layer Ordering & Priority Capping
      postProgress(requestId, 'ordering', 0.92, '正在按经典绘画顺序进行图层编排...');
      if (isCancelled) return;

      const rawStrokes = [
        ...contourStrokes,
        ...featureAnchorStrokes,
        ...hairFlowStrokes,
        ...shadingStrokes,
      ];
      const ordered = orderStrokes(rawStrokes, options.targetStrokeCount);

      // Phase 7: Finalizing
      postProgress(requestId, 'finalizing', 1.0, '素描笔触脚本生成完成！');

      const script: StrokeScript = {
        version: '1.0',
        sourceImage: {
          width: meta.width,
          height: meta.height,
          processedWidth: imageData.width,
          processedHeight: imageData.height,
          mimeType: meta.mimeType,
        },
        canvas: {
          width: imageData.width,
          height: imageData.height,
          background: '#fbf9f5',
          padding: 0,
        },
        generation: {
          seed,
          createdAt: new Date().toISOString(),
          options,
          strokeCount: ordered.length,
        },
        strokes: ordered,
      };

      const response: StrokeWorkerResponse = {
        type: 'complete',
        requestId,
        script,
      };

      self.postMessage(response);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      const response: StrokeWorkerResponse = {
        type: 'error',
        requestId: request.requestId,
        error: {
          code: 'GENERATION_FAILED',
          message,
          detail: err,
        },
      };
      self.postMessage(response);
    }
  }
};

function postProgress(
  requestId: string,
  phase: StrokeWorkerPhase,
  progress: number,
  message: string
): void {
  const msg: StrokeWorkerResponse = {
    type: 'progress',
    requestId,
    phase,
    progress,
    message,
  };
  self.postMessage(msg);
}

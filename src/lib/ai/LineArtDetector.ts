import * as ort from 'onnxruntime-web';
import type { ImageMap } from '../../types/image-processing';


// Use CDN for WASM files to completely bypass Vite's dynamic import blocking
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';

export class LineArtDetector {
  private session: ort.InferenceSession | null = null;

  async init() {
    if (this.session) return;
    try {
      this.session = await ort.InferenceSession.create('/models/pidinet_tiny.onnx', {
        executionProviders: ['wasm']
      });
      console.log('ONNX Session initialized with PiDiNet Tiny.');
    } catch (err) {
      console.error('Failed to init ONNX session:', err);
      throw err;
    }
  }

  async detect(imageData: ImageData): Promise<ImageMap> {
    if (!this.session) await this.init();
    
    const width = imageData.width;
    const height = imageData.height;
    
    // Prepare input tensor: [1, 3, height, width]
    const tensorData = new Float32Array(3 * height * width);
    const channelSize = height * width;
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIdx = (y * width + x) * 4;
        const tensorIdx = y * width + x;
        // Normalize 0-255 to 0.0-1.0 (standard for PiDiNet)
        tensorData[tensorIdx] = imageData.data[pixelIdx] / 255.0;                 // R
        tensorData[channelSize + tensorIdx] = imageData.data[pixelIdx + 1] / 255.0; // G
        tensorData[2 * channelSize + tensorIdx] = imageData.data[pixelIdx + 2] / 255.0; // B
      }
    }
    
    const inputName = this.session!.inputNames[0];
    const tensor = new ort.Tensor('float32', tensorData, [1, 3, height, width]);
    
    // Run Inference
    const results = await this.session!.run({ [inputName]: tensor });
    
    // PiDiNet outputs: ['side1', 'side2', 'side3', 'side4', 'fused']
    // 'fused' contains the unified, multi-scale clean edge map
    const outputTensor = results['fused'] || results[this.session!.outputNames[this.session!.outputNames.length - 1]];
    const outputData = outputTensor.data as Float32Array;
    
    const edgeMap: ImageMap = {
      width,
      height,
      data: new Float32Array(width * height)
    };
    
    // Map ONNX output back to edgeMap
    for (let i = 0; i < width * height; i++) {
      let val = outputData[i];
      // PiDiNet tiny might output raw logits, apply sigmoid if values are outside [0, 1]
      if (val < 0 || val > 1) {
        val = 1 / (1 + Math.exp(-val));
      }
      edgeMap.data[i] = val;
    }
    
    return edgeMap;
  }
}

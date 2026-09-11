import React, { useCallback, useState } from 'react';
import { Pencil, Info } from 'lucide-react';
import { ImageUploader } from '../components/ImageUploader';
import { SketchCanvas } from '../components/SketchCanvas';
import { PlaybackControls } from '../components/PlaybackControls';
import { GenerationStatus } from '../components/GenerationStatus';
import { useStrokeGeneration } from '../hooks/useStrokeGeneration';
import { useSketchPlayback } from '../hooks/useSketchPlayback';
import type { SketchCanvasEngine } from '../lib/canvas/SketchCanvasEngine';
import type { UploadImageMeta } from '../types/image-processing';

export const SketchMvpPage: React.FC = () => {
  const [engine, setEngine] = useState<SketchCanvasEngine | null>(null);
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 680, height: 680 });
  const [uploadedMeta, setUploadedMeta] = useState<UploadImageMeta | null>(null);

  const { state: genState, generateFromFile, cancel, reset } = useStrokeGeneration();
  const {
    state: playbackState,
    play,
    pause,
    resume,
    replay,
    seek,
    setSpeed,
    exportPng,
  } = useSketchPlayback(engine, genState.script || null);

  const handleEngineReady = useCallback((newEngine: SketchCanvasEngine) => {
    setEngine(newEngine);
  }, []);

  const handleFileSelect = useCallback(
    async (file: File) => {
      try {
        reset();
        const script = await generateFromFile(file);
        setUploadedMeta({
          fileName: file.name,
          mimeType: file.type,
          sizeBytes: file.size,
          width: script.sourceImage.width,
          height: script.sourceImage.height,
        });
        setCanvasDimensions({
          width: script.canvas.width,
          height: script.canvas.height,
        });
      } catch (err) {
        console.error('Stroke generation failed:', err);
      }
    },
    [generateFromFile, reset]
  );

  const handleExport = useCallback(() => {
    try {
      const dataUrl = exportPng();
      const link = document.createElement('a');
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      link.download = `pencil-sketch-${timestamp}.png`;
      link.href = dataUrl;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error('Export PNG failed:', err);
      alert('导出 PNG 失败：' + (err instanceof Error ? err.message : String(err)));
    }
  }, [exportPng]);

  const isScriptReady = !!genState.script && genState.script.strokes.length > 0;
  const canPlay = isScriptReady && playbackState.status !== 'playing';
  const canPause = playbackState.status === 'playing';
  const canResume = playbackState.status === 'paused';
  const canReplay = isScriptReady;
  const canExport = !!engine;

  return (
    <div className="app-container">
      {/* Top Navigation Bar */}
      <header className="app-header">
        <div className="header-brand">
          <div className="logo-icon">
            <Pencil size={22} />
          </div>
          <div className="brand-text">
            <h1>Pencil Sketch Studio</h1>
            <span className="version-tag">MVP · 逐笔绘制引擎</span>
          </div>
        </div>
        <div className="header-tips">
          <Info size={15} />
          <span>真实笔触脚本 · 无蒙版渐显 · 纯粹石墨素描回放</span>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="app-workspace">
        {/* Left / Center: Drawing Board Viewport */}
        <div className="workspace-canvas-panel">
          <div className="canvas-container-outer">
            <SketchCanvas
              width={canvasDimensions.width}
              height={canvasDimensions.height}
              onEngineReady={handleEngineReady}
            />
          </div>

          {/* Bottom Floating Playback Controls Bar */}
          <div className="workspace-controls-bar">
            <PlaybackControls
              state={playbackState}
              canPlay={canPlay}
              canPause={canPause}
              canResume={canResume}
              canReplay={canReplay}
              canExport={canExport}
              onPlay={play}
              onPause={pause}
              onResume={resume}
              onReplay={replay}
              onSeek={seek}
              onSetSpeed={setSpeed}
              onExport={handleExport}
            />
          </div>
        </div>

        {/* Right: Sidebar with Upload and Generation Inspector */}
        <aside className="workspace-sidebar">
          <div className="sidebar-card">
            <h2 className="card-title">图片输入</h2>
            <ImageUploader
              onFileSelect={handleFileSelect}
              disabled={genState.status === 'generating'}
            />
          </div>

          <div className="sidebar-card">
            <div className="card-title-row">
              <h2 className="card-title">笔触生成状态</h2>
              {genState.status === 'generating' && (
                <button className="btn-cancel-generation" onClick={cancel}>
                  取消
                </button>
              )}
            </div>
            <GenerationStatus state={genState} />
          </div>

          {uploadedMeta && (
            <div className="sidebar-card">
              <h2 className="card-title">原图参数</h2>
              <div className="meta-list">
                <div className="meta-row">
                  <span className="k">文件名</span>
                  <span className="v" title={uploadedMeta.fileName}>{uploadedMeta.fileName}</span>
                </div>
                <div className="meta-row">
                  <span className="k">分辨率</span>
                  <span className="v">{uploadedMeta.width} × {uploadedMeta.height} px</span>
                </div>
                <div className="meta-row">
                  <span className="k">文件大小</span>
                  <span className="v">{(uploadedMeta.sizeBytes / 1024).toFixed(1)} KB</span>
                </div>
              </div>
            </div>
          )}

          <div className="sidebar-card tips-card">
            <h2 className="card-title">素描绘制层次规范</h2>
            <ol className="drawing-layers-list">
              <li><span className="layer-dot layer-1" /> <strong>淡结构线</strong>：轻描主体骨架与比例网格</li>
              <li><span className="layer-dot layer-2" /> <strong>主体轮廓</strong>：确立外缘与关键特征</li>
              <li><span className="layer-dot layer-3" /> <strong>浅层调子</strong>：铺设大面积受光/背光灰面</li>
              <li><span className="layer-dot layer-4" /> <strong>暗部排线</strong>：依据表面切向交叉排线</li>
              <li><span className="layer-dot layer-5" /> <strong>重音强调</strong>：瞳孔、暗角与深邃阴影压深</li>
            </ol>
          </div>
        </aside>
      </main>
    </div>
  );
};

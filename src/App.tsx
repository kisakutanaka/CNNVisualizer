import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, CSSProperties } from 'react'
import type * as tf from '@tensorflow/tfjs'
import {
  loadMobileNet,
  classify,
  buildActivationModel,
  buildGradCamModels,
  analyzeAllLayers,
  OVERLAY_LAYERS,
  type Prediction,
  type LayerAnalysis,
} from './mobilenet'
import { drawActivationOverlay, drawChannelTile } from './heatmap'
import './App.css'

/** 合成ヒートマップへの実際の貢献度(0〜1)を、チャンネルごとに算出する。負の貢献は0として扱う */
function getChannelProminence(contributions: Float32Array): Float32Array {
  const prominence = new Float32Array(contributions.length)
  let max = 0
  for (const contribution of contributions) {
    if (contribution > max) max = contribution
  }
  if (max <= 0) return prominence
  for (let i = 0; i < contributions.length; i++) {
    prominence[i] = Math.max(0, contributions[i]) / max
  }
  return prominence
}

function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [activationModel, setActivationModel] = useState<tf.LayersModel | null>(null)
  const [gradCamModels, setGradCamModels] = useState<Map<string, tf.LayersModel> | null>(null)
  const [layerNames, setLayerNames] = useState<string[]>([])
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [isClassifying, setIsClassifying] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [layerIndex, setLayerIndex] = useState(0)
  const [layerAnalysis, setLayerAnalysis] = useState<Map<string, LayerAnalysis>>(new Map())
  const [showChannelGrid, setShowChannelGrid] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const currentAnalysis = layerAnalysis.get(OVERLAY_LAYERS[layerIndex])
  const channelHeatmaps = useMemo(
    () => currentAnalysis?.channelHeatmaps ?? [],
    [currentAnalysis],
  )
  const channelProminence = useMemo(
    () =>
      getChannelProminence(currentAnalysis?.gradCam.channelContributions ?? new Float32Array()),
    [currentAnalysis],
  )

  useEffect(() => {
    loadMobileNet().then((loadedModel) => {
      setModel(loadedModel)
      const built = buildActivationModel(loadedModel)
      setActivationModel(built.activationModel)
      setLayerNames(built.layerNames)
      setGradCamModels(buildGradCamModels(loadedModel, OVERLAY_LAYERS))
    })
  }, [])

  // 選択中の層のGrad-CAM結果を描き直すだけ。計算自体はhandleImageLoadで画像ごとに1回だけ済ませてある
  useEffect(() => {
    if (!canvasRef.current || !currentAnalysis) return
    drawActivationOverlay(canvasRef.current, currentAnalysis.gradCam.heatmap)
  }, [currentAnalysis])

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = () => {
      setPredictions([])
      setLayerIndex(0)
      setShowChannelGrid(false)
      setLayerAnalysis(new Map())
      setImageUrl(reader.result as string)
    }
    reader.readAsDataURL(file)
  }

  function handleImageLoad() {
    if (!model || !imgRef.current) return
    setIsClassifying(true)
    const result = classify(model, imgRef.current)
    setPredictions(result)
    setIsClassifying(false)

    if (!activationModel || !gradCamModels) return
    setIsAnalyzing(true)
    // setIsAnalyzingの描画を先に反映させてから、重い解析（Grad-CAM＋個別ニューロン）に入る
    setTimeout(() => {
      if (!imgRef.current) return
      const results = analyzeAllLayers(
        activationModel,
        layerNames,
        gradCamModels,
        OVERLAY_LAYERS,
        imgRef.current,
      )
      setLayerAnalysis(results)
      setIsAnalyzing(false)
    }, 0)
  }

  function showPreviousLayer() {
    setLayerIndex((i) => Math.max(0, i - 1))
  }

  function showNextLayer() {
    setLayerIndex((i) => Math.min(OVERLAY_LAYERS.length - 1, i + 1))
  }

  return (
    <main id="app">
      <h1>CNN Visualizer</h1>

      {!model && <p className="status">モデルを読み込み中…</p>}

      <div className="image-inputs">
        <label className="upload-button">
          画像をアップロード
          <input type="file" accept="image/*" onChange={handleFileChange} />
        </label>

        <label className="upload-button">
          カメラで撮影
          <input
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleFileChange}
          />
        </label>
      </div>

      {imageUrl && (
        <>
          {/* imgは個別ニューロン表示中も非表示にするだけでDOMからは外さない
              （tf.browser.fromPixelsが引き続き画像データを読めるようにするため） */}
          <div className="preview" hidden={showChannelGrid}>
            <img
              ref={imgRef}
              src={imageUrl}
              alt="アップロードした画像のプレビュー"
              onLoad={handleImageLoad}
            />
            <canvas ref={canvasRef} className="activation-overlay" />
            {(isClassifying || isAnalyzing) && (
              <div className="processing-overlay">
                <span>解析中…</span>
              </div>
            )}
          </div>

          {showChannelGrid && (
            <div className="channel-grid-wrap">
              <p className="channel-grid-heading">
                {OVERLAY_LAYERS[layerIndex]}（{channelHeatmaps.length}チャンネル）
              </p>
              <p className="channel-grid-caption">
                枠線が太く濃いほど、予測への寄与度が高いニューロン
              </p>
              <div className="channel-grid">
                {channelHeatmaps.map((heatmap, i) => (
                  <canvas
                    key={i}
                    className="channel-tile"
                    style={
                      {
                        '--prominence': channelProminence[i] ?? 0,
                      } as CSSProperties
                    }
                    ref={(el) => {
                      if (el && imgRef.current) drawChannelTile(el, heatmap, imgRef.current)
                    }}
                  />
                ))}
              </div>
            </div>
          )}

          <div className="layer-nav">
            <button
              type="button"
              onClick={showPreviousLayer}
              disabled={isAnalyzing || layerIndex === 0}
              aria-label="前の層"
            >
              ‹
            </button>
            <span className="layer-indicator">
              層 {layerIndex + 1} / {OVERLAY_LAYERS.length}
            </span>
            <button
              type="button"
              onClick={showNextLayer}
              disabled={isAnalyzing || layerIndex === OVERLAY_LAYERS.length - 1}
              aria-label="次の層"
            >
              ›
            </button>
          </div>

          <button
            type="button"
            className="view-toggle"
            disabled={isAnalyzing}
            onClick={() => setShowChannelGrid((v) => !v)}
          >
            {showChannelGrid ? 'オーバーレイ表示に戻る' : '個別ニューロンを見る'}
          </button>
        </>
      )}

      {predictions.length > 0 && (
        <ul className="predictions">
          {predictions.map((prediction) => (
            <li key={prediction.className}>
              <span className="label">{prediction.className}</span>
              <span className="probability">
                {(prediction.probability * 100).toFixed(1)}%
              </span>
            </li>
          ))}
        </ul>
      )}
    </main>
  )
}

export default App

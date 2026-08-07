import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, CSSProperties } from 'react'
import type * as tf from '@tensorflow/tfjs'
import {
  loadMobileNet,
  classify,
  buildActivationModel,
  buildGradCamModels,
  getGradCam,
  getChannelHeatmaps,
  OVERLAY_LAYERS,
  type Prediction,
  type ActivationHeatmap,
} from './mobilenet'
import { drawActivationOverlay, drawChannelTile } from './heatmap'
import './App.css'

/** 予測への寄与度(0〜1)を、チャンネルごとに算出する。負の寄与（予測を妨げる方向）は0として扱う */
function getChannelProminence(weights: Float32Array): Float32Array {
  const prominence = new Float32Array(weights.length)
  let max = 0
  for (const weight of weights) {
    if (weight > max) max = weight
  }
  if (max <= 0) return prominence
  for (let i = 0; i < weights.length; i++) {
    prominence[i] = Math.max(0, weights[i]) / max
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
  const [layerIndex, setLayerIndex] = useState(0)
  const [imageLoadedAt, setImageLoadedAt] = useState(0)
  const [showChannelGrid, setShowChannelGrid] = useState(false)
  const [channelHeatmaps, setChannelHeatmaps] = useState<ActivationHeatmap[]>([])
  const [channelWeights, setChannelWeights] = useState<Float32Array>(new Float32Array())
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const channelProminence = useMemo(
    () => getChannelProminence(channelWeights),
    [channelWeights],
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

  // 画像もしくは選択中の層が変わるたびに、Grad-CAMのオーバーレイを描き直す
  // （予測クラスへの寄与度で重み付けした合成ヒートマップ。チャンネルごとの寄与度は
  // 個別ニューロン表示グリッドの上位ハイライトにも使う）
  useEffect(() => {
    if (
      !activationModel ||
      !gradCamModels ||
      !imgRef.current ||
      !canvasRef.current ||
      imageLoadedAt === 0
    ) {
      return
    }
    const result = getGradCam(
      activationModel,
      layerNames,
      gradCamModels,
      OVERLAY_LAYERS[layerIndex],
      imgRef.current,
    )
    drawActivationOverlay(canvasRef.current, result.heatmap)
    setChannelWeights(result.channelWeights)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layerIndex, imageLoadedAt, activationModel, gradCamModels])

  // 個別ニューロン表示中は、層やグリッド表示のON/OFFが変わるたびにチャンネルごとの活性化を取得する
  useEffect(() => {
    if (!showChannelGrid || !activationModel || !imgRef.current || imageLoadedAt === 0) {
      return
    }
    const heatmaps = getChannelHeatmaps(
      activationModel,
      layerNames,
      OVERLAY_LAYERS[layerIndex],
      imgRef.current,
    )
    setChannelHeatmaps(heatmaps)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showChannelGrid, layerIndex, imageLoadedAt, activationModel])

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = () => {
      setPredictions([])
      setLayerIndex(0)
      setShowChannelGrid(false)
      setChannelHeatmaps([])
      setChannelWeights(new Float32Array())
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
    setImageLoadedAt(Date.now())
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
              disabled={layerIndex === 0}
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
              disabled={layerIndex === OVERLAY_LAYERS.length - 1}
              aria-label="次の層"
            >
              ›
            </button>
          </div>

          <button
            type="button"
            className="view-toggle"
            onClick={() => setShowChannelGrid((v) => !v)}
          >
            {showChannelGrid ? 'オーバーレイ表示に戻る' : '個別ニューロンを見る'}
          </button>
        </>
      )}

      {isClassifying && <p className="status">分類中…</p>}

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

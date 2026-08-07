import type { ActivationHeatmap } from './mobilenet'

/** 活性化の強さ(0〜1)を青→緑→赤のヒートマップ色に変換する */
function heatmapColor(value: number): [number, number, number] {
  const r = Math.round(255 * value)
  const g = Math.round(255 * (1 - Math.abs(value - 0.5) * 2))
  const b = Math.round(255 * (1 - value))
  return [r, g, b]
}

function heatmapToImageData(
  heatmap: ActivationHeatmap,
  alphaFromValue: (value: number) => number,
): ImageData {
  const imageData = new ImageData(heatmap.width, heatmap.height)
  for (let i = 0; i < heatmap.data.length; i++) {
    const value = heatmap.data[i]
    const [r, g, b] = heatmapColor(value)
    imageData.data[i * 4] = r
    imageData.data[i * 4 + 1] = g
    imageData.data[i * 4 + 2] = b
    imageData.data[i * 4 + 3] = alphaFromValue(value)
  }
  return imageData
}

/** ヒートマップをcanvasに描画する。活性化が弱い部分ほど透明にし、元画像に重ねられるようにする */
export function drawActivationOverlay(
  canvas: HTMLCanvasElement,
  heatmap: ActivationHeatmap,
) {
  canvas.width = heatmap.width
  canvas.height = heatmap.height

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.putImageData(heatmapToImageData(heatmap, (value) => Math.round(value * 255)), 0, 0)
}

// ヒートマップ合成用に使い回す一時canvas（putImageDataは透過を合成できないため経由させる）
let heatmapLayerCanvas: HTMLCanvasElement | null = null
function getHeatmapLayerCanvas(): HTMLCanvasElement {
  if (!heatmapLayerCanvas) heatmapLayerCanvas = document.createElement('canvas')
  return heatmapLayerCanvas
}

// タイルの描画解像度。活性化マップ自体の解像度（深い層では7x7程度まで下がる）に
// 引きずられて背景画像まで粗くならないよう、canvasの実解像度はこれで固定する
const TILE_RESOLUTION = 128

// グレースケール化した元画像のキャッシュ。タイルの数だけ`filter: grayscale`をかけ直したり、
// 元画像の解像度（iPhone写真だと3000〜4000px級）から毎回縮小描画したりすると重いため、
// 画像ごとに一度だけ「グレースケール化 かつ タイル解像度への縮小」まで済ませてキャッシュし、
// 各タイルへは等倍コピーだけで済ませる
let grayscaleSourceCache: { src: string; canvas: HTMLCanvasElement } | null = null
function getGrayscaleSource(sourceImage: HTMLImageElement): HTMLCanvasElement {
  if (grayscaleSourceCache?.src === sourceImage.src) {
    return grayscaleSourceCache.canvas
  }
  const canvas = document.createElement('canvas')
  canvas.width = TILE_RESOLUTION
  canvas.height = TILE_RESOLUTION
  const ctx = canvas.getContext('2d')
  if (ctx) {
    ctx.filter = 'grayscale(1)'
    ctx.drawImage(sourceImage, 0, 0, TILE_RESOLUTION, TILE_RESOLUTION)
  }
  grayscaleSourceCache = { src: sourceImage.src, canvas }
  return canvas
}

/**
 * 個別ニューロン表示グリッド用のタイルを描画する。
 * モノクロにした元画像を背景に敷いてから活性化ヒートマップを半透明で重ね、
 * 層が深くなり活性化マップの解像度が下がっても画像内のどこに反応しているか分かるようにする
 */
export function drawChannelTile(
  canvas: HTMLCanvasElement,
  heatmap: ActivationHeatmap,
  sourceImage: HTMLImageElement,
) {
  canvas.width = TILE_RESOLUTION
  canvas.height = TILE_RESOLUTION

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  // 背景は既にタイル解像度へ縮小済みのキャッシュを等倍コピーするだけ
  const grayscaleSource = getGrayscaleSource(sourceImage)
  ctx.drawImage(grayscaleSource, 0, 0)

  // ヒートマップは活性化マップ本来の解像度で作ってから、タイル解像度へ拡大描画する
  const layer = getHeatmapLayerCanvas()
  layer.width = heatmap.width
  layer.height = heatmap.height
  const layerCtx = layer.getContext('2d')
  if (!layerCtx) return
  layerCtx.putImageData(
    heatmapToImageData(heatmap, (value) => Math.round(value * 255)),
    0,
    0,
  )

  ctx.drawImage(layer, 0, 0, canvas.width, canvas.height)
}

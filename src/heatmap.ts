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

// 背景キャッシュの解像度。個別ニューロン表示グリッドの背景としてWebGLテクスチャに
// 使われるだけなので、この程度で十分（GPU側で拡大される）
const TILE_RESOLUTION = 128

// 縮小した元画像（色つきのまま）のキャッシュ。個別ニューロン表示グリッドの背景として使う。
// 元画像の解像度（iPhone写真だと3000〜4000px級）から毎回縮小描画すると重いため、
// 画像ごとに一度だけ縮小して キャッシュしておく。
// グレースケール化はCanvas2Dの`filter`（Safariでのサポートが不安定）ではなく、
// WebGL側のフラグメントシェーダーで行う（channelGridRenderer.ts）
let scaledSourceCache: { src: string; canvas: HTMLCanvasElement } | null = null
export function getScaledSource(sourceImage: HTMLImageElement): HTMLCanvasElement {
  if (scaledSourceCache?.src === sourceImage.src) {
    return scaledSourceCache.canvas
  }
  const canvas = document.createElement('canvas')
  canvas.width = TILE_RESOLUTION
  canvas.height = TILE_RESOLUTION
  const ctx = canvas.getContext('2d')
  if (ctx) {
    ctx.drawImage(sourceImage, 0, 0, TILE_RESOLUTION, TILE_RESOLUTION)
  }
  scaledSourceCache = { src: sourceImage.src, canvas }
  return canvas
}

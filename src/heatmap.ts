import type { ActivationHeatmap } from './mobilenet'

/** 活性化の強さ(0〜1)を青→緑→赤のヒートマップ色に変換する */
function heatmapColor(value: number): [number, number, number] {
  const r = Math.round(255 * value)
  const g = Math.round(255 * (1 - Math.abs(value - 0.5) * 2))
  const b = Math.round(255 * (1 - value))
  return [r, g, b]
}

function paintHeatmap(
  canvas: HTMLCanvasElement,
  heatmap: ActivationHeatmap,
  alphaFromValue: (value: number) => number,
) {
  canvas.width = heatmap.width
  canvas.height = heatmap.height

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const imageData = ctx.createImageData(heatmap.width, heatmap.height)
  for (let i = 0; i < heatmap.data.length; i++) {
    const value = heatmap.data[i]
    const [r, g, b] = heatmapColor(value)
    imageData.data[i * 4] = r
    imageData.data[i * 4 + 1] = g
    imageData.data[i * 4 + 2] = b
    imageData.data[i * 4 + 3] = alphaFromValue(value)
  }
  ctx.putImageData(imageData, 0, 0)
}

/** ヒートマップをcanvasに描画する。活性化が弱い部分ほど透明にし、元画像に重ねられるようにする */
export function drawActivationOverlay(
  canvas: HTMLCanvasElement,
  heatmap: ActivationHeatmap,
) {
  paintHeatmap(canvas, heatmap, (value) => Math.round(value * 255))
}

/** 個別ニューロン表示グリッド用のタイルを描画する。常に不透明で描画する */
export function drawChannelTile(canvas: HTMLCanvasElement, heatmap: ActivationHeatmap) {
  paintHeatmap(canvas, heatmap, () => 255)
}

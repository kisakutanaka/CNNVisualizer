import type { ActivationHeatmap } from './mobilenet'
import { getGrayscaleSource } from './heatmap'

// 頂点ごとに (位置x,y / テクスチャ座標u,v / チャンネル番号 / 寄与度) を渡し、
// タイル1枚1枚を個別にdrawImageするのではなく、全チャンネル分の四角形を
// 1本の頂点バッファにまとめて1回のdraw callで描き切る
const VERTEX_SHADER = `#version 300 es
in vec2 aPosition;
in vec2 aTexCoord;
in float aLayer;
in float aProminence;

uniform vec2 uResolution;

out vec2 vTexCoord;
out float vLayer;
out float vProminence;

void main() {
  vec2 zeroToOne = aPosition / uResolution;
  vec2 clipSpace = zeroToOne * 2.0 - 1.0;
  gl_Position = vec4(clipSpace.x, -clipSpace.y, 0.0, 1.0);
  vTexCoord = aTexCoord;
  vLayer = aLayer;
  vProminence = aProminence;
}
`

const FRAGMENT_SHADER = `#version 300 es
precision mediump float;

in vec2 vTexCoord;
in float vLayer;
in float vProminence;

uniform sampler2D uBgTexture;
uniform mediump sampler2DArray uHeatTexture;

out vec4 outColor;

void main() {
  vec3 bg = texture(uBgTexture, vTexCoord).rgb;
  float value = texture(uHeatTexture, vec3(vTexCoord, vLayer)).r;

  // heatmap.tsのheatmapColorと同じ配色（青→緑→赤）
  vec3 heatColor = vec3(
    value,
    1.0 - abs(value - 0.5) * 2.0,
    1.0 - value
  );

  // 寄与度が低いニューロンは、活性化がどれだけ強くても色が沈んでグレー（背景）に近づく。
  // 「赤い＝反応した」ではなく「赤くて浮き上がっている＝分類の判断に効いた」という
  // 見た目にするため、活性化の強さと寄与度の両方でアルファを決める
  float alpha = value * vProminence;
  outColor = vec4(mix(bg, heatColor, alpha), 1.0);
}
`

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)
  if (!shader) throw new Error('シェーダーの作成に失敗しました')
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader)
    gl.deleteShader(shader)
    throw new Error(`シェーダーのコンパイルに失敗しました: ${info}`)
  }
  return shader
}

type GridGLState = {
  gl: WebGL2RenderingContext
  program: WebGLProgram
  vao: WebGLVertexArrayObject
  vertexBuffer: WebGLBuffer
  bgTexture: WebGLTexture
  heatTexture: WebGLTexture
  uResolution: WebGLUniformLocation | null
  uBgTexture: WebGLUniformLocation | null
  uHeatTexture: WebGLUniformLocation | null
}

const glStateCache = new WeakMap<HTMLCanvasElement, GridGLState>()

function initGridGLState(canvas: HTMLCanvasElement): GridGLState {
  const gl = canvas.getContext('webgl2')
  if (!gl) throw new Error('WebGL2に対応していません')

  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER)
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER)
  const program = gl.createProgram()
  if (!program) throw new Error('プログラムの作成に失敗しました')
  gl.attachShader(program, vertexShader)
  gl.attachShader(program, fragmentShader)
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`プログラムのリンクに失敗しました: ${gl.getProgramInfoLog(program)}`)
  }

  const vao = gl.createVertexArray()
  const vertexBuffer = gl.createBuffer()
  const bgTexture = gl.createTexture()
  const heatTexture = gl.createTexture()
  if (!vao || !vertexBuffer || !bgTexture || !heatTexture) {
    throw new Error('WebGLリソースの作成に失敗しました')
  }

  const aPosition = gl.getAttribLocation(program, 'aPosition')
  const aTexCoord = gl.getAttribLocation(program, 'aTexCoord')
  const aLayer = gl.getAttribLocation(program, 'aLayer')
  const aProminence = gl.getAttribLocation(program, 'aProminence')

  gl.bindVertexArray(vao)
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
  const stride = 6 * 4
  gl.enableVertexAttribArray(aPosition)
  gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, stride, 0)
  gl.enableVertexAttribArray(aTexCoord)
  gl.vertexAttribPointer(aTexCoord, 2, gl.FLOAT, false, stride, 2 * 4)
  gl.enableVertexAttribArray(aLayer)
  gl.vertexAttribPointer(aLayer, 1, gl.FLOAT, false, stride, 4 * 4)
  gl.enableVertexAttribArray(aProminence)
  gl.vertexAttribPointer(aProminence, 1, gl.FLOAT, false, stride, 5 * 4)
  gl.bindVertexArray(null)

  return {
    gl,
    program,
    vao,
    vertexBuffer,
    bgTexture,
    heatTexture,
    uResolution: gl.getUniformLocation(program, 'uResolution'),
    uBgTexture: gl.getUniformLocation(program, 'uBgTexture'),
    uHeatTexture: gl.getUniformLocation(program, 'uHeatTexture'),
  }
}

// チャンネル数分の四角形（2三角形=6頂点）をまとめた1本の頂点バッファを組み立てる。
// texcoordはbg・heatともに0〜1で共通（heat側はテクスチャ配列のレイヤーindexで
// チャンネルを選ぶので、チャンネルごとのアトラス座標計算が不要になる）
function buildVertices(
  count: number,
  columns: number,
  tileSize: number,
  gap: number,
  prominence: Float32Array,
): Float32Array {
  const floatsPerVertex = 6 // x, y, u, v, layer, prominence
  const vertices = new Float32Array(count * 6 * floatsPerVertex)

  let o = 0
  for (let i = 0; i < count; i++) {
    const col = i % columns
    const row = Math.floor(i / columns)
    const x0 = col * (tileSize + gap)
    const y0 = row * (tileSize + gap)
    const x1 = x0 + tileSize
    const y1 = y0 + tileSize
    const layer = i
    const p = prominence[i] ?? 0

    const corners = [
      [x0, y0, 0, 0],
      [x1, y0, 1, 0],
      [x0, y1, 0, 1],
      [x1, y0, 1, 0],
      [x1, y1, 1, 1],
      [x0, y1, 0, 1],
    ]
    for (const [x, y, u, v] of corners) {
      vertices[o++] = x
      vertices[o++] = y
      vertices[o++] = u
      vertices[o++] = v
      vertices[o++] = layer
      vertices[o++] = p
    }
  }

  return vertices
}

function buildHeatmapTextureData(heatmaps: ActivationHeatmap[]) {
  const width = heatmaps[0]?.width ?? 0
  const height = heatmaps[0]?.height ?? 0
  const data = new Uint8Array(width * height * heatmaps.length)
  let o = 0
  for (const h of heatmaps) {
    for (let i = 0; i < h.data.length; i++) {
      data[o++] = Math.max(0, Math.min(255, Math.round(h.data[i] * 255)))
    }
  }
  return { data, width, height }
}

// 一部モバイルGPUのレンダーバッファ上限を考慮し、canvasの実解像度に安全マージンを設ける
const MAX_CANVAS_HEIGHT_PX = 8000

/**
 * 個別ニューロン表示グリッドを描画する。
 * モノクロの元画像を背景に敷き、その上に各チャンネルの活性化ヒートマップを半透明で重ねる
 * （寄与度が低いほど薄れて背景に沈む）。チャンネル数だけcanvasやdrawImageを繰り返すのではなく、
 * 全チャンネル分の頂点をまとめた1本の頂点バッファと、チャンネルごとに1レイヤーを割り当てた
 * テクスチャ配列を使い、1回のバインド・1回のdraw callでグリッド全体を描き切る
 */
export function renderChannelGrid(
  canvas: HTMLCanvasElement,
  heatmaps: ActivationHeatmap[],
  prominence: Float32Array,
  sourceImage: HTMLImageElement,
  columns: number,
  cssTileSize: number,
  cssGap: number,
): void {
  if (heatmaps.length === 0) return

  let state = glStateCache.get(canvas)
  if (!state) {
    state = initGridGLState(canvas)
    glStateCache.set(canvas, state)
  }
  const { gl } = state

  const rows = Math.ceil(heatmaps.length / columns)
  const cssWidth = columns * cssTileSize + (columns - 1) * cssGap
  const cssHeight = rows * cssTileSize + (rows - 1) * cssGap

  let dpr = window.devicePixelRatio || 1
  if (cssHeight * dpr > MAX_CANVAS_HEIGHT_PX) {
    dpr = Math.max(1, MAX_CANVAS_HEIGHT_PX / cssHeight)
  }

  canvas.style.width = `${cssWidth}px`
  canvas.style.height = `${cssHeight}px`
  canvas.width = Math.round(cssWidth * dpr)
  canvas.height = Math.round(cssHeight * dpr)

  const vertices = buildVertices(
    heatmaps.length,
    columns,
    cssTileSize * dpr,
    cssGap * dpr,
    prominence,
  )
  gl.bindBuffer(gl.ARRAY_BUFFER, state.vertexBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW)

  // --- 背景（モノクロ元画像）テクスチャ。全タイルで共通の1枚を使い回す ---
  const grayscaleSource = getGrayscaleSource(sourceImage)
  gl.activeTexture(gl.TEXTURE0)
  gl.bindTexture(gl.TEXTURE_2D, state.bgTexture)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, grayscaleSource)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

  // --- 活性化ヒートマップ。チャンネルごとに1レイヤーを割り当てたテクスチャ配列 ---
  const { data, width, height } = buildHeatmapTextureData(heatmaps)
  gl.activeTexture(gl.TEXTURE1)
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, state.heatTexture)
  // 活性化マップの幅（深い層では7pxなど）が4の倍数でないことが多く、デフォルトの
  // UNPACK_ALIGNMENT(4)のままだと行の境界がずれてテクスチャが化ける（値が読めなくなる）
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
  gl.texImage3D(
    gl.TEXTURE_2D_ARRAY,
    0,
    gl.R8,
    width,
    height,
    heatmaps.length,
    0,
    gl.RED,
    gl.UNSIGNED_BYTE,
    data,
  )
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

  // --- 描画。バインドはここまでの数回だけで、グリッド全体を1回のdrawArraysで描き切る ---
  gl.viewport(0, 0, canvas.width, canvas.height)
  gl.clearColor(0, 0, 0, 0)
  gl.clear(gl.COLOR_BUFFER_BIT)
  gl.useProgram(state.program)
  gl.uniform2f(state.uResolution, canvas.width, canvas.height)
  gl.uniform1i(state.uBgTexture, 0)
  gl.uniform1i(state.uHeatTexture, 1)
  gl.bindVertexArray(state.vao)
  gl.drawArrays(gl.TRIANGLES, 0, heatmaps.length * 6)
  gl.bindVertexArray(null)
}

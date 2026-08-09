import * as tf from '@tensorflow/tfjs'
import { IMAGENET_CLASSES } from './data/imagenetClasses'
import { IMAGENET_CLASSES_JA } from './data/imagenetClassesJa'

const MODEL_URL =
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'

const INPUT_SIZE = 224

// 可視化対象とする層の種類。特徴マップを生成する層だけに絞る
// （BatchNormalization等の補助的な層は対象外）
const VISUALIZABLE_LAYER_TYPES = ['Conv2D', 'DepthwiseConv2D', 'Activation']

export type Prediction = { className: string; classNameJa: string; probability: number }
export type LayerActivation = { name: string; shape: number[] }

// スワイプで切り替える層。MobileNetV1(alpha 0.25)は13個のブロック（dw+pw）で構成される。
// 全56層（dw/pw/活性化前後すべて）を見せると細かすぎるため、各ブロックの出力（conv_pw_X_relu）
// と最初のconv1_reluのみに絞り、層が進むごとに特徴が複雑になる様子を追いやすくしている
export const OVERLAY_LAYERS = [
  'conv1_relu',
  'conv_pw_1_relu',
  'conv_pw_2_relu',
  'conv_pw_3_relu',
  'conv_pw_4_relu',
  'conv_pw_5_relu',
  'conv_pw_6_relu',
  'conv_pw_7_relu',
  'conv_pw_8_relu',
  'conv_pw_9_relu',
  'conv_pw_10_relu',
  'conv_pw_11_relu',
  'conv_pw_12_relu',
  'conv_pw_13_relu',
]

export function loadMobileNet(): Promise<tf.LayersModel> {
  return tf.loadLayersModel(MODEL_URL)
}

function preprocess(img: HTMLImageElement): tf.Tensor4D {
  return tf.tidy(() => {
    const pixels = tf.browser.fromPixels(img)
    const resized = tf.image.resizeBilinear(pixels, [INPUT_SIZE, INPUT_SIZE])
    const normalized = resized.toFloat().div(127.5).sub(1)
    return normalized.expandDims(0)
  })
}

export function classify(
  model: tf.LayersModel,
  img: HTMLImageElement,
  topK = 3,
): Prediction[] {
  return tf.tidy(() => {
    const input = preprocess(img)
    const output = model.predict(input) as tf.Tensor
    const probabilities = output.reshape([output.shape.at(-1) ?? 1000])
    const { values, indices } = tf.topk(probabilities, topK)
    const probabilityValues = values.dataSync()
    const classIndices = indices.dataSync()
    return Array.from(classIndices).map((classIndex, i) => ({
      className: IMAGENET_CLASSES[classIndex],
      classNameJa: IMAGENET_CLASSES_JA[classIndex],
      probability: probabilityValues[i],
    }))
  })
}

/** 可視化対象の層だけを出力する多出力モデルを組み立てる */
export function buildActivationModel(model: tf.LayersModel) {
  const targetLayers = model.layers.filter((layer) =>
    VISUALIZABLE_LAYER_TYPES.includes(layer.getClassName()),
  )
  const activationModel = tf.model({
    inputs: model.inputs,
    outputs: targetLayers.map((layer) => layer.output as tf.SymbolicTensor),
  })
  const layerNames = targetLayers.map((layer) => layer.name)
  return { activationModel, layerNames }
}

export function getActivationShapes(
  activationModel: tf.LayersModel,
  layerNames: string[],
  img: HTMLImageElement,
): LayerActivation[] {
  return tf.tidy(() => {
    const input = preprocess(img)
    const outputs = activationModel.predict(input) as tf.Tensor[]
    return outputs.map((tensor, i) => ({
      name: layerNames[i],
      shape: tensor.shape,
    }))
  })
}

export type ActivationHeatmap = { width: number; height: number; data: Float32Array }

/**
 * Grad-CAM計算用に、各層より後ろの処理だけを再現する「後半モデル」を層ごとに組み立てる。
 * MobileNetV1は分岐のない一本道の構造なので、対象層の出力形状を入力とする新しいモデルに
 * 後続の層を順番に繋ぎ直すだけで、そこから先の計算を再現できる
 */
export function buildGradCamModels(
  model: tf.LayersModel,
  layerNames: string[],
): Map<string, tf.LayersModel> {
  const gradCamModels = new Map<string, tf.LayersModel>()

  for (const layerName of layerNames) {
    const layerIndex = model.layers.findIndex((layer) => layer.name === layerName)
    if (layerIndex === -1) continue

    const outputShape = model.layers[layerIndex].outputShape as number[]
    const tailInput = tf.input({ shape: outputShape.slice(1) })
    let x: tf.SymbolicTensor = tailInput
    for (let i = layerIndex + 1; i < model.layers.length; i++) {
      x = model.layers[i].apply(x) as tf.SymbolicTensor
    }
    gradCamModels.set(layerName, tf.model({ inputs: tailInput, outputs: x }))
  }

  return gradCamModels
}

export type GradCamResult = {
  /** 予測クラスへの寄与度で重み付けした合成ヒートマップ（0〜1正規化） */
  heatmap: ActivationHeatmap
  /**
   * チャンネル（ニューロン）ごとの、合成ヒートマップ(cam)の強調領域への実際の寄与度
   * （重み×活性化を、正規化後のcamの値で空間的に重み付けして合計したもの。負の値は0扱い）。
   * チャンネル自身のピーク値だけで見ると、そのピークが合成画像の強調領域と別の場所にある場合に
   * 個別ニューロン表示の枠線と合成画像の見た目が食い違うため、「合成画像で実際に強調されている
   * 領域にどれだけ効いたか」を直接測ることで両者を整合させている
   */
  channelContributions: Float32Array
}

export type LayerAnalysis = {
  gradCam: GradCamResult
  /**
   * チャンネルごとの活性化マップ（個別ニューロン表示グリッド用）。
   * その層の全チャンネル共通の基準で0〜1正規化しているため、赤さの強さをチャンネル間で比較できる
   */
  channelHeatmaps: ActivationHeatmap[]
}

/**
 * OVERLAY_LAYERSの全層分を一括で解析する（Grad-CAMのオーバーレイ＋個別ニューロンの活性化マップ）。
 * 同じ画像なら結果は変わらないので、層の切り替えやグリッド表示のON/OFFのたびに計算し直すのではなく、
 * 画像が変わった時に一度だけ計算してキャッシュし、以降の表示切り替えは差し替えだけで済ませるのが狙い。
 *
 * - 元画像の活性化（activationModel.predict）は1回のみ計算し、各層はその結果を使い回す
 * - 予測クラスは画像1枚につき1つに決まるので、これも1回だけ求めて全層で使い回す
 *   （層ごとに求め直すと、対象層より後ろの計算＝浅い層ほどネットワークのほぼ全体を
 *   無駄に繰り返し計算することになるため）
 */
export function analyzeAllLayers(
  activationModel: tf.LayersModel,
  layerNames: string[],
  gradCamModels: Map<string, tf.LayersModel>,
  overlayLayers: string[],
  img: HTMLImageElement,
): Map<string, LayerAnalysis> {
  // tf.tidyの戻り値はTensorContainer制約があり、Mapをそのまま返せないため
  // Mapはtidyの外で組み立て、tidyの中身は計算とデータ抽出のみを行う
  const results = new Map<string, LayerAnalysis>()

  tf.tidy(() => {
    const input = preprocess(img)
    const outputs = activationModel.predict(input) as tf.Tensor4D[]

    // 予測クラスの特定は、最も計算コストの低い最終層(=対象層より後ろの処理が一番短い)で行う
    const deepestLayerName = overlayLayers[overlayLayers.length - 1]
    const deepestTailModel = gradCamModels.get(deepestLayerName)
    const deepestActivation = outputs[layerNames.indexOf(deepestLayerName)]
    const predictedScores = deepestTailModel?.predict(deepestActivation) as tf.Tensor
    const classIndex = predictedScores.reshape([-1]).argMax().dataSync()[0]

    for (const layerName of overlayLayers) {
      const tailModel = gradCamModels.get(layerName)
      if (!tailModel) continue
      const activation = outputs[layerNames.indexOf(layerName)]

      // --- Grad-CAM ---
      const gradFn = tf.grad(
        (x: tf.Tensor) =>
          (tailModel.predict(x) as tf.Tensor).reshape([-1]).gather([classIndex]).sum(),
      )
      const grads = gradFn(activation) as tf.Tensor4D
      const channelWeights = grads.mean([1, 2]) as tf.Tensor2D // [1, C]

      const weighted = activation.mul(channelWeights.reshape([1, 1, 1, -1]))
      const cam = (weighted.sum(-1).squeeze([0]) as tf.Tensor2D).relu()
      const camMin = cam.min()
      const camMax = cam.max()
      const normalizedCam = cam.sub(camMin).div(camMax.sub(camMin).add(1e-6))
      const [camHeight, camWidth] = normalizedCam.shape

      // 各チャンネルが「合成ヒートマップが実際に強調している領域」にどれだけ寄与したか。
      // チャンネル自身のピーク値だけを見ると、そのピークが合成画像の強調領域と別の場所にある場合に
      // 個別ニューロン表示の枠線と合成画像の見た目が食い違うため、正規化後のcamの値で
      // 空間的に重み付けして合計する（cam = Σ_c weighted_c なので、この重み付き合計は
      // 「合成画像の強調領域はこのチャンネルの寄与でどれだけ説明できるか」に対応する）
      const camWeights = normalizedCam.reshape([1, camHeight, camWidth, 1])
      const channelContributions = weighted.mul(camWeights).sum([1, 2]) as tf.Tensor2D // [1, C]

      // --- 個別ニューロン(チャンネル)ごとの活性化マップ ---
      // 正規化はチャンネルごとではなく、同じ層の全チャンネル・全ピクセルに共通の基準で行う。
      // チャンネルごとに正規化すると、実際の反応量が小さいチャンネルでも「その中で一番強い場所」が
      // 必ず真っ赤になってしまい、色の赤さが「どこで反応したか」しか表さず「どれだけ強く反応したか」
      // が失われる。層全体で共通の基準にすることで、赤さがその層内での相対的な反応の強さを表すようになる
      const channelActivation = activation.squeeze([0]) as tf.Tensor3D // [H, W, C]
      const layerMin = channelActivation.min()
      const layerMax = channelActivation.max()
      const normalizedChannels = channelActivation
        .sub(layerMin)
        .div(layerMax.sub(layerMin).add(1e-6))
      const [chHeight, chWidth, numChannels] = normalizedChannels.shape
      const flatChannels = normalizedChannels.dataSync() // [h][w][c]の順に並んだフラット配列

      const channelHeatmaps: ActivationHeatmap[] = []
      for (let c = 0; c < numChannels; c++) {
        const data = new Float32Array(chHeight * chWidth)
        for (let i = 0; i < chHeight * chWidth; i++) {
          data[i] = flatChannels[i * numChannels + c]
        }
        channelHeatmaps.push({ width: chWidth, height: chHeight, data })
      }

      results.set(layerName, {
        gradCam: {
          heatmap: {
            width: camWidth,
            height: camHeight,
            data: Float32Array.from(normalizedCam.dataSync()),
          },
          channelContributions: Float32Array.from(
            channelContributions.reshape([-1]).dataSync(),
          ),
        },
        channelHeatmaps,
      })
    }
  })

  return results
}

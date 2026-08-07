import * as tf from '@tensorflow/tfjs'
import { IMAGENET_CLASSES } from './data/imagenetClasses'

const MODEL_URL =
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'

const INPUT_SIZE = 224

// 可視化対象とする層の種類。特徴マップを生成する層だけに絞る
// （BatchNormalization等の補助的な層は対象外）
const VISUALIZABLE_LAYER_TYPES = ['Conv2D', 'DepthwiseConv2D', 'Activation']

export type Prediction = { className: string; probability: number }
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
  /** チャンネル（ニューロン）ごとの予測への寄与度。値が大きいほど予測に強く貢献している */
  channelWeights: Float32Array
}

/**
 * Grad-CAM: 予測スコアを対象層の活性化で微分し、チャンネルごとの重要度(寄与度)を求めたうえで
 * 活性化を重み付き合成する。単純平均と違い「予測にどれだけ効いたか」を反映した可視化になる
 */
export function getGradCam(
  activationModel: tf.LayersModel,
  layerNames: string[],
  gradCamModels: Map<string, tf.LayersModel>,
  layerName: string,
  img: HTMLImageElement,
): GradCamResult {
  const tailModel = gradCamModels.get(layerName)
  if (!tailModel) throw new Error(`GradCam model not found for layer: ${layerName}`)

  return tf.tidy(() => {
    const input = preprocess(img)
    const outputs = activationModel.predict(input) as tf.Tensor4D[]
    const index = layerNames.indexOf(layerName)
    const activation = outputs[index] // [1, H, W, C]

    const predictedScores = tailModel.predict(activation) as tf.Tensor
    const classIndex = predictedScores.reshape([-1]).argMax().dataSync()[0]

    const gradFn = tf.grad(
      (x: tf.Tensor) =>
        (tailModel.predict(x) as tf.Tensor).reshape([-1]).gather([classIndex]).sum(),
    )
    const grads = gradFn(activation) as tf.Tensor4D
    const channelWeights = grads.mean([1, 2]) as tf.Tensor2D // [1, C]

    const weighted = activation.mul(channelWeights.reshape([1, 1, 1, -1]))
    const cam = (weighted.sum(-1).squeeze([0]) as tf.Tensor2D).relu()
    const min = cam.min()
    const max = cam.max()
    const normalized = cam.sub(min).div(max.sub(min).add(1e-6))
    const [height, width] = normalized.shape

    return {
      heatmap: { width, height, data: Float32Array.from(normalized.dataSync()) },
      channelWeights: Float32Array.from(channelWeights.reshape([-1]).dataSync()),
    }
  })
}

/**
 * 指定した層の、チャンネル（ニューロン）ごとの活性化マップを個別に返す。
 * チャンネルごとに独立して0〜1へ正規化するため、活性化の強さが小さいチャンネルも見やすくなる
 */
export function getChannelHeatmaps(
  activationModel: tf.LayersModel,
  layerNames: string[],
  layerName: string,
  img: HTMLImageElement,
): ActivationHeatmap[] {
  return tf.tidy(() => {
    const input = preprocess(img)
    const outputs = activationModel.predict(input) as tf.Tensor4D[]
    const index = layerNames.indexOf(layerName)
    const activation = outputs[index].squeeze([0]) as tf.Tensor3D // [H, W, C]
    const min = activation.min([0, 1], true)
    const max = activation.max([0, 1], true)
    const normalized = activation.sub(min).div(max.sub(min).add(1e-6))
    const [height, width, numChannels] = normalized.shape
    const flat = normalized.dataSync() // [h][w][c]の順に並んだフラット配列

    const channels: ActivationHeatmap[] = []
    for (let c = 0; c < numChannels; c++) {
      const data = new Float32Array(height * width)
      for (let i = 0; i < height * width; i++) {
        data[i] = flat[i * numChannels + c]
      }
      channels.push({ width, height, data })
    }
    return channels
  })
}

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

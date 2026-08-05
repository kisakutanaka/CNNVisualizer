import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import type * as tf from '@tensorflow/tfjs'
import {
  loadMobileNet,
  classify,
  buildActivationModel,
  getActivationShapes,
  type Prediction,
} from './mobilenet'
import './App.css'

function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [activationModel, setActivationModel] = useState<tf.LayersModel | null>(null)
  const [layerNames, setLayerNames] = useState<string[]>([])
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [isClassifying, setIsClassifying] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    loadMobileNet().then((loadedModel) => {
      setModel(loadedModel)
      const built = buildActivationModel(loadedModel)
      setActivationModel(built.activationModel)
      setLayerNames(built.layerNames)
    })
  }, [])

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = () => {
      setPredictions([])
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

    if (activationModel) {
      const shapes = getActivationShapes(activationModel, layerNames, imgRef.current)
      console.log('layer activations:', shapes)
    }
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
        <div className="preview">
          <img
            ref={imgRef}
            src={imageUrl}
            alt="アップロードした画像のプレビュー"
            onLoad={handleImageLoad}
          />
        </div>
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

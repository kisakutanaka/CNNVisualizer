import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import '@tensorflow/tfjs'
import * as mobilenet from '@tensorflow-models/mobilenet'
import './App.css'

type Prediction = { className: string; probability: number }

function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [model, setModel] = useState<mobilenet.MobileNet | null>(null)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [isClassifying, setIsClassifying] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    mobilenet.load({ version: 2, alpha: 0.5 }).then(setModel)
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

  async function handleImageLoad() {
    if (!model || !imgRef.current) return
    setIsClassifying(true)
    const result = await model.classify(imgRef.current)
    setPredictions(result)
    setIsClassifying(false)
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

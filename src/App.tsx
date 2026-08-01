import { useState } from 'react'
import type { ChangeEvent } from 'react'
import './App.css'

function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null)

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = () => setImageUrl(reader.result as string)
    reader.readAsDataURL(file)
  }

  return (
    <main id="app">
      <h1>CNN Visualizer</h1>

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
          <img src={imageUrl} alt="アップロードした画像のプレビュー" />
        </div>
      )}
    </main>
  )
}

export default App

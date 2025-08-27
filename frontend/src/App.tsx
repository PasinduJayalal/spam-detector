import { useState } from 'react'
import Header from './components/Header'

type ModelName = "sms" | "email";


function App() {
  const [selectedModel, setSelectedModel] = useState<ModelName>("sms");
  return (
    <div>
      <Header header='Spam Detector' model={selectedModel} onModelChange={setSelectedModel} status='loading' />
      <main style={{ padding: 16 }}>
        Active model: <strong>{selectedModel.toUpperCase()}</strong>
      </main>
    </div>
  )
}

export default App

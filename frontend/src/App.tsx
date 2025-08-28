import { useState, useEffect } from 'react'
import Header from './components/Header'

type ModelName = "sms" | "email";
type HealthStatus = "loading" | "ok" | "down";



function App() {
  const [selectedModel, setSelectedModel] = useState<ModelName>("sms");
  const [health, setHealth] = useState<HealthStatus>("loading");

  useEffect(() => {
    const controller = new AbortController();

    async function ping() {
      try {
        const res = await fetch(`${import.meta.env.VITE_API_URL}/health`, {
          signal: controller.signal,
        });
        setHealth(res.ok ? "ok" : "down");
      } catch {
        setHealth("down");
      }
    }

    ping();
    return () => controller.abort();
  }, []);

  return (
    <div>
      <Header header='Spam Detector' model={selectedModel} onModelChange={setSelectedModel} status={health} />
      <main style={{ padding: 16 }}>
        Active model: <strong>{selectedModel.toUpperCase()}</strong>
      </main>
    </div>
  )
}

export default App

import { useState, useEffect } from 'react'
import Header from './components/Header'

type ModelName = "sms" | "email";
type HealthStatus = "loading" | "ok" | "down";

type MetaResponse = {
  max_text_len: number;
  model_names: ModelName[];
};

function App() {
  const [selectedModel, setSelectedModel] = useState<ModelName>("sms");
  const [health, setHealth] = useState<HealthStatus>("loading");
  const [meta, setMeta] = useState<MetaResponse | null>(null);
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

  useEffect(() => {
    const controller = new AbortController();

    async function fetchMeta() {
      try {
        const res = await fetch(`${import.meta.env.VITE_API_URL}/meta`, {
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        const max = Number(data?.max_text_len);
        const names = Array.isArray(data?.model_names) ? data.model_names : [];

        if (!Number.isFinite(max) || names.length === 0) throw new Error("Bad /meta");

        setMeta({ max_text_len: max, model_names: names });
      } catch {
        setMeta({ max_text_len: 4000, model_names: ["sms", "email"] });
      }
    }
    fetchMeta();
    return () => controller.abort();
  },[]);

  return (
    <div>
      <Header header='Spam Detector' model={selectedModel} onModelChange={setSelectedModel} status={health} />
      <main style={{ padding: 16 }}>
        Active model: <strong>{selectedModel.toUpperCase()}</strong>
        {meta ? (
          <>
            <p>
              Max text length: <strong>{meta.max_text_len}</strong>
            </p>
            <p>
              Models from /meta: <strong>{meta.model_names.join(", ")}</strong>
            </p>
          </>
        ) : (
          <p>Loading /meta…</p>
        )}
      </main>
    </div>
      )
    }

export default App

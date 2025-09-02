import { useState, useEffect } from 'react'
import Header from './components/Header'
import PredictorForm from "./components/PredictorForm";
import ResultPanel from './components/ResultPanel';
import 'react-toastify/dist/ReactToastify.css';
import { ToastContainer } from 'react-toastify';
import { toast } from 'react-toastify';


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

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ label?: string; score?: number } | null>(null);
  //const [error, setError] = useState<string | null>(null);



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
        //console.log("meta data:", data)

        if (!Number.isFinite(max) || names.length === 0) throw new Error("Bad /meta");

        setMeta({ max_text_len: max, model_names: names });
      } catch {
        setMeta({ max_text_len: 4000, model_names: ["sms", "email"] });
        //toast.error('Could not load server meta.', { toastId: 'meta-error' });
      }
    }
    fetchMeta();
    return () => controller.abort();
  }, []);


  async function handlePredict(text: string) {
    //setError(null);
    setResult(null);
    setLoading(true);

    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: selectedModel, text }),
      });
      if (!res.ok) {
        toast.error(`Prediction failed (HTTP ${res.status})`, { toastId: 'predict-error' });
        throw new Error(String(res.status));
      }
      const data = await res.json();
      const item = data?.items?.[0] ?? data ?? {};
      setResult({ label: item.label, score: Number(item.score) });
      toast.success('Prediction complete!', { toastId: 'predict-success' });
    } catch {
      //setError("Could not get a prediction.");
      toast.error('Could not get a prediction.', { toastId: 'predict-error' });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <Header header='Spam Detector' model={selectedModel} onModelChange={setSelectedModel} status={health} />
      <ToastContainer position="top-right" autoClose={4000} theme="light" />
      {!meta ? (
        <main className="mx-auto max-w-5xl p-4">Loading…</main>
      ) : (
        <>
          <PredictorForm
            model={selectedModel}
            maxTextLen={meta.max_text_len}
            loading={loading}
            onSubmit={handlePredict}
          />
          <ResultPanel model={selectedModel} result={result} />
        </>
      )}
    </div>
  )
}

export default App

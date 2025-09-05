from fastapi.testclient import TestClient

from api.app import app
import api.load_model as lm
import api.app as app_module

class FakeModel:
    def decision_function(self, X):
        out = []
        for text in X:
            t = (text or "").lower()
            if "neutral" in t:
                out.append(0.0)      
            elif ("free" in t) or ("win" in t):
                out.append(2.0)      
            else:
                out.append(-2.0)     
        return out

    def predict(self, X):
        return [1 if m >= 0 else 0 for m in self.decision_function(X)]

def make_client(monkeypatch):
    if hasattr(app_module, "load_all_models"):
        monkeypatch.setattr(app_module, "load_all_models", lambda: {})
    return TestClient(app)

def test_health_ok(monkeypatch):
    client = make_client(monkeypatch)
    r = client.get("/health")

    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_meta_shape(monkeypatch):
    client = make_client(monkeypatch)

    if hasattr(app_module, "available_models"):
        monkeypatch.setattr(app_module, "available_models", lambda: ["sms", "email"])
    else:
        monkeypatch.setattr(lm, "available_models", lambda: ["sms", "email"])

    r = client.get("/meta")
    if r.status_code != 200:
        print("META FAIL:", r.status_code, r.text)
    data = r.json()

    models = data.get("models") or data.get("model_names")

    assert r.status_code == 200
    assert isinstance(models, list) and set(models) == {"sms", "email"}
    assert "max_text_len" in data
    
def test_predict_single_happy(monkeypatch):
    client = make_client(monkeypatch)

    if hasattr(app_module, "get_model"):
        monkeypatch.setattr(app_module, "get_model", lambda name: FakeModel())
    else:
        monkeypatch.setattr(lm, "get_model", lambda name: FakeModel())

    r = client.post("/predict", json={"model": "sms", "text": "win free phone"})
    if r.status_code != 200:
        print("SINGLE FAIL:", r.status_code, r.text)
    obj = r.json()

    assert r.status_code == 200
    assert obj.get("model") in ("sms", "email")
    assert obj["label"] in ("Spam", "Not Spam", "Invalid")
    assert 0.0 <= float(obj["score"]) <= 1.0

def test_predict_batch_and_bad_inputs(monkeypatch):
    client = make_client(monkeypatch)
    if hasattr(app_module, "get_model"):
        monkeypatch.setattr(app_module, "get_model", lambda name: FakeModel())
    else:
        monkeypatch.setattr(lm, "get_model", lambda name: FakeModel())

    
    r = client.post("/predict", json={"model": "email", "texts": ["", "hello", "win big"]})
    if r.status_code != 200:
        print("BATCH FAIL:", r.status_code, r.text)
    data = r.json()
    assert r.status_code == 200
    assert "results" in data and len(data["results"]) == 3
    assert data["results"][0]["label"] == "Invalid" and data["results"][0]["score"] == 0.0
    assert 0.0 <= data["results"][1]["score"] <= 1.0

    
    r = client.post("/predict", json={"model": "sms", "text": "x", "texts": ["y"]})
    assert r.status_code == 400

    r = client.post("/predict", json={"model": "sms", "text": "   "})
    assert r.status_code == 400

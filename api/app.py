# api/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from typing import Union

from api import config
from api.load_model import load_all_models, get_model , available_models
from api.schemas import PredictIn, PredictOut, PredictBatchOut
from src.infer import predict_one_with_score, predict_batch_with_score



app = FastAPI()


ALLOWED_ORIGINS = config.ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("Loading models...")
    load = load_all_models()
    print(f"Loaded models: {list(load.keys())}")

@app.get("/")
def god():
    return {"status": "God is with you! Pasindu! Dont you dare give up!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/your_world")
def your_world():
    return {"status": "The world is yours!"}

@app.get("/meta")
def meta():
    return {
        "models": available_models(),
        "max_text_len": config.MAX_TEXT_LEN,
        "origins": config.ALLOWED_ORIGINS,
        }

@app.post("/predict", response_model=Union[PredictOut, PredictBatchOut])
def predict(payload: PredictIn):
    try:
        model = get_model(payload.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if payload.text is None and payload.texts is None:
        # return {"status": 400, "message": "Either 'text' or 'texts' must be provided"}
        raise HTTPException(status_code=400, detail="Either 'text' or 'texts' must be provided") 
    elif payload.text is not None and payload.texts is not None:
        # return {"status": 400, "message": "Provide only one of 'text' or 'texts'"}
        raise HTTPException(status_code=400, detail="Provide only one of 'text' or 'texts'")
    elif payload.text is not None and payload.text.strip()=="":
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    else:
        if payload.text:
            label, score = predict_one_with_score(model, payload.text)
            return PredictOut(label=label, score=score, model=payload.model, text=payload.text)
        elif payload.texts:
            results = predict_batch_with_score(model, payload.texts)
            return PredictBatchOut(results=[PredictOut(**res, model=payload.model) for res in results])
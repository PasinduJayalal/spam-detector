import joblib
import numpy as np


def load_sms_model():
    sms = joblib.load('models/sms_pipeline.pkl')
    return sms

def load_email_model():
    email = joblib.load('models/email_pipeline.pkl')
    return email

def predict_one(model, text: str) -> str:
    
    text = text.strip()
    
    if not text:
        raise ValueError("Text cannot be empty")
    
    result = model.predict([text])[0]
    
    return "Spam" if result == 1 else "Not Spam"

def predict_one_with_score(model, text: str):
    
    text = text.strip()
    
    if not text:
        raise ValueError("Text cannot be empty")

    margin = model.decision_function([text])[0]
    
    score = 1 / (1 + math.exp(-margin))
    
    label = "Spam" if margin >= 0 else "Not Spam"

    return label, score


def predict_batch_with_score(model, texts: list[str]):
    if not texts:
        raise ValueError("Text list cannot be empty")
    
    results = []
    for text in texts:
        text_clean = text.strip()
        if not text_clean:
            results.append({"label": "Invalid", "score": 0.0, "text": text})
            continue

        margin = model.decision_function([text_clean])[0]
        score = 1 / (1 + np.exp(-margin))
        label = "Spam" if margin >= 0 else "Not Spam"

        results.append({"label": label, "score": float(score), "text": text})
        
    return results

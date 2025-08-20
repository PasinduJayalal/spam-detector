import joblib


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

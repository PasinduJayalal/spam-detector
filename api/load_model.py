from src.infer import load_sms_model, load_email_model

MODEL = {}
model_types = ["sms", "email"]

def available_models():
    return list(MODEL.keys())

def load_model(model_name: str):
    model_name = model_name.lower().strip()
    if model_name == "sms":
        MODEL[model_name] = load_sms_model()
    elif model_name == "email":
        MODEL[model_name] = load_email_model()
    else:
        raise ValueError(f"Model {model_name} not recognized")
    
    return MODEL[model_name]

def load_all_models():
    for model_name in model_types:
        try:
            load_model(model_name)
            print(f"{model_name} model loaded")
        except Exception as e:
            print(f"Could not load {model_name} model: {e}")
    return MODEL

def get_model(model_name: str):
    model_name = model_name.lower().strip()
    if model_name not in model_types:
        raise ValueError("Model must be 'sms' or 'email'")
    if model_name not in MODEL:
        raise ValueError(f"Model {model_name} is not loaded")
    
    return MODEL[model_name]
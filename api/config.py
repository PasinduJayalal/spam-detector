from dotenv import load_dotenv
import os


load_dotenv()


ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", 4000))
MODEL_SMS_PATH = os.getenv("MODEL_SMS_PATH", "models/sms_pipeline.pkl")
MODEL_EMAIL_PATH = os.getenv("MODEL_EMAIL_PATH", "models/email_pipeline.pkl")

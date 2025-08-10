from src.utils import load_dataset

# Test SMS dataset
X_sms, y_sms = load_dataset(r"D:\Professional\SpamDetector\data\SMSSpamCollection")
print("SMS:", len(X_sms), "messages,", "Labels:", set(y_sms))

# Test Email dataset
X_email, y_email = load_dataset(r"D:\Professional\SpamDetector\data\spam_ham_dataset.csv")
print("Email:", len(X_email), "messages,", "Labels:", set(y_email))

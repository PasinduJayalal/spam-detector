# 📧 SMS & Email Spam Detector

A machine learning project to classify **SMS** and **Email** messages as **Spam** or **Not Spam**.  
- **ML Stack**: scikit-learn, spaCy, joblib
- **Backend**: FastAPI, Uvicorn, Pydantic, python-dotenv
- **Frontend**: React, TypeScript, TailwindCSS (via Vite)

This project explores preprocessing, TF-IDF feature extraction, and classification using **Linear Support Vector Machines (LinearSVC)**, then serves the models via an API and web UI.

---

## 📂 Project Structure (Currently)

```
├── data/ # Raw and demo datasets
│ ├── SMSspam2.csv
│ ├── spam_ham_dataset.csv
│ ├── demo_sms.txt
│ └── demo_email.txt
│
├── models/ # Saved trained models (ignored by Git)
│ ├── sms_pipeline.pkl
│ └── email_pipeline.pkl
│
├── reports/ # Markdown reports with results
│ ├── sms_results.md
│ └── email_results.md
│
├── src/ # Source code
│ ├── utils.py
│ ├── pipelines.py
│ ├── train_sms.py
│ ├── train_email.py
│ └── predict.py
│
├── api/                  # FastAPI backend
│   ├── app.py            # FastAPI app (routes, middleware)
│   ├── schemas.py        # Pydantic models (request/response)
│   ├── load_model.py     # Model loading and caching
│   └── lconfig.py         # Loads settings from .env
├── frontend/             # React + TypeScript + Tailwind app
│   ├── src/              # Components, API helpers, types
│   └── .env              # Frontend API URL (VITE_API_URL)
│
├── tests/ # Smoke & CLI tests
│  └── smoke/
│    ├── test_loader.py
│    ├── test_pipelines.py
│    └── test_predict.py
│
├── requirements.txt      # ML/training dependencies
├── api/requirements.txt  # Backend dependencies
├── .env                  # Backend settings (ignored by Git)
└── README.md
```

---

## ⚙️ Installation  (Backend)

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/spam-detector.git
   cd spam-detector```

2. Create and activate a virtual environment:

    ```python -m venv venv
    source venv/bin/activate   # bash
    venv\Scripts\activate      # powershell```

3. Install dependencies:

    ```pip install -r requirements.txt```
    ```pip install -r api/requirements.txt```
    
4. Create a ```.env``` file (backend root) with:
``` ALLOWED_ORIGINS=http://localhost:5173```
```MAX_TEXT_LEN=4000```
```MODEL_SMS_PATH=models/sms_pipeline.pkl```
```MODEL_EMAIL_PATH=models/email_pipeline.pkl ```

5. Run the API:
```python -m uvicorn api.app:app --reload --port 8000```
---
## ⚙️ Installation (Frontend)
1. Go into the frontend folder:
``` cd frontend ```
2. Install Node dependencies:
``` npm install ```
3. Create ``frontend/.env``:
```VITE_API_URL=http://127.0.0.1:8000```
4. Run the dev server:
``` npm run dev```
---
## 📊 Datasets

### SMS Spam Collection

- **Source:** [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Description:** A collection of 5,572 SMS messages labeled as ham (not spam) or spam.

### Email Spam Dataset
- **Source:** [Kaggle - Email Spam Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)
- **Description:** A dataset of 5,171 emails labeled as ham or spam with text content.
---

## 🏋️ Training

### Train SMS spam model:
```python -m src.train_sms```

### Train Email spam model:
```python -m src.train_email```

### Models will be saved under `models/.`

---

## 🔍 Prediction
### Predict a single SMS:
```python -m src.predict --model sms --text "Congratulations, you won free tickets!"```
### Predict from a file of SMS messages:
```python -m src.predict --model sms --file data/demo_sms.txt```
### Predict a single Email:
```python -m src.predict --model email --text "This is a test email"```
### Predict from a file of Emails:
```python -m src.predict --model sms --file data/demo_email.txt```

---
## 📑 Results

- SMS results: `reports/sms_results.md`
- Email results: `reports/email_results.md`

Both include **classification reports, confusion matrices, and cross-validation scores**

---
## 🔍 API Endpoints
- ``GET /`` → Welcome message
- ``GET /health`` → {`` "status": "ok" }``
- ``GET /meta`` → Info about models, max text length, CORS origins
- ``POST /predict`` → Predict single or batch messages

##### Example single requests: 

```{ "model": "sms", "text": "Congratulations, you won free tickets!" }```

##### Example batch request:

```{ "model": "sms", "texts": ["Win a FREE iPhone!!!", "Hey, are we on for 4pm?"] }```

---

## ✅ Testing
1. Install `pytest`
``` pip install pytest```
2. Run smoke tests:
```pytest tests/smoke -v```

---

## 👨‍💻 Author
- Pasindu Jayalal – [GitHub](https://github.com/PasinduJayalal)







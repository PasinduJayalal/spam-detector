# 📧 SMS & Email Spam Detector

A **full-stack machine learning project** to classify SMS and Email messages as **Spam** or **Not Spam**.  
Built with **Python (scikit-learn, spaCy, FastAPI)** on the backend and **React + TypeScript + TailwindCSS** on the frontend.

---

## 🛠️ Tech Stack

- **Python**: scikit-learn, spaCy, joblib, FastAPI, pytest
- **Frontend**: React, TypeScript, TailwindCSS, Vite, Vitest, RTL, MSW
- **Other**: GitHub Actions (CI), CSV datasets (Kaggle)

---

## 🚀 Features

- **Text Preprocessing (spaCy + custom rules):**
  - URL, email, and whitespace removal
  - Tokenization, lemmatization, and stopword filtering
  - Support for both SMS and Email datasets

- **ML Pipelines (scikit-learn):**
  - Word-level and character-level **TF-IDF vectorizers**
  - Classification with **LinearSVC** (best-performing model)
  - Tuned n-gram ranges (1–2 for words, 3–5 for chars)

- **Training & Evaluation:**
  - Stratified train/test splits
  - Evaluation with classification reports, F1-score, confusion matrices
  - Pipelines serialized with **joblib** for deployment

- **Backend (FastAPI):**
  - Endpoints: `/health`, `/meta`, `/predict`
  - Pydantic models for validation
  - Logging with client IP, user-agent, response time
  - CORS enabled for frontend ↔ backend communication

- **Frontend (React + TypeScript + TailwindCSS):**
  - Model selector (SMS / Email)
  - Input box for messages
  - Predict button with loading state
  - Results panel with spam label & probability score
  - **ScoreBar** with color-coded spam probability

- **Testing:**
  - Backend unit tests with **pytest** (utils, pipelines, infer, API)
  - Frontend tests with **Vitest + React Testing Library + MSW**
  - Mocked models for fast deterministic inference
  - GitHub Actions planned for CI


---

## 📂 Project Structure (Currently)

```
├── api/ # FastAPI backend
│ ├── app.py
│ ├── load_model.py
│ ├── schemas.py
│ ├── config.py
│ └── ...
├── src/ # ML training code
│ ├── utils.py
│ ├── pipelines.py
│ ├── train_sms.py
│ ├── train_email.py
│ ├── infer.py
│ └── predict.py
├── frontend/ # React + Vite frontend
│ ├── src/
│ │ ├── App.tsx
│ │ ├── main.tsx
│ │ ├── components/
│ │ │ ├── Header.tsx
│ │ │ ├── PredictorForm.tsx
│ │ │ ├── ResultPanel.tsx
│ │ │ └── ScoreBar.tsx
│ │ └── tests/ # Vitest + RTL tests
├── tests/ # Backend pytest unit tests
│ ├── unit/
│ │ ├── test_utils.py
│ │ ├── test_pipelines_word.py
│ │ ├── test_pipelines_char.py
│ │ ├── test_infer.py
│ │ └── test_api.py
├── models/ # Serialized ML pipelines (.pkl)
├── data/ # Datasets (CSV)
│ ├── SMSspam2.csv
│ ├── spam_ham_dataset.csv
│ └── golden_sms.csv / golden_email.csv (unit test sets)
├── reports/ # Evaluation results
│ ├── sms_results.md
│ └── email_results.md
├── requirements.txt # Python dependencies
├── README.md
└── .gitignore
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
🧪 Running Tests

### Backend (pytest)
``` pytest tests/unit -v ```

### Frontend (Vitest + RTL)

```cd frontend```
```npm test```

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


## 👨‍💻 Author
- Pasindu Jayalal – [GitHub](https://github.com/PasinduJayalal)







# 📧 SMS & Email Spam Detector

A machine learning project to classify **SMS** and **Email** messages as **Spam** or **Not Spam**.  
Built with **scikit-learn**, **spaCy**, and **Python**, this project explores text preprocessing, TF-IDF feature extraction, and classification using **Linear Support Vector Machines (LinearSVC)**.

---

## 📂 Project Structure (Currently)

```
├── data/ # Raw and demo datasets
│ ├── SMSspam2.csv
│ ├── spam_ham_dataset.csv
│ ├── demo_sms.txt
│ └── demo_email.txt
├── models/ # Saved trained models (ignored by Git)
│ ├── sms_pipeline.pkl
│ └── email_pipeline.pkl
├── reports/ # Markdown reports with results
│ ├── sms_results.md
│ └── email_results.md
├── src/ # Source code
│ ├── utils.py
│ ├── pipelines.py
│ ├── train_sms.py
│ ├── train_email.py
│ └── predict.py
└── tests/ # Smoke & CLI tests
  └── smoke/
    ├── test_loader.py
    ├── test_pipelines.py
    └── test_predict.py
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/spam-detector.git
   cd spam-detector```

2. Create and activate a virtual environment:

    ```python -m venv .env
    source .env/bin/activate   # bash
    .env\Scripts\activate      # powershell```

3. Install dependencies:

    ```pip install -r requirements.txt```
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

Both include **classification reports, confusion matrices, and cross-validation scores**.

---

## ✅ Testing
1. Install `pytest`
``` pip install pytest```
2. Run smoke tests:
```pytest tests/smoke -v```

---

## 👨‍💻 Author
- Pasindu Jayalal – [GitHub](https://github.com/PasinduJayalal)







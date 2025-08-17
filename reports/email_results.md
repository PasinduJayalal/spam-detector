# Email Spam Detection — Word 1–2 TF-IDF + LinearSVC

**Date:** 2025-08-15   
**Dataset:** spam_ham_dataset.csv  
**Vectorizer:** TfidfVectorizer(analyzer='word', ngram_range=(1, 2), lowercase=False)  
**Model:** LinearSVC(class_weight='balanced', C=1.0, random_state=42)

---

## Cross-Validation (Train Split)
- 5-fold Stratified CV (shuffle=True, random_state=42)  
- **F1 scores:** [0.9956, 0.9716, 0.9846, 0.9845, 0.9760]  
- **Mean ± Std:** 0.9825 ± 0.0083

---

## Holdout Test (25% split)

**Classification Report:**

              precision    recall  f1-score   support

    Not Spam       1.00      0.99      0.99       918
        Spam       0.97      0.99      0.98       375

    accuracy                           0.99      1293
   macro avg       0.98      0.99      0.99      1293
weighted avg       0.99      0.99      0.99      1293


**Confusion Matrix:**

                  Predicted: Not Spam  Predicted: Spam
Actual: Not Spam                  907               11
Actual: Spam                        2              373

---

## Observations
- Word 1–2 n-grams gave excellent results for email spam detection.  
- Achieved **98% F1 for spam** and **99% overall accuracy**.  
- Very few errors: only 2 spam missed and 11 hams flagged incorrectly.  
- Outperformed char n-grams slightly in this dataset.
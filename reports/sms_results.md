# SMS Spam Detection — Char 3–5 TF-IDF + LinearSVC

**Date:** 2025-08-14   
**Dataset:** SMSspam2.csv  
**Vectorizer:** TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=2, lowercase=False)  
**Model:** LinearSVC(class_weight='balanced', C=1.0, random_state=42)

---

## Cross-Validation (Train Split)
- 5-fold Stratified CV (shuffle=True, random_state=42)  
- **F1 scores:** [0.9252, 0.9677, 0.9541, 0.9537, 0.9677]  
- **Mean ± Std:** 0.9537 ± 0.0155

---

## Holdout Test (25% split)

**Classification Report:**

                precision    recall  f1-score   support

    Not Spam       0.99      1.00      1.00      1206
        Spam       0.99      0.95      0.97       187

    accuracy                           0.99      1393
   macro avg       0.99      0.98      0.98      1393
weighted avg       0.99      0.99      0.99      1393


**Confusion Matrix:**

                  Predicted: Not Spam  Predicted: Spam
Actual: Not Spam                 1205                1
Actual: Spam                        9              178

---

## Observations
- Char 3–5 n-grams worked well, catching spam tricks like misspellings.  
- Model achieved **97% F1 for spam** and **99% overall accuracy**.  
- Slightly more false negatives (9 spam missed), but very few false positive (1 ham misclassified).



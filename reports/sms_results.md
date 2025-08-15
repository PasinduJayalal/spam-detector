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

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
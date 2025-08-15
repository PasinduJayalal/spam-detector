from src.utils import load_dataset, clean_text_list, preprocess_text_list
from src.pipelines import make_pipeline, make_char_pipeline

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("Training Email spam baseline (LinearSVC) ...")
    X, y = load_dataset("data/spam_ham_dataset.csv")
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # clf = Pipeline([
    #     ("preprocess", make_char_pipeline(clean_text_list, ngram_range=(3,5), min_df=2)),
    #     ("model", LinearSVC(class_weight="balanced", C=1.0, random_state=42)),
    # ])
    clf = Pipeline([
        ('preprocess', make_pipeline(clean_text_list, preprocess_text_list)),
        ('model', LinearSVC(class_weight="balanced", C=1.0 ,random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual: Not Spam', 'Actual: Spam'], columns=['Predicted: Not Spam', 'Predicted: Spam'])
    print(cm_df)
    
    print("F1 (5-fold CV):", np.round(scores, 4))
    print("Mean ± Std:", np.round(scores.mean(), 4), "±", np.round(scores.std(), 4))
    
    joblib.dump(clf, "models/email_pipeline.pkl")
    print("Saved model → models/email_pipeline.pkl")


if __name__ == "__main__":
    main()
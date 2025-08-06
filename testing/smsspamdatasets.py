import pandas as pd
import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix


datasets = pd.read_csv(
    "SMSSpamCollection", sep="\t", header=None, names=["label", "text"]
)

datasets["label"] = datasets["label"].map({"ham": 0, "spam": 1})

# print(datasets.head())
text = datasets["text"].tolist()
labels = np.array(datasets["label"].tolist())


nlp = spacy.load("en_core_web_sm")


def clean_text(text, lower=True, remove_urls=True, remove_emails=True):
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)

    if remove_emails:
        text = re.sub(r"\b[\w\.-]+?@\w+?\.\w{2,4}\b", "", text)

    if lower:
        text = text.lower()

    # Optional: remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def tfidf_vectorize(corpus):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    return X , vectorizer


cleaned_texts = [clean_text(text) for text in text]
preprocessed_texts = [preprocess(text) for text in cleaned_texts]
converted_texts, tfidf  = tfidf_vectorize(preprocessed_texts)

model = {
    "svm": {
        "model": svm.SVC(gamma="auto"),
        "params": {"C": [1, 10, 20], "kernel": ["rbf", "linear"]},
    },
    "rf": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [1, 5, 10]},
    },
    "lr": {
        "model": LogisticRegression(solver="liblinear", class_weight="balanced"),
        "params": {"C": [1, 5, 10]},
    },
}


# print(converted_texts.shape)
def get_model_predictions(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


stratified_kfold = StratifiedKFold(n_splits=3)


model_ytest = []
scores = []
model_predictions = {name: [] for name in model}
# lr = []


# for train, test in stratified_kfold.split(converted_texts, labels):
#     X_train, X_test = converted_texts[train], converted_texts[test]
#     y_train, y_test = labels[train], labels[test]

#     lr.extend(
#         get_model_predictions(
#             LogisticRegression(solver="liblinear", class_weight="balanced"),
#             X_train,
#             X_test,
#             y_train,
#             y_test,
#         )
#     )
#     ytest.extend(y_test)


for train, test in stratified_kfold.split(converted_texts, labels):
    X_train, X_test = converted_texts[train], converted_texts[test]
    y_train, y_test = labels[train], labels[test]
    
    model_ytest.extend(y_test)
    
    for model_name, mp in model.items():
        clf = GridSearchCV(mp["model"], mp["params"], cv=5, scoring='f1')
        clf.fit(X_train, y_train)
        clf_predictions = clf.predict(X_test)
        model_predictions[model_name].extend(clf_predictions)
        scores.append(
            {
                "model": model_name,
                "best_score": clf.best_score_,
                "best_params": clf.best_params_,
            }
        )


for model_name, preds in model_predictions.items():
    print(f"\n=== Evaluation for {model_name.upper()} ===")
    print(classification_report(model_ytest, preds, target_names=["Not Spam", "Spam"]))
    cm = confusion_matrix(model_ytest, preds)
    print(pd.DataFrame(
        cm,
        index=["Actual: Not Spam", "Actual: Spam"],
        columns=["Predicted: Not Spam", "Predicted: Spam"]
    ))







# print(classification_report(ytest, predictions, target_names=["Not Spam", "Spam"]))
# cm = confusion_matrix(ytest, predictions)
# cm_df = pd.DataFrame(
#     cm,
#     index=["Actual: Not Spam", "Actual: Spam"],
#     columns=["Predicted: Not Spam", "Predicted: Spam"],
# )
df = pd.DataFrame(scores, columns=["model", "best_score", "best_params"])
print(df) 
# print(cm_df)

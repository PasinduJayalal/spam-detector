import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("larger_dummy_spam_data.csv")
corpus = df["text"].tolist()
lists = df["label"].tolist()


def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def TfIdfVectorizer(corpus):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    return X


preprocessed_corpus = [preprocess_text(text) for text in corpus]
converted_corpus = TfIdfVectorizer(preprocessed_corpus)


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
        "model": LogisticRegression(solver="liblinear"),
        "params": {"C": [1, 5, 10]},
    },
}


scores = []


# for model_name, mp in model.items():
#     clf = RandomizedSearchCV(mp["model"], mp["params"], cv=5)
#     clf.fit(converted_corpus, lists)
#     scores.append({
#         "model": model_name,
#         "best_score": clf.best_score_,
#         "best_params": clf.best_params_,
#     })

for model_name, mp in model.items():
    clf = GridSearchCV(mp["model"], mp["params"], cv=5)
    clf.fit(converted_corpus, lists)
    scores.append({
        "model": model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_,
    })
    

df = pd.DataFrame(scores, columns=["model", "best_score", "best_params"])
print(df) 

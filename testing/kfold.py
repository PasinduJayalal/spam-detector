import spacy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


npl = spacy.load("en_core_web_sm")
df = pd.read_csv('larger_dummy_spam_data.csv')
corpus = df['text'].tolist()
labels = df['label'].tolist()


def preprocess_text(text):
    doc = npl(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def TfIdfVectorizer(corpus):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    return X

preprocessed_corpus = [preprocess_text(text) for text in corpus]

converted_corpus = TfIdfVectorizer(preprocessed_corpus)

X_train, X_test, y_train, y_test = train_test_split(
    converted_corpus, labels, test_size=0.25)

# lr = LogisticRegression(solver='liblinear',multi_class='ovr')
# lr.fit(X_train, y_train)
# lr.score(X_test, y_test)

# nb = MultinomialNB()
# nb.fit(X_train, y_train)
# nb.score(X_test, y_test)

# svc = SVC(kernel='linear')
# svc.fit(X_train, y_train)
# svc.score(X_test, y_test)


# print("Logistic Regression Accuracy:", lr.score(X_test, y_test))
# print("Naive Bayes Accuracy:", nb.score(X_test, y_test))
# print("SVC Accuracy:", svc.score(X_test, y_test))


def get_score(model, X_test, y_test, X_train, y_train):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# lr = get_score(LogisticRegression(solver='liblinear', multi_class='ovr'), X_test, y_test, X_train, y_train) 
# nb = get_score(MultinomialNB(), X_test, y_test, X_train, y_train)
# svc = get_score(SVC(kernel='linear'), X_test, y_test, X_train,y_train)

# print("Logistic Regression Accuracy:", lr)
# print("Naive Bayes Accuracy:", nb)
# print("SVC Accuracy:", svc)

kfold = StratifiedKFold(n_splits=3)

lr = []
nb =[]
svc = []

labels = np.array(labels)

for train_index, test_index in kfold.split(converted_corpus, labels):
    X_train, X_test, y_train, y_test = converted_corpus[train_index], converted_corpus[test_index], labels[train_index], labels[test_index]
    lr.append(get_score(LogisticRegression(solver='liblinear', multi_class='ovr'), X_test, y_test, X_train, y_train))
    nb.append(get_score(MultinomialNB(), X_test, y_test, X_train, y_train))
    svc.append(get_score(SVC(kernel='linear'), X_test, y_test, X_train, y_train))
    
print("StratifiedKFold Results:")
print("Logistic Regression Fold Accuracies:")
for i, score in enumerate(lr, 1):
    print(f"  Fold {i}: {score:.2f}")
print("  Average:", np.average(lr), "\n")

print("Naive Bayes Fold Accuracies:")
for i, score in enumerate(nb, 1):
    print(f"  Fold {i}: {score:.2f}")
print("  Average:", np.average(nb), "\n")

print("SVC Fold Accuracies:")
for i, score in enumerate(svc, 1):
    print(f"  Fold {i}: {score:.2f}")
print("  Average:", np.average(svc))


lr_cross_val_score= cross_val_score(LogisticRegression(solver='liblinear'), converted_corpus, labels, cv=3)
nb_croos_val_score = cross_val_score(MultinomialNB(), converted_corpus, labels, cv=3)
svc_cross_val_score = cross_val_score(SVC(kernel='linear'), converted_corpus, labels, cv=3)

print("\nCross-Validation Scores:")
for i, score in enumerate(lr_cross_val_score, 1):
    print(f"Logistic Regression Fold {i} Score: {score:.2f}")
print("Average Logistic Regression Score:", np.mean(lr_cross_val_score))

for i, score in enumerate(nb_croos_val_score, 1):
    print(f"Naive Bayes Fold {i} Score: {score:.2f}")
print("Average Naive Bayes Score:", np.mean(nb_croos_val_score))

for i, score in enumerate(svc_cross_val_score, 1):
    print(f"SVC Fold {i} Score: {score:.2f}")
print("Average SVC Score:", np.mean(svc_cross_val_score))





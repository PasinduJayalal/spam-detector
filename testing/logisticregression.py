import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix



npl = spacy.load("en_core_web_sm")
df = pd.read_csv('larger_dummy_spam_data.csv')
corpus = df['text'].tolist()
labels = df['label'].tolist()


def preprocess_text(text):
    doc = npl(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

preprocessed_corpus = [preprocess_text(text) for text in corpus]

X_train, X_test, y_train, y_test = train_test_split(
    preprocessed_corpus, labels, test_size=0.25, random_state=42, stratify=labels
)

clf = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('Logistic Regression', LogisticRegression(max_iter=1000))  # Using Logistic Regression 
])


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual: Not Spam', 'Actual: Spam'], columns=['Predicted: Not Spam', 'Predicted: Spam'])
print(cm_df)
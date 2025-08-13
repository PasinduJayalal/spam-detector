import spacy
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix



# df = pd.read_csv(
#     "SMSSpamCollection", sep="\t", header=None, names=["label", "text"]
# )

df = pd.read_csv("spam2.csv", encoding="latin-1")
df = df[['v1', 'v2']]

df = df.rename(columns={'v1': 'label', 'v2': 'text'})

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

nlp = spacy.load("en_core_web_sm")

corpus = df['text'].tolist()
labels = np.array(df["label"].tolist())


def clean_text(text, lower=True, remove_urls=True, remove_emails=True):
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)

    if remove_emails:
        text = re.sub(r'\b[\w\.-]+?@\w+?\.\w{2,4}\b', '', text)

    if lower:
        text = text.lower()

    # Optional: remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

cleaned_texts = [clean_text(text) for text in corpus]
preprocessed_texts = [preprocess_text(text) for text in cleaned_texts]

X_train, X_test, y_train, y_test = train_test_split(
    preprocessed_texts, labels, test_size=0.25, random_state=42, stratify=labels
)

clf = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('Support Vector Machine', SVC(class_weight="balanced", kernel='linear', C=1.0, gamma="auto"))   
])


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))     
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual: Not Spam', 'Actual: Spam'], columns=['Predicted: Not Spam', 'Predicted: Spam'])
print(cm_df)

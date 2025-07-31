import pandas as pd
import spacy
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix



nlp = spacy.load("en_core_web_sm")
df = pd.read_csv('larger_dummy_spam_data.csv')
corpus = df['text'].tolist()  
labels = df['label'].tolist()


def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

preprocess_texted_corpus = [preprocess_text(text) for text in corpus]


X_train, X_test, y_train, y_test = train_test_split(
    preprocess_texted_corpus, labels, test_size=0.25, random_state=42, stratify=labels
)

x = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('SVC', SVC(kernel='linear')) 
])

x.fit(X_train, y_train)
y_pred = x.predict(X_test)


print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cm_df = pd.DataFrame(cm, index=['Actual: Not Spam', 'Actual: Spam'], columns=['Predicted: Not Spam', 'Predicted: Spam'])
print(cm_df)        

import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF model
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


nlp = spacy.load("en_core_web_sm")

df = pd.read_csv('larger_dummy_spam_data.csv')  # Assuming you have

corpus = df['text'].tolist()  # Replace 'text' with the actual column name containing the text data
labels = df['label'].tolist()  # Replace 'label' with the actual column name


def preprocess_text(text):
    doc= nlp(text)
    
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    return " ".join(tokens)

preprocessed_corpus = [preprocess_text(text) for text in corpus]


X_train, X_test, y_train, y_test = train_test_split(
    preprocessed_corpus, labels, test_size=0.25, random_state=42, stratify=labels
)

# Create a pipeline with TF-IDF vectorization and Naive Bayes classifier
clf = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer(ngram_range=(1, 1))),  # Using TF-IDF vectorization
    ('Multinomial NB', MultinomialNB())
])
# Train the model
clf.fit(X_train, y_train)  


# Predict
y_pred = clf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))     


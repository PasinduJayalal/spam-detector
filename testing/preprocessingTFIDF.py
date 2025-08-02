import spacy
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF model
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


nlp = spacy.load("en_core_web_sm")
# nlp.Defaults.stop_words.add("my_new_stopword")
# nlp.Defaults.stop_words |= {"my_new_stopword1","my_new_stopword2",}
# nlp.Defaults.stop_words.remove("whatever")
# nlp.Defaults.stop_words -= {"whatever", "whenever"}

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
    ('vectorizer_tfidf', TfidfVectorizer(ngram_range=(1, 2))),  # Using TF-IDF vectorization
    ('Multinomial NB', MultinomialNB())
])
# Train the model
clf.fit(X_train, y_train)  


# Predict
y_pred = clf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))     
# print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual: Not Spam', 'Actual: Spam'], columns=['Predicted: Not Spam', 'Predicted: Spam'])
print(cm_df)



# plt.figure(figsize=(6,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # Save the model
# joblib.dump(clf, 'spam_classifier_tfidf.pkl')

# mj = joblib.load('spam_classifier_tfidf.pkl')

# mj_pred = mj.predict(X_test)
# print(classification_report(y_test, mj_pred, target_names=['Not Spam', 'Spam']))





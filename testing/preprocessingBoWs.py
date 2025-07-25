import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # For Bag of Words model
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


nlp = spacy.load("en_core_web_sm")
# doc = nlp("I am learning to build a spam detector!")
df = pd.read_csv('larger_dummy_spam_data.csv')  # Assuming you have a CSV file with text data
corpus = df['text'].tolist()  # Replace 'text' with the actual column name containing the text data
labels = df['label'].tolist()  # Replace 'label' with the actual column name
# corpus = [
#     "Congratulations! You've won a $1,000 gift card. Click here to claim now.",     # spam
#     "Reminder: Your appointment is scheduled for tomorrow at 10:00 AM.",             # not spam
#     "URGENT! Your account has been compromised. Verify immediately.",                # spam
#     "Can we reschedule our meeting to next week?",                                   # not spam
#     "You have been selected for a free vacation to the Bahamas!",                    # spam
#     "Hey, just checking in — let me know if you're free to talk later.",             # not spam
#     "Win a brand new iPhone! Limited time offer — apply now!",                       # spam
#     "Monthly report is attached, please review and share feedback.",                 # not spam
# ]
# labels = [1, 0, 1, 0, 1, 0, 1, 0]

# for token in doc:
    #print(token.text, token.lemma_, token.is_stop)
    #print(f"Token: {token.text}, Lemma: {token.lemma_}, Is POS: {token.pos_},Is Stop Word: {token.is_stop}")

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

processed_corpus = [preprocess_text(text) for text in corpus]

#print(f"Processed Corpus:{processed_corpus}")



# vectorizer = CountVectorizer(ngram_range=(1, 2))
# vectorizer.fit(processed_corpus)
# # print(vectorizer.vocabulary_)
# #print(vectorizer.vocabulary_["win"])
# X = vectorizer.transform(processed_corpus)
# print(X.toarray())

X_train, X_test, y_train, y_test = train_test_split(
    processed_corpus, labels, test_size=0.25, random_state=42, stratify=labels
)


clf = Pipeline([
    ('vectorizer_bow', CountVectorizer(ngram_range = (1, 1))),        #using the ngram_range parameter 
    ('Multi NB', MultinomialNB())         
])

# Train
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
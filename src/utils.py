import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing



corpus = [
    "Congratulations! You've won a $1,000 gift card. Click here to claim now.",     # spam
    "Reminder: Your appointment is scheduled for tomorrow at 10:00 AM.",             # not spam
    "URGENT! Your account has been compromised. Verify immediately.",                # spam
    "Can we reschedule our meeting to next week?",                                   # not spam
    "You have been selected for a free vacation to the Bahamas!",                    # spam
    "Hey, just checking in — let me know if you're free to talk later.",             # not spam
    "Win a brand new iPhone! Limited time offer — apply now!",                       # spam
    "Monthly report is attached, please review and share feedback.",                 # not spam
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]
nlp = spacy.load("en_core_web_sm")

def clean_text(text, lower=True, remove_urls=True, remove_emails=True, normalize_ws=True):
    """
    Cleans a string by removing URLs, emails, lowercasing, and fixing spaces.
    Only works on one string at a time (beginner-friendly).
    """
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)

    if remove_emails:
        text = re.sub(r"\b[\w\.-]+?@\w+?\.\w{2,4}\b", "", text)

    if lower:
        text = text.lower()

    if normalize_ws:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def clean_text_list(texts):
    return [clean_text(t) for t in texts]

def preprocess_text_list(texts):
    return [preprocess_text(t) for t in texts]

# def tfidf_vectorize(corpus):
#     vectorizer = TfidfVectorizer(ngram_range=(1, 2))
#     X = vectorizer.fit_transform(corpus)
#     return X, vectorizer


TextCleanerTransformer = preprocessing.FunctionTransformer(clean_text_list, validate=False)
SpacyPreprocessorTransformer = preprocessing.FunctionTransformer(preprocess_text_list, validate=False)


pipeline = Pipeline([
    ("text_cleaner", TextCleanerTransformer),
    ("spacy_preprocessor", SpacyPreprocessorTransformer),
    ("tfidf_vectorizer", TfidfVectorizer(ngram_range=(1, 2),lowercase=False)),
])

x = pipeline.fit(corpus)
print("Pipeline fitted successfully.",x)

x = pipeline.transform(corpus)
print("Transformed corpus shape:", x.shape)



    

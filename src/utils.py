import spacy
import re
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing


# datasets = pd.read_csv(
#     r"D:\Professional\SpamDetector\data\SMSSpamCollection", sep="\t", header=None, names=["label", "text"]
# )

# datasets["label"] = datasets["label"].map({"ham": 0, "spam": 1})


# text = datasets["text"].tolist()
# labels = np.array(datasets["label"].tolist())

def load_dataset(path: str):
    
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".csv":
        df = pd.read_csv(path)
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        
    elif ext in [".tsv", ".txt"] or ext == "":
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or TSV file.")
    
    df['text'] = df['text'].fillna('') 
    df = df[df['text'].str.strip() != '']
    
    return df["text"].tolist(), np.array(df["label"]) 

X_sms, y_sms = load_dataset("D:/Professional/SpamDetector/data/SMSSpamCollection")

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


TextCleanerTransformer = preprocessing.FunctionTransformer(clean_text_list, validate=False)
SpacyPreprocessorTransformer = preprocessing.FunctionTransformer(preprocess_text_list, validate=False)


pipeline = Pipeline([
    ("text_cleaner", TextCleanerTransformer),
    ("spacy_preprocessor", SpacyPreprocessorTransformer),
    ("tfidf_vectorizer", TfidfVectorizer(ngram_range=(1, 2),lowercase=False)),
])

x = pipeline.fit(X_sms)
print("Pipeline fitted successfully.",x)

x = pipeline.transform(X_sms)
print("Transformed corpus shape:", x.shape)



    

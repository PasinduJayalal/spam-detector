import spacy
import re
import pandas as pd
import os
import numpy as np


def load_dataset(path: str):
    
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        
    elif ext in [".tsv", ".txt"] or ext == "":
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
        
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or TSV file.")
    
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})
    
    if not {"label", "text"}.issubset(df.columns):
        raise ValueError("Dataset must have 'label' and 'text' columns.")
    
    df = df[["label", "text"]]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df['text'] = df['text'].fillna('') 
    df = df[df['text'].str.strip() != '']
    
    return df["text"].tolist(), np.array(df["label"]) 



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




    

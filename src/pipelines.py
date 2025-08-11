from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer




def make_pipeline(clean_text_list, preprocess_text_list):
    
    text_cleaner  = FunctionTransformer(clean_text_list, validate=False)
    spacy_preprocessor  = FunctionTransformer(preprocess_text_list, validate=False)
    
    pipeline = Pipeline([
        ("text_cleaner", text_cleaner ),
        ("spacy_preprocessor", spacy_preprocessor ),
        ("tfidf_vectorizer", TfidfVectorizer(ngram_range=(1, 2),lowercase=False)),
    ])
    return pipeline

# tests/unit/test_pipelines_word.py
import numpy as np
import scipy.sparse as sp

from src.utils import clean_text_list, preprocess_text_list
from src.pipelines import make_pipeline

SAMPLE_TEXTS = [
    "FREE money now!!! visit https://SPAM.com",  
    "free money offer today",                    
    "Hello friend :)",                           
    "🔥🔥🔥",                                     
    "   ",                                       
]

def test_word_pipeline_basic_shape():
    pipe = make_pipeline(clean_text_list, preprocess_text_list)
    X = pipe.fit_transform(SAMPLE_TEXTS)

    assert sp.isspmatrix(X)
    assert X.shape[0] == len(SAMPLE_TEXTS)
    assert X.shape[1] > 0  

def test_word_pipeline_determinism():
    pipe = make_pipeline(clean_text_list, preprocess_text_list)
    pipe.fit(SAMPLE_TEXTS)

    X1 = pipe.transform(SAMPLE_TEXTS)
    X2 = pipe.transform(SAMPLE_TEXTS)

    np.testing.assert_allclose(X1.toarray(), X2.toarray())

def test_word_pipeline_uses_cleaner():
    pipe = make_pipeline(clean_text_list, preprocess_text_list)
    pipe.fit(SAMPLE_TEXTS)
    vocab = pipe.named_steps["tfidf_vectorizer"].vocabulary_

    
    for term in list(vocab.keys())[:10]:
        assert term == term.lower()
    
    assert not any("http" in k or "www" in k for k in vocab)

def test_word_pipeline_bigrams_present():
    pipe = make_pipeline(clean_text_list, preprocess_text_list)
    pipe.fit(SAMPLE_TEXTS)
    vocab = pipe.named_steps["tfidf_vectorizer"].vocabulary_

    assert "free money" in vocab  

def test_word_pipeline_handles_edge_inputs():
    texts = ["", "   ", "🔥🔥", "Hello there friend"]  
    pipe = make_pipeline(clean_text_list, preprocess_text_list)
    X = pipe.fit_transform(texts)

    assert X.shape[0] == len(texts)
    assert X.shape[1] > 0

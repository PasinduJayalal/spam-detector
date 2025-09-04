# tests/unit/test_pipelines_char.py
import numpy as np
import scipy.sparse as sp

from src.utils import clean_text_list
from src.pipelines import make_char_pipeline

SAMPLE_TEXTS = [
    "FREE money now!!! visit https://SPAM.com",  
    "free money offer today",
    "Hello friend :)",
    "🔥🔥🔥",                                      
    "   ",                                        
]

def test_char_pipeline_basic_shape():
    pipe = make_char_pipeline(clean_text_list, ngram_range=(3, 5), min_df=1)
    X = pipe.fit_transform(SAMPLE_TEXTS)

    assert sp.isspmatrix(X)
    assert X.shape[0] == len(SAMPLE_TEXTS)
    assert X.shape[1] > 0  

def test_char_pipeline_determinism():
    pipe = make_char_pipeline(clean_text_list, ngram_range=(3, 5), min_df=1)
    pipe.fit(SAMPLE_TEXTS)

    X1 = pipe.transform(SAMPLE_TEXTS).toarray()
    X2 = pipe.transform(SAMPLE_TEXTS).toarray()
    np.testing.assert_allclose(X1, X2)

def test_char_pipeline_respects_min_df_and_ngram():
    
    texts = ["free", "free", "xxx"]  
    pipe = make_char_pipeline(clean_text_list, ngram_range=(3, 3), min_df=2)
    pipe.fit(texts)
    vocab = pipe.named_steps["tfidf_vectorizer"].vocabulary_

    assert "fre" in vocab
    assert "ree" in vocab
    assert "xxx" not in vocab  

def test_char_pipeline_handles_edge_inputs():
    texts = ["", "   ", "🔥🔥", "Hello there friend"]  
    pipe = make_char_pipeline(clean_text_list, ngram_range=(3, 5), min_df=1)
    X = pipe.fit_transform(texts)

    assert X.shape[0] == len(texts)
    assert X.shape[1] > 0

def test_char_pipeline_no_url_fragments_in_vocab():
    pipe = make_char_pipeline(clean_text_list, ngram_range=(3, 5), min_df=1)
    pipe.fit(SAMPLE_TEXTS)
    vocab = pipe.named_steps["tfidf_vectorizer"].vocabulary_
    bad_bits = [k for k in vocab if ("htt" in k or "ttp" in k or "www" in k)]
    assert bad_bits == []

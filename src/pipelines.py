from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def make_pipeline(clean_text_list, preprocess_text_list):

    text_cleaner = FunctionTransformer(clean_text_list, validate=False)
    spacy_preprocessor = FunctionTransformer(preprocess_text_list, validate=False)

    pipeline = Pipeline(
        [
            ("text_cleaner", text_cleaner),
            ("spacy_preprocessor", spacy_preprocessor),
            ("tfidf_vectorizer", TfidfVectorizer(ngram_range=(1, 2), lowercase=False)),
        ]
    )
    return pipeline


def make_char_pipeline(clean_text_list, ngram_range=(3, 5), min_df=2):

    text_cleaner = FunctionTransformer(clean_text_list, validate=False)

    return Pipeline(
        [
            ("text_cleaner", text_cleaner),
            (
                "tfidf_vectorizer",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=ngram_range,
                    lowercase=False,
                    min_df=min_df,
                ),
            ),
        ]
    )

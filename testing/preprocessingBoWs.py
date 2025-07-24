import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I am learning to build a spam detector!")

for token in doc:
    print(token.text, token.lemma_, token.is_stop)

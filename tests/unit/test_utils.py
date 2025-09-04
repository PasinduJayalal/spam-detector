from src.utils import load_dataset, clean_text, preprocess_text, clean_text_list, preprocess_text_list

def test_load_email_dataset_csv():
    texts, labels = load_dataset("tests/data/golden_email.csv")
    assert len(texts) == 7
    assert len(labels) == len(texts)
    assert set(labels) == {0, 1}
    assert texts[0] != "" 
    
def test_load_sms_dataset_tsv():
    texts, labels = load_dataset("tests/data/golden_sms.csv")
    assert len(texts) == 7
    assert len(labels) == len(texts)
    assert set(labels) == {0, 1}
    assert texts[0] != ""
    
    
def test_clean_text():
    
    raw_text = "Claim NOW at https://scam.com !!!"
    cleaned = clean_text(raw_text)
    assert "claim now at !!!" == cleaned
    
    raw_text = "Contact me at test@mail.com"
    cleaned = clean_text(raw_text)
    assert "contact me at" == cleaned
    
    raw_text = "Visit WWW.FreeStuff.org for GIFTS"
    cleaned = clean_text(raw_text)
    assert "visit for gifts" == cleaned
    
    raw_text = "FREE CASH NOW"
    cleaned = clean_text(raw_text)
    assert "free cash now" == cleaned
    
    raw_text = " Hello there\tfriend "
    cleaned = clean_text(raw_text)
    assert cleaned == "hello there friend"
    
    raw_text = " Email me: A@B.co or visit http://x.y !!! "
    cleaned = clean_text(raw_text)
    assert cleaned == "email me: or visit !!!"
    
    raw_text = "🔥🔥🔥 WOW!!! "
    cleaned = clean_text(raw_text)
    assert cleaned == "🔥🔥🔥 wow!!!"
    
    raw_text = "version 2.0 is out now"
    cleaned = clean_text(raw_text)
    assert cleaned == "version 2.0 is out now"
    
    raw_text = "Log in at https://secure.bank.example/login"
    cleaned = clean_text(raw_text)
    assert cleaned == "log in at"
    
    raw_text = "hello world"
    cleaned = clean_text(raw_text)
    assert cleaned == "hello world"
    
def test_preprocess_text():
    
    raw_text = "This is a test"
    preprocessed = preprocess_text(raw_text)
    assert "test" == preprocessed
    
    raw_text = "running quickly!"
    preprocessed = preprocess_text(raw_text)
    assert "run quickly" == preprocessed
    
    raw_text = "Children are playing games"
    preprocessed = preprocess_text(raw_text)
    assert "child play game" == preprocessed
    
    raw_text = "She went to school"
    preprocessed = preprocess_text(raw_text)
    assert "go school" == preprocessed
    
    raw_text = "🔥🔥🔥 WOW!!!"
    preprocessed = preprocess_text(raw_text)
    assert "🔥 🔥 🔥 WOW" == preprocessed
    
    raw_text = "!!! … ?"
    preprocessed = preprocess_text(raw_text)
    assert "" == preprocessed
    
    raw_text = "今日はいい天気ですね"
    preprocessed = preprocess_text(raw_text)
    assert isinstance(preprocessed, str)
    assert len(preprocessed) > 0
    
def test_clean_text_list():
    texts = ["Hello World!", "Visit https://example.com", "Contact me at"]
    cleaned_texts = clean_text_list(texts)
    assert cleaned_texts == ["hello world!", "visit", "contact me at"]
    
def test_preprocess_text_list():
    texts = ["This is a test", "running quickly!", "Children are playing games"]
    preprocessed_texts = preprocess_text_list(texts)
    assert preprocessed_texts == ["test", "run quickly", "child play game"]
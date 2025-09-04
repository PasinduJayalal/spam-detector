import math
from src.infer import predict_one_with_score, predict_batch_with_score


class FakeModel:
    def decision_function(self, X):
        out = []
        for text in X:
            t = (text or "").lower()
            if "neutral" in t:
                out.append(0.0)   
            elif ("free" in t) or ("win" in t):
                out.append(2.0)   
            else:
                out.append(-2.0)  
        return out

def test_predict_one_with_score_cases():
    m = FakeModel()

    
    label, score = predict_one_with_score(m, "free gift")
    assert label == "Spam"
    assert 0.0 <= score <= 1.0

    
    label, score = predict_one_with_score(m, "hello friend")
    assert label == "Not Spam"
    assert 0.0 <= score <= 1.0

    
    label, score = predict_one_with_score(m, "neutral message")
    assert label == "Spam"
    assert math.isclose(score, 0.5, rel_tol=1e-6)

def test_predict_one_with_score_rejects_empty():
    m = FakeModel()
    try:
        predict_one_with_score(m, "   ")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()

def test_predict_batch_with_score_mixed_inputs():
    m = FakeModel()
    texts = ["", "   ", "win money", "hello there"]
    results = predict_batch_with_score(m, texts)

    assert len(results) == len(texts)

    
    assert results[0]["label"] == "Invalid" and results[0]["score"] == 0.0
    assert results[1]["label"] == "Invalid" and results[1]["score"] == 0.0

    
    assert results[2]["label"] == "Spam"
    assert 0.0 < results[2]["score"] <= 1.0

    
    assert results[3]["label"] == "Not Spam"
    assert 0.0 <= results[3]["score"] < 0.5

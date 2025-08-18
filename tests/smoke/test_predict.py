import subprocess
import sys



def test_predict_sms():
    result = subprocess.run(
        [sys.executable, "-m", "src.predict", "--model", "sms", "--file", "data/demo_sms.txt"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "SMS:" in result.stdout
    assert "Predicted:" in result.stdout
    
def test_predict_email():
    result = subprocess.run(
        [sys.executable, "-m", "src.predict", "--model", "email", "--file", "data/demo_email.txt"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Email:" in result.stdout
    assert "Predicted:" in result.stdout
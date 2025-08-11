from src.utils import clean_text_list, preprocess_text_list, load_dataset
from src.pipelines import make_pipeline

X, y = load_dataset(r"D:\Professional\SpamDetector\data\SMSSpamCollection")
pipe = make_pipeline(clean_text_list, preprocess_text_list)
X_vec = pipe.fit_transform(X)
print("Matrix shape:", X_vec.shape)

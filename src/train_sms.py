from src.utils import load_dataset, clean_text_list, preprocess_text_list
from src.pipelines import make_pipeline

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


def main():
    X, y = load_dataset("data/SMSspam2.csv")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


    clf = Pipeline([
        ('preprocess', make_pipeline(clean_text_list, preprocess_text_list)),
        ('model', LinearSVC(class_weight="balanced", C=1.0))
    ])

    # pipe = make_pipeline(clean_text_list, preprocess_text_list)

    # X_vec = pipe.fit_transform(X_train)

    # print("Matrix shape:", X_vec.shape)

    # model = SVC(class_weight="balanced", kernel='linear', C=1.0)
    # model.fit(X_vec, y_train)
    # y_pred = model.predict(pipe.transform(X_test))

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred ,target_names=['Not Spam', 'Spam']))
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual: Not Spam', 'Actual: Spam'], columns=['Predicted: Not Spam', 'Predicted: Spam'])
    print(cm_df)

if __name__ == "__main__":
    main()
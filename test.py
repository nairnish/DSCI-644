import time
import sys
import pandas as pd
from attempt_1 import my_model
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

sys.path.insert(0, '..')
from my_evaluation import my_evaluation

def test(data):
    clf = my_model()
    y = data["REFACTORINGS (LABELS)"]
    X = data.drop(['REFACTORINGS (LABELS)'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    f1 = metrics.f1_score(y_test, predictions, average='micro')
    # eval = my_evaluation(predictions, y_test)
    # f1 = eval.f1(target=1)
    return f1


if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("/Users/nishantnair/DSCI-644/Project/data_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    f1 = test(data)
    print("F1 score: %f" % f1)
    runtime = (time.time() - start) / 60.0
    print(runtime)
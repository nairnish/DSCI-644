import time
import sys
import pandas as pd
from attempt_1 import my_model
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


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
    prec = metrics.precision_score(y_test, predictions, average='micro')
    print(classification_report(y_test,predictions))
    return f1,prec


if __name__ == "__main__":
    start = time.time()

    # Load data
    data = pd.read_csv("/Users/prajwalkrishn/Desktop/DSCI644_Project/DSCI-644/Prajwal/datamulti_train.csv",encoding='utf-8')

    # Replace missing values with empty strings
    data = data.fillna("")
    f1,prec = test(data)

    print("F1 score: %f" % f1)
    print("Precision score: %f" % prec)

    runtime = (time.time() - start) / 60.0
    print(runtime)
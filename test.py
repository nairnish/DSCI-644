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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


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
    # eval = my_evaluation(predictions, y_test)
    # f1 = eval.f1(target=1)
    # prec = eval.
    return f1,prec


if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("/Users/nishantnair/DSCI-644/Project/DSCI-644/dataone_train .csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    f1,prec = test(data)
    print("F1 score: %f" % f1)
    print("Precision score: %f" % prec)
    runtime = (time.time() - start) / 60.0
    print(runtime)

    # Fit model
    # clf = my_model()
    # y = data["REFACTORINGS (LABELS)"]
    # X = data.drop(['REFACTORINGS (LABELS)'], axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # clf.fit(X_train, y_train)
    # # clf.fit(X, y)
    # # Predict on training data
    # predictions = clf.predict(X_test)
    # print(predictions)
    # # print(probs)
    # Predict probabilities
    # probs = clf.predict_proba(X)
    # probs = pd.DataFrame({key: probs[:, i] for i, key in enumerate(clf.classes_)})
    # Evaluate results
    # metrics = my_evaluation(predictions, y, probs)
    # result = {}
    # for target in clf.classes_:
    #     result[target] = {}
    #     result[target]["prec"] = metrics.precision(target)
    #     result[target]["recall"] = metrics.recall(target)
    #     result[target]["f1"] = metrics.f1(target)
    #     result[target]["auc"] = metrics.auc(target)
    # print(result)
    # f1 = {average: metrics.f1(target=None, average=average) for average in ["macro", "micro", "weighted"]}
    # print("Average F1 scores: ")
    # print(f1)

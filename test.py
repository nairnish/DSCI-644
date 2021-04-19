import time
import sys
import pandas as pd
from prod import my_model
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np


sys.path.insert(0, '..')
from my_evaluation import my_evaluation
from collections import Counter

def test(data):
    clf = my_model()
    # y - Refactoring labels
    y = data["REFACTORINGS (LABELS)"]
    # X - Commit Texts
    X = data.drop(['REFACTORINGS (LABELS)'], axis=1)

    # Splitting into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # fit
    clf.fit(X_train, y_train)
    # predictions
    predictions = clf.predict(X_test)
    # scores
    f1 = metrics.f1_score(y_test, predictions, average='micro')
    prec = metrics.precision_score(y_test, predictions, average='micro')
    print(classification_report(y_test,predictions))

    # confusion matrix
    conf = confusion(y_test,predictions,y)

    new = pd.DataFrame.from_dict(conf)

    data = new.transpose()

    x_test = y_test.axes[0].values

    y_df = y_test.to_frame()
    x_df = pd.DataFrame(x_test, columns=['id'])
    pred_df = pd.DataFrame(predictions, columns=['predicted'])

    # Exporting predictions to analyse tp, fp, tn, fn
    y_df.to_csv('y_data.csv', sep='\t', encoding='utf-8', index=False)
    x_df.to_csv('x_data.csv', sep='\t', encoding='utf-8', index=False)
    pred_df.to_csv('pred_data.csv', sep='\t', encoding='utf-8', index=False)

    print(data)

    print(conf)

    print(y_test)
    print(predictions)
    return f1,prec

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def confusion(actuals, predictions, all_labels):
        # compute confusion matrix for each class in self.classes_
        # self.confusion_matrix = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below
        correct = predictions == actuals
        acc = float(Counter(correct)[True])/len(correct)
        conf_matrix = {}
        class_name = np.unique(all_labels)
        for label in class_name:
            tp = Counter(correct & (predictions == label))[True]
            fp = Counter((actuals != label) & (predictions == label))[True]
            tn = Counter(correct & (predictions != label))[True]
            fn = Counter((actuals == label) & (predictions != label))[True]
            conf_matrix[label] = {"TP":tp, "TN": tn, "FP": fp, "FN": fn}
        # print(conf_matrix)
        return conf_matrix


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

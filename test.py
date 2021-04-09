import time
import sys
import pandas as pd
from attempt_2 import my_model
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
    y = data["REFACTORINGS (LABELS)"]
    X = data.drop(['REFACTORINGS (LABELS)'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    f1 = metrics.f1_score(y_test, predictions, average='micro')
    prec = metrics.precision_score(y_test, predictions, average='micro')
    print(classification_report(y_test,predictions))
    # conf = {}
    conf = confusion(y_test,predictions,y)

    new = pd.DataFrame.from_dict(conf)

    data = new.transpose()

    y_df = y_test.to_frame()
    pred_df = pd.DataFrame(predictions, columns=['predicted'])

    y_df.to_csv('y_data', sep='\t', encoding='utf-8', index=False)
    pred_df.to_csv('pred_data', sep='\t', encoding='utf-8', index=False)

    # y_df.to_csv(index=False)
    # pred_df.to_csv(index=False)



    print(data)

    print(conf)

    print(y_test)
    print(predictions)
    # print(confusion_matrix(y_test,predictions))
    class_name = np.unique(y)
    # print(confusion_matrix(y_test, predictions, labels=class_name))

    # FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    # FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    # TP = np.diag(confusion_matrix)
    # TN = confusion_matrix.values.sum() - (FP + FN + TP)
    #
    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)
    #
    # # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)



    # eval = my_evaluation(predictions, y_test)
    # f1 = eval.f1(target=1)
    # prec = eval.
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

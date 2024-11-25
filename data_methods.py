# File that has all the functions used in the project.

# import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 

# create a function for the ExtraTreesClassifier
def ExtraTreesClassifier_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = ExtraTreesClassifier(random_state=42).fit(X_train, y_train)
    print(f'Training Score: {clf.score(X_train, y_train)}')
    print(f'Testing Score: {clf.score(X_test, y_test)}')
    return clf


# create a function to find the feature importance
def feature_importance(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = ExtraTreesClassifier(max_depth=12, random_state=42).fit(X_train, y_train)
    feature_importances = clf.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(10, 5))
    sorted_idx = np.argsort(feature_importances)
    sortedfeature_names = feature_names[sorted_idx]
    plt.barh(sortedfeature_names, feature_importances[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.show()
    return sorted(zip(feature_importances, feature_names), reverse=True)

# create a function for the depth parameter
def ExtraTreesClassifier_model_depth(X, y, z):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=z)
    clf = ExtraTreesClassifier(max_depth=12, random_state=z).fit(X_train, y_train)
    print(f'Training Score: {clf.score(X_train, y_train)}')
    print(f'Testing Score: {clf.score(X_test, y_test)}')
    return clf


# function to be called for ExtraTreesClassifier

def XTrees_analysis(X,y,z,title):
    ExtraTreesClassifier_model(X, y)
    feature_importance(X, y, title)
    ExtraTreesClassifier_model_depth(X, y, z)
    return


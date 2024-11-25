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

# create a function for the train test split
def split_data(X, y, z):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=z)
    return X_train, X_test, y_train, y_test

# create a function for the ExtraTreesClassifier
# pass the X_train and y_train to the model from the split_data function



def extratreesclassifier_model(X, y, z, title, title2):
    X_train, X_test, y_train, y_test = split_data(X, y, z)
    clf = ExtraTreesClassifier(random_state=z).fit(X_train, y_train)
    print(f'Scores for the Model\n')
    print(f'Training Score : {clf.score(X_train, y_train)}')
    print(f'Testing Score: {clf.score(X_test, y_test)}')
    feature_importances = clf.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(10, 5))
    sorted_idx = np.argsort(feature_importances)
    sortedfeature_names = feature_names[sorted_idx]
    plt.barh(sortedfeature_names, feature_importances[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.show()
    # for i in range(len(feature_names)):
    #     print(f'{feature_names[i]}: {feature_importances[i]}')
    # print('\n')
    print('Classification Report\n')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix\n')
    print(confusion_matrix(y_test, y_pred))
# for depth from 1 to 16 create a loop to test the model and print the training and testing scores
# for depth from 1 to 16 create a loop to test the model and print the training and testing scores
  
    depths = range(1, 16)
    training_scores = []
    testing_scores = []

    for i in depths:
        clf_depth = ExtraTreesClassifier(max_depth=i, random_state=z).fit(X_train, y_train)
        training_scores.append(clf_depth.score(X_train, y_train))
        testing_scores.append(clf_depth.score(X_test, y_test))

    plt.figure(figsize=(10, 5))
    plt.plot(depths, training_scores, label='Training Score')
    plt.plot(depths, testing_scores, label='Testing Score')
    plt.xlabel('Depth')
    plt.ylabel('Score')
    plt.title(f'Training and Testing Scores vs Depth\n{title2}')
    plt.legend()
    plt.show()


#     return sorted(zip(feature_importances, feature_names), reverse=True)

# # create a function for the depth parameter
# def extratreesclassifier_model_depth(X, y, z):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=z)
#     clf = ExtraTreesClassifier(max_depth=12, random_state=z).fit(X_train, y_train)
#     print(f'Training Score: {clf.score(X_train, y_train)}')
#     print(f'Testing Score: {clf.score(X_test, y_test)}')
#     return clf


# for i in range(1, 16):
#     print(f'Depth: {i}')
#     ExtraTreesClassifier_model_depth(X, y, i)

# # function to be called for ExtraTreesClassifier





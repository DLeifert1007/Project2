# File that has all the functions used in the project.

# import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import GradientBoostingClassifier



# 

# create a function for the train test split
def split_data(X, y, z):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=z)
    return X_train, X_test, y_train, y_test

def pipeline_gradient_boosting(X, y, z, title, title2):
    X_train, X_test, y_train, y_test = split_data(X, y, z)
    clf = GradientBoostingClassifier(random_state=z).fit(X_train, y_train)
    model_name = clf.__class__.__name__
    
    # capture the clf scores in a variable to use for printing the results
    training_score = clf.score(X_train, y_train)
    testing_score = clf.score(X_test, y_test)

    feature_importances = clf.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(10, 5))
    sorted_idx = np.argsort(feature_importances)
    sortedfeature_names = feature_names[sorted_idx]
    plt.barh(sortedfeature_names, feature_importances[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.show()
    print('Classification Report\n')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix\n')
    print(confusion_matrix(y_test, y_pred))

# Balanced accuracy score

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    bal_acc_score_train = balanced_accuracy_score(y_train, y_train_pred)
    bal_acc_score_test = balanced_accuracy_score(y_test, y_test_pred)
    
#   Add Random Under Sampling to the model
    rus = RandomUnderSampler(random_state=z)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    clf_rus = GradientBoostingClassifier(random_state=z).fit(X_resampled, y_resampled)
    # print(f'Scores for the Model with Random Under Sampling\n')
    rand_under_score_train = clf_rus.score(X_resampled, y_resampled)
    rand_under_score_test = clf_rus.score(X_test, y_test)

#   Add Random Over Sampling to the model just out of curiosity and for comparision to undersampling
#   Since the dataset is likely randomly generated, using over-sampling will likely amplify noise 
    ros = RandomOverSampler(random_state=z)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    clf_ros = GradientBoostingClassifier(random_state=z).fit(X_resampled, y_resampled)
    rand_over_score_train = clf_ros.score(X_resampled, y_resampled)
    rand_over_score_test = clf_ros.score(X_test, y_test)

# # iterate over the depths of the trees
  
#     depths = range(1, 16)
#     training_scores = []
#     testing_scores = []

#     for i in depths:
#         clf_depth = GradientBoostingClassifier(max_depth=i, random_state=z).fit(X_train, y_train)
#         training_scores.append(clf_depth.score(X_train, y_train))
#         testing_scores.append(clf_depth.score(X_test, y_test))

    # plt.figure(figsize=(10, 5))
    # plt.plot(depths, training_scores, label='Training Score')
    # plt.plot(depths, testing_scores, label='Testing Score')
    # plt.xlabel('Depth')
    # plt.ylabel('Score')
    # plt.title(f'Training and Testing Scores vs Depth\n{title2}')
    # plt.legend()
    # plt.show()

  # print the results
    print('\n')
    print(f'Model: {model_name}')
    print(f'Dataset: {title2}\n')
    print(f'Scores for the Original Model')
    print(f'Training Score: {training_score}')
    print(f'Testing Score: {testing_score}')
    print(f'Balanced Accuracy Score for Training : {bal_acc_score_train}')
    print(f'Balanced Accuracy Score for Testing : {bal_acc_score_test}\n')
    print(f'Scores for the Model with Random UnderSampling')
    print(f'Training Score Random Undersampling : {rand_under_score_train}')
    print(f'Testing Score Random Undersampling: {rand_under_score_test}\n')
    print(f'Scores for the Model with Random Over Sampling')
    print(f'Training Score Random Oversampling : {rand_over_score_train}')
    print(f'Testing Score Random Oversampling: {rand_over_score_test}')


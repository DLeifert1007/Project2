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

# create a function for the ExtraTreesClassifier
# pass the X_train and y_train to the model from the split_data function



def extratreesclassifier_model(X, y, z, title, title2):
    X_train, X_test, y_train, y_test = split_data(X, y, z)
    clf = ExtraTreesClassifier(random_state=z).fit(X_train, y_train)
    model_name = clf.__class__.__name__
    
    # capture the clf scores in a variable to use for printing the results
    training_score = clf.score(X_train, y_train)
    testing_score = clf.score(X_test, y_test)
    print(f'Scores for the Model\n')
    print(f'Training Score: {training_score}')
    print(f'Testing Score: {testing_score}')


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

# Balanced accuracy score

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    bal_acc_score_train = balanced_accuracy_score(y_train, y_train_pred)
    bal_acc_score_test = balanced_accuracy_score(y_test, y_test_pred)
    print(f'Balanced Accuracy Score for Training : {bal_acc_score_train}')
    print(f'Balanced Accuracy Score for Testing : {bal_acc_score_test}')
    
#   Add Random Under Sampling to the model
    rus = RandomUnderSampler(random_state=z)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    clf_rus = ExtraTreesClassifier(random_state=z).fit(X_resampled, y_resampled)
    print(f'Scores for the Model with Random Under Sampling\n')
    rand_under_score_train = clf_rus.score(X_resampled, y_resampled)
    rand_under_score_test = clf_rus.score(X_test, y_test)
    print(f'Training Score RU : {rand_under_score_train}')
    print(f'Testing Score RU: {rand_under_score_test}')

#   Add Random Over Sampling to the model
    ros = RandomOverSampler(random_state=z)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    clf_ros = ExtraTreesClassifier(random_state=z).fit(X_resampled, y_resampled)
    rand_over_score_train = clf_ros.score(X_resampled, y_resampled)
    rand_over_score_test = clf_ros.score(X_test, y_test)
    print(f'Scores for the Model with Random Over Sampling\n')
    print(f'Training Score : {rand_over_score_train}')
    print(f'Testing Score: {rand_over_score_test}')

# iterate over the depths of the trees
  
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

  # print the results
    print(f'Model: {model_name}')
    print(f'Dataset: {title2}')
    print(f'Training Score: {training_score}')
    print(f'Testing Score: {testing_score}')
    print(f'Balanced Accuracy Score for Training : {bal_acc_score_train}')
    print(f'Balanced Accuracy Score for Testing : {bal_acc_score_test}')
    print(f'Training Score Random Undersampling : {rand_under_score_train}')
    print(f'Testing Score Random Undersampling: {rand_under_score_test}')
    print(f'Training Score Random Oversampling : {rand_over_score_train}')
    print(f'Testing Score Random Oversampling: {rand_over_score_test}')


def gradientboostclassifier_model(X, y, z, title, title2):
    X_train, X_test, y_train, y_test = split_data(X, y, z)
    clf = GradientBoostingClassifier(random_state=z).fit(X_train, y_train)
    model_name = clf.__class__.__name__
    
    # capture the clf scores in a variable to use for printing the results
    training_score = clf.score(X_train, y_train)
    testing_score = clf.score(X_test, y_test)
    print(f'Scores for the Model\n')
    print(f'Training Score: {training_score}')
    print(f'Testing Score: {testing_score}')


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

# Balanced accuracy score

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    bal_acc_score_train = balanced_accuracy_score(y_train, y_train_pred)
    bal_acc_score_test = balanced_accuracy_score(y_test, y_test_pred)
    print(f'Balanced Accuracy Score for Training : {bal_acc_score_train}')
    print(f'Balanced Accuracy Score for Testing : {bal_acc_score_test}')
    
#   Add Random Under Sampling to the model
    rus = RandomUnderSampler(random_state=z)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    clf_rus = GradientBoostingClassifier(random_state=z).fit(X_resampled, y_resampled)
    print(f'Scores for the Model with Random Under Sampling\n')
    rand_under_score_train = clf_rus.score(X_resampled, y_resampled)
    rand_under_score_test = clf_rus.score(X_test, y_test)
    print(f'Training Score RU : {rand_under_score_train}')
    print(f'Testing Score RU: {rand_under_score_test}')

#   Add Random Over Sampling to the model
    ros = RandomOverSampler(random_state=z)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    clf_ros = GradientBoostingClassifier(random_state=z).fit(X_resampled, y_resampled)
    rand_over_score_train = clf_ros.score(X_resampled, y_resampled)
    rand_over_score_test = clf_ros.score(X_test, y_test)
    print(f'Scores for the Model with Random Over Sampling\n')
    print(f'Training Score : {rand_over_score_train}')
    print(f'Testing Score: {rand_over_score_test}')

# iterate over the depths of the trees
  
    depths = range(1, 16)
    training_scores = []
    testing_scores = []

    for i in depths:
        clf_depth = GradientBoostingClassifier(max_depth=i, random_state=z).fit(X_train, y_train)
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

  # print the results
    print(f'Model: {model_name}')
    print(f'Dataset: {title2}')
    print(f'Training Score: {training_score}')
    print(f'Testing Score: {testing_score}')
    print(f'Balanced Accuracy Score for Training : {bal_acc_score_train}')
    print(f'Balanced Accuracy Score for Testing : {bal_acc_score_test}')
    print(f'Training Score Random Undersampling : {rand_under_score_train}')
    print(f'Testing Score Random Undersampling: {rand_under_score_test}')
    print(f'Training Score Random Oversampling : {rand_over_score_train}')
    print(f'Testing Score Random Oversampling: {rand_over_score_test}')



    
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





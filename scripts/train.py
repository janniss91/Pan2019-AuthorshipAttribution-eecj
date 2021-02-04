"""
Here we should do the training of the actual model.

Since we want to use SVM, I suggest the SVM from scikit-learn.

from sklearn.svm import SVC

"""

from read_data import LanguageData
from read_data import LANGUAGES
from read_data import PROBLEM_IDS

import feature_selection
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from scipy.sparse import hstack




def train(train_feature_vector, train_candidate_vector):
    """
    Train and return a SVM model
    train_feature_vector : an array of shape (numbers_of_train_document, numbers_of_features)
    train_candidate_vector : an array of shape (numbers_of_train_document, 1)
    """
    # parameters for cross-validation
    params = [{'kernel': ['rbf', 'sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
              {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]
    grid = GridSearchCV(SVC(), params, cv=5, refit=True, verbose=3)
    grid.fit(train_feature_vector, train_candidate_vector)

    svm_model = grid.best_estimator_.set_params(probability=True)
    svm_model.fit(train_feature_vector, train_candidate_vector)

    return svm_model


def predict(model, feature_vector, min_difference=0.1, min_probability=0.25):
    """
    Predict with threshold and return an array of shape (numbers_of_samples, numbers_of_classes)
    model : a svm model
    feature_vector: an array of shape (numbers_of_samples, numbers_of_features)
    min_difference : classify to unknown if difference highest and second highest probabilities is below min_difference
    min_probability : classify to unknown if highest probability is below min_probability
    """
    # probability prediction
    proba_predictions = model.predict_proba(feature_vector)
    # label prediction
    label_predictions = model.predict(feature_vector)

    for i, proba_prediction in enumerate(proba_predictions):
        # get the highest and second highest probabilities
        max_proba = np.amax(proba_prediction)
        second_proba = np.sort(proba_prediction)[-2]

        # check should we classify as unknown
        if max_proba - second_proba < min_difference or max_proba < min_probability:
            label_predictions[i] = -1

    return label_predictions


def test(model, test_feature_vector, test_candidate_vector):
    predictions = predict(model, test_feature_vector)
    print("Accuracy : ", accuracy_score(test_candidate_vector, predictions))
    print("Precision : ", precision_score(test_candidate_vector, predictions, average='macro'))
    print("F1 score : ", f1_score(test_candidate_vector, predictions, average='macro'))
    print("Recall : ", recall_score(test_candidate_vector, predictions, average='macro'))


if __name__ == "__main__":
    """
    Here goes the whole pipeline of data reading, text processing, feature
    selection, training and evaluating
    """

    # Data Reading - Three language objects are set up that carry all texts
    # split up into problems -> candidates -> texts
    language_data_objs = [
        LanguageData(language, id_list)
        for num, (language, id_list) in enumerate(zip(LANGUAGES, PROBLEM_IDS))
    ]
    # NOTE: Look in read_data.py how to access these objects.
    english, french, italian, spanish = language_data_objs

    # Text Processing goes here.
    english_text = english.get_all_known()
    french_text = french.get_all_known()
    italian_text = italian.get_all_known()
    spanish_text = spanish.get_all_known()

    # get labels - define labels for every candidate
    array = np.repeat(np.arange(180), 7)


    # get char-ngram

    # get word-ngram

    # get punct-ngram

    # get distortion ngram


    # Train and Test
    problem1_label = np.repeat([1, 2, 3, 4, 5, 6, 7, 8, 9], 7)
    problem1 = english.problems["problem00001"]
    problem1_text = problem1.get_all_known()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(problem1_text, problem1_label, test_size=0.33)
    train_f, test_f = feature_selection.char_ngrams(X_train, X_test)
    model = train(train_f, y_train)

    test(model, test_f, y_test)



    # Evaluate

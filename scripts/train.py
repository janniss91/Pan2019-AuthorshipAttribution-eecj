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
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing
from itertools import islice


def train(train_feature_vector, train_candidate_vector, tune=False, params=None):
    """
    Train and return a SVM model
    train_feature_vector : an array of shape (numbers_of_train_document, numbers_of_features)
    train_candidate_vector : an array of shape (numbers_of_train_document, 1)
    tune: if you want to tune the model (I already tune it. If you want to tune it again, just set tune=True)
    """
    if tune:
        # parameters for cross-validation
        params = [{'kernel': ['rbf', 'sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                   'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                  {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]
        grid = GridSearchCV(SVC(), params, cv=2, refit=True, verbose=3)
        grid.fit(train_feature_vector, train_candidate_vector)

        svm_model = grid.best_estimator_.set_params(probability=True)

    elif params:
        svm_model = SVC(probability=True).set_params(**params)
    else:
        svm_model = SVC(probability=True, kernel='sigmoid', C=10, gamma=0.001)

    svm_model.fit(train_feature_vector, train_candidate_vector)

    return svm_model


def predict(model, feature_vector, min_difference=0.004, min_probability=0.01):
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
            label_predictions[i] = -1  # set unknown candidate to label -1

    return label_predictions


def evaluate(prediction_vector, truth_vector, problem_id=None):
    if not problem_id:
        print("Average ")
    else:
        print("Problem" + str(problem_id))
    print("Accuracy : ", accuracy_score(truth_vector, prediction_vector))
    print("F1 score : ", f1_score(truth_vector, prediction_vector, average='macro'))
    print("-----------------------")


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
    # Get Training data - get all known texts for training and combine all text to a list
    train_text = english.get_all_known() + french.get_all_known() + italian.get_all_known() + spanish.get_all_known()
    # get labels and combine all labels to a numpy array
    train_label_vector = np.array(english.get_all_labels() + french.get_all_labels() + italian.get_all_labels() + spanish.get_all_labels())

    # get all unknown text for testing
    english_unknown_text, english_unknown_label = map(list,zip(*english.get_all_unknown()))
    french_unknown_text, french_unknown_label = map(list, zip(*french.get_all_unknown()))
    italian_unknown_text, italian_unknown_label = map(list, zip(*italian.get_all_unknown()))
    spanish_unknown_text, spanish_unknown_label = map(list, zip(*spanish.get_all_unknown()))
    # combine all unknown text
    test_text = english_unknown_text + french_unknown_text + italian_unknown_text + spanish_unknown_text
    # combine all truth labels
    test_label_vector = np.array(english_unknown_label + french_unknown_label + italian_unknown_label + spanish_unknown_label)


    # get char-ngram
    char_train, char_test = feature_selection.char_ngrams(train_text, test_text)

    # get word-ngram
    word_train, word_test = feature_selection.word_ngrams(train_text, test_text)

    # get punct-ngram
    punct_train, punct_test = feature_selection.punct_ngrams(train_text, test_text)

    # get distortion ngram
    dist_train, dist_test = feature_selection.dist_ngrams(train_text, test_text)

    # combine all features vectors to one numpy array
    train_feature_vector = np.hstack((char_train, word_train, punct_train, dist_train))
    test_feature_vector = np.hstack((char_test, word_test, punct_test, dist_test))

    # Normalize the vector
    scaler = preprocessing.MaxAbsScaler()
    train_feature_vector = scaler.fit_transform(train_feature_vector)
    test_feature_vector = scaler.transform(test_feature_vector)
    # classify vectors by problems - for evaluating model on different testing data in different languages
    sizes = []  # numbers of unknown files in every Problem
    for lang in language_data_objs:
        for problem in lang.problems.values():
            sizes.append(len(problem.unknown_with_labels))
    it = iter(test_feature_vector)
    unknown_text_list = [list(islice(it, n)) for n in sizes]
    it2 = iter(test_label_vector)
    unknown_label_list = [list(islice(it2, n)) for n in sizes]

    # Train and Test
    model = train(train_feature_vector, train_label_vector)  # svm model
    all_predictions = predict(model, test_feature_vector)  # all predictions of unknown text
    # make predictions of different problems - for evaluating
    predictions_per_problem = []  # [predictions_for_unknown_in_problem1, predictions_for_unknown_in_problem2, ...]
    for vector in unknown_text_list:
        predictions_per_problem.append(predict(model, vector))

    # Evaluate
    for i, (predictions, labels) in enumerate(zip(predictions_per_problem, unknown_label_list)):
        evaluate(predictions, labels, problem_id=i+1)
    evaluate(all_predictions, test_label_vector)



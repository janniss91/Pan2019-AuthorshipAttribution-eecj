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
from prettytable import PrettyTable
import json


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
                   'C': [0.001, 0.01, 0.1, 10, 25, 50, 100, 1000]},
                  {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]
        grid = GridSearchCV(SVC(), params, cv=7, refit=True, verbose=3)
        grid.fit(train_feature_vector, train_candidate_vector)

        svm_model = grid.best_estimator_.set_params(probability=True)

    elif params:
        svm_model = SVC(probability=True).set_params(**params)
    else:
        svm_model = SVC(probability=True, kernel='sigmoid', C=10, gamma=0.01)

    svm_model.fit(train_feature_vector, train_candidate_vector)

    return svm_model


def predict(models, feature_vectors, min_difference=0.1, min_probability=0.2):
    """
    Predict with threshold and return an array of shape (numbers_of_samples, numbers_of_classes)
    model : a svm model
    feature_vector: an array of shape (numbers_of_samples, numbers_of_features)
    min_difference : classify to unknown if difference highest and second highest probabilities is below min_difference
    min_probability : classify to unknown if highest probability is below min_probability
    """
    predict_probas = []
    for model, vector in zip(models, feature_vectors):
        # probability prediction
        predict_probas.append(model.predict_proba(vector))
    average_proba = sum(predict_probas) / len(models)  # soft-voting

    labels = np.zeros((average_proba.shape[0],))
    for i, proba in enumerate(average_proba):
        # get the highest and second highest probabilities
        max_proba = np.amax(proba)
        max_index = proba.argmax()
        second_proba = np.sort(proba)[-2]

        # check should we classify as unknown
        if max_proba - second_proba < min_difference or max_proba < min_probability:
            labels[i] = -1  # set unknown candidate to label -1
        else:
            labels[i] = max_index + 1  # convert index to label

    return labels


def evaluate(prediction_vector, truth_vector):
    return accuracy_score(truth_vector, prediction_vector), f1_score(truth_vector, prediction_vector, average='macro')


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

    # print the result
    x = PrettyTable()
    x.field_names = ["Problem", "Language", "Baseline Accuracy", "Baseline F1-score", "Soft Vote Accuracy", "Soft Vote F1-score"]
    total_baseline_acc = 0
    total_baseline_f1 = 0
    total_soft_acc = 0
    total_soft_f1 = 0

    params_dict = {}

    # iterate over langauges
    for langauge in language_data_objs:
        langauge_name = langauge.lang_name
        # iterate over problems
        for problem_name in langauge.problems:
            problem = langauge.problems[problem_name]
            train_text = problem.get_all_known()
            train_label = np.array(problem.labels)
            test_text = [x[0] for x in problem.unknown_with_labels]
            test_label = np.array([x[1] for x in problem.unknown_with_labels])
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
            char_train = scaler.fit_transform(char_train)
            char_test = scaler.transform(char_test)
            word_train = scaler.fit_transform(word_train)
            word_test = scaler.transform(word_test)
            punct_train = scaler.fit_transform(punct_train)
            punct_test = scaler.transform(punct_test)
            dist_train = scaler.fit_transform(dist_train)
            dist_test = scaler.transform(dist_test)

            # train models with different features
            params = json.load(open('params.json'))
            char_svm = train(char_train, train_label, params=params[problem_name]['char_svm'])
            word_svm = train(word_train, train_label, params=params[problem_name]['word_svm'])
            punct_svm = train(punct_train, train_label, params=params[problem_name]['punct_svm'])
            dist_svm = train(dist_train, train_label, params=params[problem_name]['dist_svm'])
            baseline_svm = train(train_feature_vector, train_label, params=params[problem_name]['base_svm'])

            # put models in a list and test vectors in a list
            models = [char_svm, word_svm, punct_svm, dist_svm]
            test_feature_vectors = [char_test, word_test, punct_test, dist_test]

            predictions_soft_vote = predict(models, test_feature_vectors)  # predictions with soft voting
            prediction_baseline = predict([baseline_svm], [test_feature_vector])  # prediction without soft voting

            # Evaluate
            soft_acc, soft_f1 = evaluate(predictions_soft_vote, test_label)
            baseline_acc, baseline_f1 = evaluate(prediction_baseline, test_label)
            total_soft_acc += soft_acc
            total_soft_f1 += soft_f1
            total_baseline_acc += baseline_acc
            total_baseline_f1 += baseline_f1
            x.add_row([problem_name[-2:], langauge_name, baseline_acc, baseline_f1, soft_acc, soft_f1])

    x.add_row(["mean", "all", total_baseline_acc/20, total_baseline_f1/20, total_soft_acc/20, total_soft_f1/20])
    print(x)




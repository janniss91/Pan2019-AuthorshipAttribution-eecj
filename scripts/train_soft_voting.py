"""
Here we should do the training of the actual model.

Since we want to use SVM, I suggest the SVM from scikit-learn.

from sklearn.svm import SVC

"""

from read_data import LanguageData
from read_data import LANGUAGES
from read_data import PROBLEM_IDS


def train(train_feature_vector, train_candidate_vector):
    pass


def test(test_feature_vector, test_candidate_vector):
    pass


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

    # Feature Selection and Feature Setup Goes Here

    # Train and Test

    # Evaluate

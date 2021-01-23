"""
Here we should implement multiple functions for feature selection.

Below are some examples of functions that care about single features.
"""


def extract_word_counts():
    """
    This function requires tokenization and lemmatization.
    :return: This should return a numpy array.
    """
    pass


def extract_sentence_lengths(text):
    """
    This function requires tokenization and lemmatization.
    This function also requires a sentence splitting method.

    :return: This should return the lengths of all 
    sentences in the text as a List of integers
    """
    pass


"""
... more feature exraction methods can go here
"""


def convert_word_counts():
    """
    Find a way to convert the word counts of a text into a meaningful
    numeric representation that can be used as input to the SVM.

    Maybe this can be a bag-of-words multiple-hot (not one-hot) kind of
    approach where a long sparse matrix with word counts is used.

    But maybe in order to not have sparsity, we can find a better solution.

    :return: numpy array
    """
    pass


def convert_sentence_lengths():
    """
    Find a way to convert the sentence lengths of a text into a meaningful
    numeric representation that can be used as input to the SVM.

    Maybe we can just take the sentence lengths as they are and put
    them into a numpy array

    :return: numpy array
    """
    pass


def combine_features_per_text():
    """
    This function should combine the arrays produced by the conversion
    functions above into one single numpy array (representing one text).

    :return: This should return a numpy array.
    """
    pass


def combine_features_all_texts():
    """
    This function should combine all features of all texts into one
    big numpy array of the shape:

    num_texts * num_total_features

    Also, the function should return the numpy array of candidates,
    which will be our labels.
    Shape should be:

    num_texts * 1

    :return: a Tuple of numpy arrays -> (features, labels)
    """

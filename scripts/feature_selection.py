"""
Here we should implement multiple functions for feature selection.

Below are some examples of functions that care about single features.
"""
from collections import Counter
import text_processing
import numpy as np
import unidecode
from nltk.util import ngrams

def extract_word_counts(text,lemm=True,lang="english"):
    """
    This function requires tokenization and lemmatization.
    text : input text
    lemm : if lemm is True, do the lemmatization.
    lang : language of the sentence. Default language is English
    :return: word_count, count of each word as a dictionary
    """
    ### Use the function from text_processing
    # tokenization
    word_list = text_processing.tokenize_text(text)

    # if lemm == True, do the lemmatization
    if lemm:
        word_list = text_processing.lemmatize_words(word_list, lang)

    # word count
    word_count = Counter(word_list)

    return word_count


def extract_sentence_lengths(text):
    """
    -BLOCKED-

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
    -BLOCKED-

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
    -BLOCKED-

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

    :return: This should return a numpy array of size (1 * num_total_features)
    """
    pass


def combine_features_all_texts():
    """
    The function will be used twice, once for our training data and
    once for our testing data.

    This function should combine all features of all texts into one
    big numpy array of the shape:

    (num_texts * num_total_features)

    Also, the function should return the numpy array of candidates,
    which will be our labels.
    Shape should be:

    (num_texts * 1)

    :return: a Tuple of numpy arrays -> (features, labels)
    """
    pass

def distorted_ngrams(text, n=2):
    """
    Replace some characters with the * symbol.
    Maintain only punctuation marks and diacritical characters.
    And extract ngrams from this text.

    :param text: input text
    :param n: number of items from ngrams, default is bigrams
    :return: dist_ngram, which a dictionary of distortion ngrams
    """

    # set up for punctuations
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # to get distorted text, make unaccented plain text first
    unacc_text = unidecode.unidecode(text)

    dist_text = text
    acc_list = []
    # convert non-diacritical alphabets into *
    for char in text:
        if char == unacc_text[text.index(char)] and char != ' ' and char not in punctuations:
            dist_text = dist_text.replace(char, '*')
        elif char != ' ':
            acc_list.append(char) # collect diacritical characters
    acc_list = list(set(acc_list))

    # make a ngrams
    ngram = list(ngrams(dist_text, n))
    dist_ngram = []

    # from the whole ngrams, leave ngrams only containing diacritical characters or punctuations
    for i in range(n):
        dist_ngram.append(list(filter(lambda x: x[i] in (acc_list or punctuations), ngram)))
    dist_ngram = sum(dist_ngram, []) # to make 1d list

    # counter dictionary for distortion ngrams count
    return Counter(dist_ngram)


if __name__ == "__main__":

    # Testing extract_word_counts
    # text = "I can't help but think that if you were still alive you would have solved it in minutes in the U.S.A., and the victim would still be alive. The bananas cost $4.5, which 77.8% of what apples cost."
    # print(extract_word_counts(text))
    # print(extract_word_counts(text, lemm=False))

    # Testing distortion ngrams
    text = "afrykanerskojęzycznym. Plébiscité sur la toile. Découvrez tous les logiciels à télécharger."
    print(distorted_ngrams(text))
    print(distorted_ngrams(text, 3))
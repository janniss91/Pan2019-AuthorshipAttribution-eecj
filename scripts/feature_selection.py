"""
Here we should implement multiple functions for feature selection.

Below are some examples of functions that care about single features.
"""
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import text_processing
import numpy as np



def extract_word_counts(sentence,lemm=True,lang="english"):
    """
    This function requires tokenization and lemmatization.

    sentence : input sentence
    lemm : if lemm is True, do the lemmatization.
    lang : language of the sentence. Default language is English

    :return: This should return a numpy array.
    """
    #     for char in '-.,\n':
    #         sentence = sentence.replace(char, ' ')
    #     sentence = sentence.lower()

    #     # Tokenize: Split the sentence into words
    #     word_list = nltk.word_tokenize(sentence)
    #     print(word_list)

    #     if lemm:
    #         # Lemmatize list of words and join
    #         lemmatized_output = ' '.join([WordNetLemmatizer().lemmatize(w) for w in word_list])
    #         print(lemmatized_output)
    #         word_list = nltk.word_tokenize(lemmatized_output)

    ### Use the function from text_processing
    word_list = text_processing.tokenize_text(sentence)

    if lemm:
        word_list = text_processing.lemmatize_words(word_list, lang)

        # word count
    word_count = Counter(word_list).most_common()
    print(word_count)

    return np.array(word_count)
    # # using split() function
    # wc = len(sentence.split())
    #
    # # using regex (findall()) function
    # wc = len(re.findall(r'\w+', sentence))
    #
    # # using sum() + strip() + split() function
    # wc = sum([i.strip(string.punctuation).isalpha() for i in
    #            sentence.split()])


def extract_sentence_lengths():
    """
    This function requires tokenization and lemmatization.
    :return: This should return a numpy array.
    """
    pass


"""
... more feature exraction methods can go here
"""


def combine_features():
    """
    This function should combine the (1d) arrays produced by the functions above
    into one single (2d) array.
    The final array can be used as 
    :return: This should return a numpy array (probably 2d).
    """
    pass


if __name__ == "__main__":
    # sent = "bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation."
    sent = "I can't help but think that if you were still alive you would have solved it in minutes in the U.S.A., and the victim would still be alive. The bananas cost $4.5, which 77.8% of what apples cost."
    print(extract_word_counts(sent))
    print(extract_word_counts(sent, lemm=False))
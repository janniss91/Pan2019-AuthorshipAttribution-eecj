"""
Here we should implement multiple functions for feature selection.

Below are some examples of functions that care about single features.
"""
from collections import Counter
import text_processing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import WordPunctTokenizer
from typing import List
import string


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


def char_ngrams(train_texts: List, test_texts: List, ngram_range=(3, 3), min_df=0.12):
    """
    This funciton returns char-ngrams feature in arrays
    train_text : The text used for training
    test_text : The text used for testing
    ngram_range: The range of n. if we want 3-grams and 4-grams, use (3,4)
    min_df : The minimal document frequency

    :return: a tuple of training vector and test vector for all texts
    """
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range,
                                 lowercase=False, min_df=min_df, sublinear_tf=True)
    train_ngram = vectorizer.fit_transform(train_texts)
    test_ngram = vectorizer.transform(test_texts)

    return train_ngram.toarray(), test_ngram.toarray()


def word_ngrams(train_texts: List, test_texts: List, ngram_range=(1, 3), min_df=0.03):
    """
        This funciton returns word-ngrams feature in arrays
        train_text : The texts used for training
        test_text : The texts used for testing
        ngram_range: The range of n. if we want 3-grams and 4-grams, use (3,4)
        min_df : The minimal document frequency

        :return: a tuple of training vector and test vector for all texts
        """
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, tokenizer=WordPunctTokenizer().tokenize,
                                 lowercase=False, min_df=min_df, sublinear_tf=True)
    train_ngram = vectorizer.fit_transform(train_texts)
    test_ngram = vectorizer.transform(test_texts)

    return train_ngram.toarray(), test_ngram.toarray()


def punct_ngrams(train_texts: List[str], test_texts: List[str], ngram_range=(2, 3), min_df=0.12):
    """
    This function sets up punctuation n-grams as features in np array format.

    :param train_texts: The list of training texts
    :param test_texts: The list of test texts
    """
    train_punct_texts = extract_punct_ngrams(train_texts)
    test_punct_texts = extract_punct_ngrams(test_texts)

    vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range,
                                 min_df=min_df)
    train_ngram = vectorizer.fit_transform(train_punct_texts)
    test_ngram = vectorizer.transform(test_punct_texts)

    return train_ngram.toarray(), test_ngram.toarray()


def extract_punct_ngrams(texts: List[str]):
    """
    This function extracts all punctuation from a list of texts.

    The output of these two texts:
    text1 = "'Bla, bla. blub:'"
    text2 = "I,am(using)many=different?punctuation!marks."

    looks like this:
    ["',.:'", ",()=?!."]

    :param texts: The list of actual texts. 
    """
    punct_texts = []
    for text in texts:
        punct_text = ""
        for letter in text:
            if letter in string.punctuation:
                punct_text += letter
        punct_texts.append(punct_text)

    return punct_texts


if __name__ == "__main__":

    # Testing extract_word_counts
    text = "I can't help but think that if you were still alive you would have solved it in minutes in the U.S.A., and the victim would still be alive. The bananas cost $4.5, which 77.8% of what apples cost."
    print(extract_word_counts(text))
    print(extract_word_counts(text, lemm=False))

    # Testing char n-gram function
    train_text1 = "idea."
    train_text2 = "flea."
    test_text1 = "dear."
    test_text2 = "glea."

    cng = char_ngrams([train_text1, train_text2], [test_text1, test_text2])
    print("char n_grams")
    print(cng)

    # Testing word n-gram function
    train_text3 = "I like tables and blue chairs."
    test_text3 = "You like tables and blue skies."

    wng = word_ngrams([train_text3], [test_text3], ngram_range=(3, 3))
    wng2 = word_ngrams([train_text3], [test_text3], ngram_range=(1, 1))
    print("word n_grams")
    print(wng)
    print(wng2)

    # Testing for punctuation n-grams
    train_text4 = "'Bla, bla. blub:'"
    train_text5 = "I,am(using)many=different?punctuation!marks."
    assert extract_punct_ngrams([train_text4, train_text5]) == ["',.:'", ",()=?!."]

    test_text4 = "+,(this'is'-a test !text."
    png = punct_ngrams([train_text4, train_text5], [test_text4], (2, 2))
    print("punctuation n_grams")
    print(png)

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
import unidecode


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


def dist_ngrams(train_text, test_text, n=2, min_df=0.12):
    """
    Replace some characters with the * symbol.
    Maintain only punctuation marks and diacritical characters.
    And extract ngrams from this text.

    :param train_text: texts for train
    :param test_text: texts for test
    :param n: number of ngrams, default is bigrams. If wants trigram, n=3
    Because distortion ngrams can't fit bigram to trigram at the same time.

    :return: arrays of distortion ngrams for train_text and test_text
    """
    if n != 2 or 3:
        raise ValueError("Only bigrams or trigrams available")

    train_dist_text = extract_dist_ngrams(train_text)
    test_dist_text = extract_dist_ngrams(test_text)

    train_dist_text = ngrams_process(train_dist_text, n)
    test_dist_text = ngrams_process(test_dist_text, n)

    ngram_range = (n, n)

    vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range, min_df=min_df)

    train_ngram = vectorizer.fit_transform(train_dist_text)
    test_ngram = vectorizer.transform(test_dist_text)

    return train_ngram.toarray(), test_ngram.toarray()

def extract_dist_ngrams(texts):
    """
    covert normal alphabets in the text into *
    leave only diacritical letters and punctuations

    :param texts: input texts
    :return: converted text
    """
    acc_list = []
    result = []

    for text in texts:
        unacc_text = unidecode.unidecode(text)
        dist_text = text

        for char in text:
            if char == unacc_text[text.index(char)] and char != ' ' and char not in string.punctuation:
                dist_text = dist_text.replace(char, '*')
            elif char != ' ':
                acc_list.append(char)  # collect diacritical characters
        acc_list = list(set(acc_list))
        result.append(dist_text)

    return result

def ngrams_process(dist_texts,n=2):
    """
    convert more than one or two continuous asterisks into one asterisk
    such as '***ę*******. **é***' -> '*ę*. *é*' in bigrams
    or '***ę*******. **é***' -> '**ę**. **é**'

    :param texts: input text which has converted normal alphabets into asterisks
    :param n: number of ngrams, default is 2. Other option is 3 for trigram
    :return: converted text which has no more than one asterisk in the strings
    """
    result = []
    for text in dist_texts:
        if n == 2:
            list_temp = dist_texts[0]
            for i in range(1, len(dist_texts)):
                if list_temp[-1] != dist_texts[i]:
                    list_temp += dist_texts[i]
        if n == 3:
            list_temp = dist_texts[0:2]
            for i in range(2,len(dist_texts)):
                if dist_texts[i] != '*':
                    list_temp += dist_texts[i]
                elif list_temp[-2:] != '**':
                    list_temp += dist_texts[i]

        result.append(list_temp)
    return result


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

    # test for distortion ngrams
    text1 = 'afrykanerskojęzycznym. Plébiscité sur la toile. Découvrez tous les logiciels à télécharger.'
    text2 = 'le vaccin AstraZeneca est autorisé en France, mais n\'est pas recommandé aux plus de 65 ans'
    text3 = 'Contrairement aux deux autres vaccins déjà disponibles, ceux de Pfizer/BioNTech et Moderna, celui d\'AstraZeneca peut être stocké à long terme dans des frigos classiques, ce qui facilite son déploiement logistique.'
    text4 = 'Pascal Soriot et Stéphane Bancel, deux Français expatriés au cœur de la course au vaccin'

    print(dist_ngrams([text1, text2], [text3, text4]))
    print(dist_ngrams([text1, text2], [text3, text4], 3))
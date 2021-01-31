"""
Here we should care about all text preprocessing.
"""
from typing import List

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
nltk.download('wordnet')


def tokenize_text(text: str) -> List[str]:
    """
    A text should be given as input and split into tokens.
    Probably use spacy for this.
    """
    # use regular expression to improve the result
    pattern = r'''(?x)          # set flag to allow verbose regexps
                (?:[A-Z]\.)+        # abbreviations e.g. U.S.A
              | \w+(?:[-']\w+)*     # words with optional internal hyphens or apostrophes e.g. I'm, well-to-do
              | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $3.6, 77%
              | \.\.\.              # ellipsis ...
              | [][.,;"'?():_`-]    # separate tokens
            '''
    tokens = nltk.regexp_tokenize(text, pattern)

    return tokens


def lemmatize_words(tokens: List[str], lang: str) -> List[str]:
    """
    A list of tokens and the langauge should be given as input.
    lang : english, french, italian and spanish.
    A list of lemmas should be returned.
    Probably use spacy for this.
    """
    stemmer = nltk.stem.SnowballStemmer(language=lang.lower())
    lemmas = [stemmer.stem(item) for item in tokens]

    return lemmas


if __name__ == '__main__':
    # Testing
    test_sentence = "I can't help but think that if you were still alive you would have solved it in minutes in the U.S.A., and the victim would still be alive. The bananas cost $4.5, which 77.8% of what apples cost."
    tokens = tokenize_text(test_sentence)
    lemmas = lemmatize_words(tokens, 'english')
    print(tokens)
    print(lemmas)





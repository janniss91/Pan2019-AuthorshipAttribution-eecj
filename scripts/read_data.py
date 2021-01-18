"""
Here we should read all data that we have at our disposition.

Some categories to split up the data:

1. Language
2. Problem
3. Candidate

My idea for the data structure are the three classes below.
Alternatively we could use dictionary structure like this:

{"language1": {"problem000xx": {"candidate1": ["text1", ...], ...}, ...},... }

But this looks a bit less readable to me.
"""


class LanguageData:
    def __init__(self):
        self.problems = []

    def store_problems():
        pass


class Problem:
    def __init__(self):
        self.candidates = []
        # This will be a list of strings.
        self.unknown = []

    def store_candidates():
        pass

    def store_unknown():
        pass


class Candidate:
    def __init__(self):
        self.texts = []

    def read_candidate_texts(dir_name):
        """
        Read all text files in a candidate directory.
        """
        pass


def read_text_file(filename: str) -> str:
    """
    Here we read one text a time and store it as a string.

    I put in an implementation already because it was so easy.
    If you want a different implementation, go ahead and change it.
    """
    with open(filename) as f:
        text = f.read()

        # TODO: Here we should probably remove newlines.

    return text

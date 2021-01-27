import os
from typing import List
import json

DATA_DIR = "data"
UNKOWN_FILE = "unknown"
LANGUAGES = ("english", "french", "italian", "spanish")
CANDIDATE_IDS = ["candidate0000" + str(num) for num in range(1, 10)]
PROBLEM_IDS = [
    ["problem000" + str(num + mult).zfill(2) for num in range(1, 6)]
    for mult in range(0, 20, 5)
]


class Candidate:
    def __init__(self, name: str, problem_id: str):
        self.name = name
        self.problem_id = problem_id
        self.known_texts = {}

        self.store_known_texts()

    def store_known_texts(self):
        """
        Store all text files in a candidate directory.
        """
        candidate_dir = os.path.join(DATA_DIR, self.problem_id, self.name)
        candidate_files = os.listdir(candidate_dir)
        for candidate_file in candidate_files:
            candidate_path = os.path.join(candidate_dir, candidate_file)
            self.known_texts[candidate_file] = read_text_file(candidate_path)

    def __iter__(self):
        return iter([text for text in self.known_texts.values()])


class Problem:
    def __init__(self, name: str):
        self.name = name
        self.candidates = {}
        self.unknown_texts = {}
        self.truths = {}  # gold-standard labels for the unknown text

        self.store_candidates()
        self.store_unknown()
        self.read_ground_true()



    def store_candidates(self):
        for candidate_id in CANDIDATE_IDS:
            self.candidates[candidate_id] = Candidate(candidate_id, self.name)

    def store_unknown(self):
        unknown_dir = os.path.join(DATA_DIR, self.name, UNKOWN_FILE)
        unknown_files = os.listdir(unknown_dir)
        for unknown_file in unknown_files:
            unknown_path = os.path.join(unknown_dir, unknown_file)
            self.unknown_texts[unknown_file] = read_text_file(unknown_path)

    def get_known(self, candidate: Candidate, known: str):
        return self.candidates[candidate].known_texts[known]

    def get_all_known(self):
        all_known = []
        for candidate in self.candidates.values():
            for known in candidate:
                all_known.append(known)

        return all_known

    def __iter__(self):
        return iter([candidate for candidate in self.candidates.values()])

    def read_ground_true(self):
        truth_file = os.path.join(DATA_DIR, self.name, 'ground-truth.json')
        with open(truth_file, 'r') as f:
            ground_truths = json.load(f)['ground_truth']
        for truth in ground_truths:
            self.truths[truth['unknown-text']] = truth['true-author']


class LanguageData:
    def __init__(self, lang_name: str, problem_ids: List[str]):
        self.lang_name = lang_name
        self.problem_ids = problem_ids
        self.problems = {}

        self.store_problems()

    def store_problems(self):
        for problem_id in self.problem_ids:
            self.problems[problem_id] = Problem(problem_id)

    def get_known(self, problem: str, candidate: str, known: str):
        return self.problems[problem].candidates[candidate].known_texts[known]

    def get_all_known(self):
        all_known = []
        for problem in self.problems.values():
            for candidate in problem:
                for known in candidate:
                    all_known.append(known)

        return all_known

    def get_unknown(self, problem: str, unknown: str):
        return self.problems[problem].unknown_texts[unknown]

    def get_all_unknown(self):
        all_unknown = []
        for problem in self.problems.values():
            for unknown in problem.unknown_texts.values():
                all_unknown.append(unknown)

    def __iter__(self):
        return iter([problem for problem in self.problems.values()])


def read_text_file(filename: str) -> str:
    """
    Here we read one text a time and store it as a string.
    Newlines are removed from the text string.
    """
    with open(filename) as f:
        text = f.read().replace("\n", " ")

    return text




if __name__ == "__main__":

    # Testing
    language_data_obs = [
        LanguageData(language, id_list)
        for num, (language, id_list) in enumerate(zip(LANGUAGES, PROBLEM_IDS))
    ]

    english, french, italian, spanish = language_data_obs
    problem1 = english.problems["problem00001"]

    # The number of files in one language should be 5 * 9 * 7 = 315
    assert len(english.get_all_known()) == 315

    # Shortcut to get a single text from language object
    assert (
        english.get_known("problem00001", "candidate00008", "known00005.txt")
        == english.problems["problem00001"]
        .candidates["candidate00008"]
        .known_texts["known00005.txt"]
    )

    # The number of files in one problem should be 9 * 7 = 63
    assert len(problem1.get_all_known()) == 63

    # Shortcut to get a single text from problem object
    assert (
        problem1.get_known("candidate00003", "known00007.txt")
        == english.problems["problem00001"]
        .candidates["candidate00003"]
        .known_texts["known00007.txt"]
    )

    # Test that the correct file content is returned by the get_text function.
    with open ("data/problem00001/candidate00004/known00002.txt") as test_file:
        text142 = test_file.read().replace("\n", " ")
    assert problem1.get_known("candidate00004", "known00002.txt") == text142

    # Test that unknown file is stored correctly.
    with open ("data/problem00001/unknown/unknown00120.txt") as test_file:
        text1u120 = test_file.read().replace("\n", " ")
    assert english.get_unknown("problem00001", "unknown00120.txt") == text1u120
    assert problem1.unknown_texts["unknown00120.txt"] == text1u120

    # Test read ground-truth.json works
    assert problem1.truths["unknown00001.txt"] == "candidate00007"

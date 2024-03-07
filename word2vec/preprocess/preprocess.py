from collections import defaultdict
import enum
import logging
import string
import os
from utils.center_word import CenterWord
import pickle


logging.basicConfig(level=logging.INFO)


class Preprocesser:
    """
    Preprocesses a corpus (text files) in a given directory
        - Generates one hot encoding vectors for each word in the vocab
        - Generates training data tuples of context words paired with center words for each window
    """

    def __init__(self, context_window, directory):
        self.context_window = int(context_window)
        self.directory = directory
        self.vocabulary = defaultdict(list)

        logging.info(
            f"""Starting preprocessor with context_window = {self.context_window}
                        and directory = {self.directory}"""
        )

    def remove_punctuation(self, string_input):
        return string_input.translate(str.maketrans("", "", string.punctuation))

    def generate_training_pairs(self):
        pass

    def process_sentence(self, sentence, filename):
        tokens = sentence.split()
        for index, token in enumerate(tokens):
            negative_interval = max([0, index - self.context_window])
            positive_interval = min([len(tokens), index + 1 + self.context_window])
            behind_center_word = tokens[negative_interval:index]
            after_center_word = (
                tokens[index + 1 : positive_interval]
            )

            self.vocabulary[token].append(
                CenterWord(
                    token, behind_center_word, after_center_word, filename, index
                )
            )


    def read_file(self, filename):
        with open(filename) as f:
            lines = f.read().splitlines()
            lines_without_punctuation = list(
                map(lambda x: self.remove_punctuation(x), lines)
            )
            for sentence in lines_without_punctuation:
                self.process_sentence(sentence, filename)

    def save_vocab(self, output_filename="vocab.pickle"):
            with open(output_filename, "wb") as handle:
                pickle.dump(self.vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def process(self):
        files = os.listdir(self.directory)
        logging.info(f"Found {len(files)} in {self.directory}")
        for filename in files:
            self.read_file(f"{self.directory}{filename}")
        logging.info("Finished generating vocabulary. Saving to disk as output.")
        self.save_vocab()


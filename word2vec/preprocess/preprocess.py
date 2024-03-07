import logging

logging.basicConfig(level=logging.INFO)


class Preprocesser:
    def __init__(self, context_window, directory):
        self.context_window = context_window
        self.directory = directory

        logging.info(
            f"""Starting preprocessor with context_window = {self.context_window}
                        and directory = {self.directory}"""
        )

    def remove_punctuation(self):
        pass

    def generate_training_pairs(self):
        pass

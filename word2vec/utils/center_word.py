class CenterWord:
    def __init__(
        self,
        word: str,
        left_context: list,
        right_context: list,
        document: str,
        document_offset: int,
    ):
        self.word = word
        self.left_context = left_context
        self.right_context = right_context
        self.document = document
        self.document_offset = document_offset

    def __repr__(self):
        return f"""Before: {self.left_context} After: {self.right_context}"""
from multiprocessing import context
import pickle
from torch.utils.data import Dataset, DataLoader
import torch


class WordVectorDataset(Dataset):
    def __init__(self, data_path: str, context_window: int):
        self.data = []
        self.context_window = int(context_window)
        self.vocab = None
        with open(data_path, "rb") as dp:
            self.vocab = pickle.load(dp)
            for _, context in self.vocab.items():
                self.data += context
                self.one_hot_vectors = torch.eye(len(self.vocab), dtype=int)
                self.one_hot_lookup = {
                    word: self.one_hot_vectors[i] for i, word in enumerate(self.vocab)
                }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context_words = self.data[idx].left_context + self.data[idx].right_context
        if len(context_words) != self.context_window:
            padding = [""] * (self.context_window - len(context_words))
            context_words += padding
        return (
            self.one_hot_lookup[context_words[0]],
            self.one_hot_lookup[self.data[idx].word],
        )

from generate_embeddings.word_vector_dataset import WordVectorDataset
from generate_embeddings.word2vec_nn import Word2Vec
from torch.utils.data import DataLoader
import torch
import logging

EMBED_SIZE = 30
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LR = 3e-4
EPOCHS = 1000
loss_fn = torch.nn.CrossEntropyLoss()


class GenerateEmbeddings:
    def __init__(self, data_path, context_window):
        dataset = WordVectorDataset(data_path=data_path, context_window=context_window)
        self.train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        self.model = Word2Vec(len(dataset.vocab), EMBED_SIZE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)
        self.model.to(device)

    def train(self):
        running_loss = []
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for center, context in self.train_dataloader:
                center, context = center.to(device), context.to(device)
                self.optimizer.zero_grad()
                logits = self.model(input=context)
                loss = loss_fn(logits, center)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            epoch_loss /= len(self.train_dataloader)
            running_loss.append(epoch_loss)

            logging.info(f"Epoch Loss: {epoch_loss}")

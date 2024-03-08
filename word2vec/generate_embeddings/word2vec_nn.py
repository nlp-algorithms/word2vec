from torch import nn

class Word2Vec(nn.Module):
    def __init__(self, vocab_len, embed_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_len,embed_size)
        self.expand = nn.Linear(embed_size, vocab_len, bias=False)

    def forward(self,input):
        hidden = self.embed(input)
        logits = self.expand(hidden)
        return logits
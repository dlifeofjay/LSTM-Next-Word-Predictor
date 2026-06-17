import torch
import torch.nn as nn


# Model Architecture
class WordPred(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, embed_dim=64):
        super().__init__()

        # Lets use embaddings
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, vocab_size),
        )

    def forward(self, X):
        emb = self.embed(X)
        _, (h_n, _) = self.lstm(emb)

        return self.fc(h_n[-1])
    


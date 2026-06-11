import torch
import torch.nn as nn


class AuroraLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        hidden_size=1024,
        num_layers=4
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        output, _ = self.lstm(x)

        output = self.fc(output)

        return output
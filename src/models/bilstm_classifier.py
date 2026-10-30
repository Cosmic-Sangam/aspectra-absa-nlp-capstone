"""BiLSTM model for Aspect Sentiment Classification."""
import torch
import torch.nn as nn
from typing import Optional


class BiLSTMSentimentClassifier(nn.Module):
    """BiLSTM for sentence+aspect → sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=2,
            bidirectional=True, batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        embeds = self.dropout(self.embedding(input_ids))

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embeds, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=input_ids.size(1)
            )
        else:
            lstm_out, _ = self.lstm(embeds)

        # Use last valid hidden state
        if lengths is not None:
            idx = (lengths - 1).clamp(min=0).long()
            batch_idx = torch.arange(lstm_out.size(0), device=lstm_out.device)
            pooled = lstm_out[batch_idx, idx]
        else:
            pooled = lstm_out[:, -1, :]

        logits = self.fc(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

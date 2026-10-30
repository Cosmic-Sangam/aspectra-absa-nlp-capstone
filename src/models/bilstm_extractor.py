"""BiLSTM model for Aspect Term Extraction (BIO tagging)."""
import torch
import torch.nn as nn
from typing import Optional


class BiLSTMAspectExtractor(nn.Module):
    """Bidirectional LSTM for token-level BIO tagging."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        num_labels: int = 3,
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
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.num_labels = num_labels

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

        logits = self.classifier(self.dropout(lstm_out))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            target = labels.clone()
            if lengths is not None:
                mask = torch.arange(labels.size(1), device=labels.device).unsqueeze(0) >= lengths.unsqueeze(1)
                target[mask] = -100
            loss = loss_fn(logits.view(-1, self.num_labels), target.view(-1))

        return {"loss": loss, "logits": logits}

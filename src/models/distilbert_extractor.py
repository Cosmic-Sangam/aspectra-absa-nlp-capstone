"""DistilBERT model for Aspect Term Extraction (BIO tagging)."""
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class DistilBertAspectExtractor(nn.Module):
    """DistilBERT + linear head for token-level BIO tagging."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        try:
            self.distilbert = DistilBertModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

"""DistilBERT model for Aspect Sentiment Classification."""
import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertSentimentClassifier(nn.Module):
    """DistilBERT for sentence+aspect → sentiment classification."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        try:
            self.distilbert = DistilBertModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.distilbert = DistilBertModel.from_pretrained(model_name)
        hidden = self.distilbert.config.hidden_size
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.pre_classifier(hidden_state)
        pooled = nn.ReLU()(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

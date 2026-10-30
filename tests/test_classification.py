"""Unit tests for sentiment classification models (BiLSTM & DistilBERT)."""
import tempfile
from pathlib import Path
import pytest
import torch

from src.models.bilstm_classifier import BiLSTMSentimentClassifier
from src.utils.config import SENTIMENT_LABELS


class TestBiLSTMSentimentClassifier:
    """Test BiLSTM Sentiment Classifier."""

    def test_model_init(self):
        model = BiLSTMSentimentClassifier(vocab_size=100, embed_dim=64, hidden_dim=32, num_classes=3)
        assert model is not None
        assert model.num_classes == 3

    def test_forward_pass_and_shape(self):
        model = BiLSTMSentimentClassifier(vocab_size=100, embed_dim=64, hidden_dim=32, num_classes=3)
        input_ids = torch.randint(0, 100, (4, 15))
        lengths = torch.tensor([15, 12, 10, 8])
        out = model(input_ids=input_ids, lengths=lengths)
        assert "logits" in out
        assert out["logits"].shape == (4, 3)

    def test_loss_and_backprop(self):
        model = BiLSTMSentimentClassifier(vocab_size=100, embed_dim=64, hidden_dim=32, num_classes=3)
        input_ids = torch.randint(0, 100, (4, 15))
        lengths = torch.tensor([15, 12, 10, 8])
        labels = torch.tensor([0, 1, 2, 0])
        out = model(input_ids=input_ids, lengths=lengths, labels=labels)
        assert out["loss"] is not None
        assert out["loss"].item() > 0
        out["loss"].backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_one_batch_training_step(self):
        model = BiLSTMSentimentClassifier(vocab_size=100, embed_dim=64, hidden_dim=32, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, 100, (4, 15))
        lengths = torch.tensor([15, 12, 10, 8])
        labels = torch.tensor([0, 1, 2, 1])

        optimizer.zero_grad()
        out = model(input_ids=input_ids, lengths=lengths, labels=labels)
        out["loss"].backward()
        optimizer.step()

        out2 = model(input_ids=input_ids, lengths=lengths, labels=labels)
        assert out2["loss"] is not None

    def test_checkpoint_saving_and_loading(self):
        model = BiLSTMSentimentClassifier(vocab_size=100, embed_dim=64, hidden_dim=32, num_classes=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "classifier.pt"
            torch.save(model.state_dict(), ckpt_path)
            assert ckpt_path.exists()

            loaded_model = BiLSTMSentimentClassifier(vocab_size=100, embed_dim=64, hidden_dim=32, num_classes=3)
            loaded_model.load_state_dict(torch.load(ckpt_path, weights_only=True))

            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                assert torch.equal(p1, p2)

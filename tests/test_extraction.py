"""Unit tests for aspect extraction models (BiLSTM & DistilBERT)."""
import tempfile
from pathlib import Path
import pytest
import torch

from src.models.bilstm_extractor import BiLSTMAspectExtractor
from src.utils.config import BIO_LABELS


class TestBiLSTMExtractor:
    """Test BiLSTM Aspect Extractor."""

    def test_model_init(self):
        model = BiLSTMAspectExtractor(vocab_size=100, embed_dim=64, hidden_dim=32, num_labels=3)
        assert model is not None
        assert model.num_labels == 3

    def test_forward_pass_and_shape(self):
        model = BiLSTMAspectExtractor(vocab_size=100, embed_dim=64, hidden_dim=32, num_labels=3)
        input_ids = torch.randint(0, 100, (2, 10))
        lengths = torch.tensor([10, 8])
        out = model(input_ids=input_ids, lengths=lengths)
        assert "logits" in out
        assert out["logits"].shape == (2, 10, 3)

    def test_loss_and_backprop(self):
        model = BiLSTMAspectExtractor(vocab_size=100, embed_dim=64, hidden_dim=32, num_labels=3)
        input_ids = torch.randint(0, 100, (2, 10))
        lengths = torch.tensor([10, 7])
        labels = torch.randint(0, 3, (2, 10))
        out = model(input_ids=input_ids, lengths=lengths, labels=labels)
        assert out["loss"] is not None
        assert out["loss"].item() > 0
        out["loss"].backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_one_batch_training_step(self):
        model = BiLSTMAspectExtractor(vocab_size=100, embed_dim=64, hidden_dim=32, num_labels=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, 100, (2, 10))
        lengths = torch.tensor([10, 5])
        labels = torch.randint(0, 3, (2, 10))

        optimizer.zero_grad()
        out1 = model(input_ids=input_ids, lengths=lengths, labels=labels)
        loss1 = out1["loss"].item()
        out1["loss"].backward()
        optimizer.step()

        # Step again
        optimizer.zero_grad()
        out2 = model(input_ids=input_ids, lengths=lengths, labels=labels)
        assert out2["loss"] is not None

    def test_checkpoint_saving_and_loading(self):
        model = BiLSTMAspectExtractor(vocab_size=100, embed_dim=64, hidden_dim=32, num_labels=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), ckpt_path)
            assert ckpt_path.exists()

            loaded_model = BiLSTMAspectExtractor(vocab_size=100, embed_dim=64, hidden_dim=32, num_labels=3)
            loaded_model.load_state_dict(torch.load(ckpt_path, weights_only=True))

            # Verify identical weights
            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                assert torch.equal(p1, p2)

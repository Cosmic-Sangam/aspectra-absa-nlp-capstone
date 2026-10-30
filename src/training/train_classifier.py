"""Training loop for sentiment classification models."""
import json
import logging
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.utils.config import (
    MODELS_DIR, PROCESSED_DIR, DEFAULT_SEED, MAX_SEQ_LEN,
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, BILSTM_HIDDEN, BILSTM_EMBED,
    BILSTM_DROPOUT, DEFAULT_BILSTM_LR, DEFAULT_TRANSFORMER_LR,
    DISTILBERT_MODEL, SENTIMENT_LABELS, setup_logging,
)
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.dataset import ClassificationDataset, TransformerClassificationDataset
from src.data.preprocessing import load_processed
from src.models.bilstm_classifier import BiLSTMSentimentClassifier
from src.models.distilbert_classifier import DistilBertSentimentClassifier

logger = setup_logging("train_classifier")


def train_bilstm_classifier(
    domain: str = "restaurant",
    epochs: int = DEFAULT_EPOCHS + 2,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_BILSTM_LR,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Train BiLSTM sentiment classifier."""
    set_seed(seed)
    device = get_device()

    train_data = load_processed(f"{domain}_train_classification")
    test_data = load_processed(f"{domain}_test_classification")

    with open(PROCESSED_DIR / "vocab.json") as f:
        vocab = json.load(f)

    train_ds = ClassificationDataset(train_data, vocab, MAX_SEQ_LEN)
    test_ds = ClassificationDataset(test_data, vocab, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = BiLSTMSentimentClassifier(
        vocab_size=len(vocab),
        embed_dim=BILSTM_EMBED,
        hidden_dim=BILSTM_HIDDEN,
        num_classes=len(SENTIMENT_LABELS),
        dropout=BILSTM_DROPOUT,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    save_dir = MODELS_DIR / "sentiment_classification" / "bilstm"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                lengths=batch["lengths"].to(device),
                labels=batch["labels"].to(device),
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out["loss"].item()

        avg_train = total_loss / len(train_loader)
        history["train_loss"].append(avg_train)

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    lengths=batch["lengths"].to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss += out["loss"].item()
                preds = out["logits"].argmax(dim=-1)
                correct += (preds == batch["labels"].to(device)).sum().item()
                total += batch["labels"].size(0)

        avg_val = val_loss / len(test_loader)
        acc = correct / total if total > 0 else 0
        history["val_loss"].append(avg_val)
        history["val_acc"].append(acc)

        logger.info("Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                     epoch+1, epochs, avg_train, avg_val, acc)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_dir / f"best_{domain}.pt")

    torch.save(model.state_dict(), save_dir / f"final_{domain}.pt")
    with open(save_dir / f"history_{domain}.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


def train_distilbert_classifier(
    domain: str = "restaurant",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 8,
    lr: float = DEFAULT_TRANSFORMER_LR,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Train DistilBERT sentiment classifier."""
    from transformers import DistilBertTokenizerFast

    set_seed(seed)
    device = get_device()

    train_data = load_processed(f"{domain}_train_classification")
    test_data = load_processed(f"{domain}_test_classification")

    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL)

    train_ds = TransformerClassificationDataset(train_data, tokenizer, MAX_SEQ_LEN)
    test_ds = TransformerClassificationDataset(test_data, tokenizer, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = DistilBertSentimentClassifier(
        model_name=DISTILBERT_MODEL, num_classes=len(SENTIMENT_LABELS),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience, patience_counter = 3, 0
    save_dir = MODELS_DIR / "sentiment_classification" / "distilbert"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out["loss"].item()

        avg_train = total_loss / len(train_loader)
        history["train_loss"].append(avg_train)

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss += out["loss"].item()
                preds = out["logits"].argmax(dim=-1)
                correct += (preds == batch["labels"].to(device)).sum().item()
                total += batch["labels"].size(0)

        avg_val = val_loss / len(test_loader)
        acc = correct / total if total > 0 else 0
        history["val_loss"].append(avg_val)
        history["val_acc"].append(acc)

        logger.info("Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                     epoch+1, epochs, avg_train, avg_val, acc)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / f"best_{domain}.pt")
            tokenizer.save_pretrained(str(save_dir))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch+1)
                break

    torch.save(model.state_dict(), save_dir / f"final_{domain}.pt")
    with open(save_dir / f"history_{domain}.json", "w") as f:
        json.dump(history, f, indent=2)

    return history

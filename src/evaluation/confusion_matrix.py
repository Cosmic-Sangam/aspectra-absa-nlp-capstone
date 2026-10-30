"""Confusion matrix plotting."""
import json
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.utils.config import RESULTS_DIR, SENTIMENT_LABELS


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
):
    """Plot and save a confusion matrix."""
    if labels is None:
        labels = SENTIMENT_LABELS

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / "plots" / "confusion_matrix.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_training_history(
    history: dict,
    title: str = "Training History",
    save_path: Optional[Path] = None,
):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    if "val_acc" in history:
        ax2 = ax.twinx()
        ax2.plot(epochs, history["val_acc"], "g-s", label="Val Acc")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="upper left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / "plots" / "training_history.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

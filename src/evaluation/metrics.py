"""Evaluation metrics for ABSA."""
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix as sk_confusion_matrix,
)
import numpy as np


def compute_extraction_metrics(
    true_bio: List[List[str]], pred_bio: List[List[str]]
) -> Dict:
    """Token-level precision, recall, F1 for BIO extraction."""
    y_true, y_pred = [], []
    for t, p in zip(true_bio, pred_bio):
        minlen = min(len(t), len(p))
        y_true.extend(t[:minlen])
        y_pred.extend(p[:minlen])

    labels = ["B-ASP", "I-ASP"]
    return {
        "precision": precision_score(y_true, y_pred, labels=labels, average="micro", zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average="micro", zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0),
    }


def extract_aspect_spans(bio_tags: List[str], tokens: List[str]) -> List[str]:
    """Convert BIO tags to aspect term strings."""
    spans = []
    current = []
    for tag, tok in zip(bio_tags, tokens):
        if tag == "B-ASP":
            if current:
                spans.append(" ".join(current))
            current = [tok]
        elif tag == "I-ASP" and current:
            current.append(tok)
        else:
            if current:
                spans.append(" ".join(current))
                current = []
    if current:
        spans.append(" ".join(current))
    return spans


def compute_exact_match_metrics(
    true_aspects_list: List[List[str]], pred_aspects_list: List[List[str]]
) -> Dict:
    """Exact aspect-term match precision, recall, F1."""
    tp, fp, fn = 0, 0, 0
    for true_set, pred_set in zip(true_aspects_list, pred_aspects_list):
        true_lower = {a.lower() for a in true_set}
        pred_lower = {a.lower() for a in pred_set}
        tp += len(true_lower & pred_lower)
        fp += len(pred_lower - true_lower)
        fn += len(true_lower - pred_lower)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_sentiment_metrics(y_true: List[int], y_pred: List[int], labels=None) -> Dict:
    """Sentiment classification metrics."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class_precision": precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        "per_class_recall": recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def compute_confusion_matrix(y_true: List[int], y_pred: List[int], labels=None) -> np.ndarray:
    """Return confusion matrix."""
    return sk_confusion_matrix(y_true, y_pred, labels=labels)

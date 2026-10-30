"""Run evaluation on test sets and save results."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import torch

from src.utils.config import (
    PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, BIO_ID2LABEL, SENT_ID2LABEL,
    SENTIMENT_LABELS, setup_logging,
)
from src.evaluation.metrics import (
    compute_extraction_metrics, compute_exact_match_metrics,
    compute_sentiment_metrics, extract_aspect_spans, compute_confusion_matrix,
)
from src.evaluation.confusion_matrix import plot_confusion_matrix, plot_training_history
from src.evaluation.error_analysis import analyze_errors
from src.data.preprocessing import load_processed
from src.inference.predictor import BiLSTMPredictor, DistilBertPredictor

logger = setup_logging("evaluate")


def evaluate_extraction(predictor, model_name: str = "bilstm", domain: str = "restaurant") -> Dict:
    """Evaluate aspect extraction on test set."""
    test_data = load_processed(f"{domain}_test_extraction")

    true_bio_all, pred_bio_all = [], []
    true_spans_all, pred_spans_all = [], []

    for sample in test_data:
        tokens = sample["tokens"]
        true_bio = sample["bio_tags"]
        aspects = predictor.extract_aspects(sample["text"])

        # Rebuild predicted BIO from extracted aspects
        pred_bio = ["O"] * len(tokens)
        for asp in aspects:
            asp_tokens = asp["term"].lower().split()
            for i in range(len(tokens)):
                if i + len(asp_tokens) <= len(tokens):
                    match = all(
                        tokens[i + j].lower() == asp_tokens[j]
                        for j in range(len(asp_tokens))
                    )
                    if match:
                        pred_bio[i] = "B-ASP"
                        for j in range(1, len(asp_tokens)):
                            pred_bio[i + j] = "I-ASP"

        true_bio_all.append(true_bio)
        pred_bio_all.append(pred_bio)
        true_spans_all.append(extract_aspect_spans(true_bio, tokens))
        pred_spans_all.append([a["term"] for a in aspects])

    token_metrics = compute_extraction_metrics(true_bio_all, pred_bio_all)
    exact_metrics = compute_exact_match_metrics(true_spans_all, pred_spans_all)

    results = {
        "model": model_name,
        "domain": domain,
        "token_level": token_metrics,
        "exact_match": exact_metrics,
        "num_samples": len(test_data),
    }

    out_dir = RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"extraction_{model_name}_{domain}.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Extraction [%s - %s] — Token F1=%.4f, Exact F1=%.4f",
                model_name, domain, token_metrics["f1"], exact_metrics["f1"])
    return results


def evaluate_classification(predictor, model_name: str = "bilstm", domain: str = "restaurant") -> Dict:
    """Evaluate sentiment classification on test set (gold aspects)."""
    test_data = load_processed(f"{domain}_test_classification")

    y_true, y_pred = [], []
    predictions = []

    for sample in test_data:
        result = predictor.classify_sentiment(sample["text"], sample["aspect_term"])
        true_label = sample["label_id"]
        pred_label = SENTIMENT_LABELS.index(result["sentiment"])
        y_true.append(true_label)
        y_pred.append(pred_label)
        predictions.append({
            "text": sample["text"],
            "aspect": sample["aspect_term"],
            "true": SENT_ID2LABEL[true_label],
            "pred": result["sentiment"],
            "confidence": round(result["confidence"], 4),
            "correct": true_label == pred_label,
        })

    metrics = compute_sentiment_metrics(y_true, y_pred, labels=list(range(len(SENTIMENT_LABELS))))
    cm = compute_confusion_matrix(y_true, y_pred, labels=list(range(len(SENTIMENT_LABELS))))

    results = {
        "model": model_name,
        "domain": domain,
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        "classification_report": metrics["classification_report"],
        "confusion_matrix": cm.tolist(),
        "num_samples": len(test_data),
    }

    out_dir = RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"classification_{model_name}_{domain}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    pred_dir = RESULTS_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"predictions_{model_name}_{domain}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    # Save confusion matrix plot
    plot_path = RESULTS_DIR / "plots" / f"cm_{model_name}_{domain}.png"
    plot_confusion_matrix(
        cm,
        labels=SENTIMENT_LABELS,
        title=f"Confusion Matrix: {model_name.upper()} ({domain.capitalize()})",
        save_path=plot_path,
    )

    # Run error analysis
    analyze_errors(pred_path)

    logger.info("Classification [%s - %s] — Acc=%.4f, Macro-F1=%.4f",
                model_name, domain, metrics["accuracy"], metrics["macro_f1"])
    return results


def plot_all_histories():
    """Plot training history curves for all trained models."""
    plot_dir = RESULTS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for task in ["aspect_extraction", "sentiment_classification"]:
        for model in ["bilstm", "distilbert"]:
            for domain in ["restaurant", "laptop"]:
                hist_path = MODELS_DIR / task / model / f"history_{domain}.json"
                if hist_path.exists():
                    try:
                        with open(hist_path) as f:
                            hist = json.load(f)
                        out_plot = plot_dir / f"loss_{task}_{model}_{domain}.png"
                        plot_training_history(
                            hist,
                            title=f"{task.replace('_', ' ').title()} ({model.upper()} - {domain})",
                            save_path=out_plot,
                        )
                        logger.info("Saved history plot: %s", out_plot.name)
                    except Exception as e:
                        logger.warning("Could not plot history for %s: %s", hist_path.name, e)


def evaluate_all():
    """Run full evaluation suite across all models and domains."""
    logger.info("Starting complete evaluation...")
    summary = {}

    for domain in ["restaurant", "laptop"]:
        # BiLSTM
        logger.info("Evaluating BiLSTM on %s...", domain)
        bilstm_pred = BiLSTMPredictor(domain=domain)
        ext_bi = evaluate_extraction(bilstm_pred, "bilstm", domain)
        cls_bi = evaluate_classification(bilstm_pred, "bilstm", domain)

        # DistilBERT
        logger.info("Evaluating DistilBERT on %s...", domain)
        distilbert_pred = DistilBertPredictor(domain=domain)
        ext_bert = evaluate_extraction(distilbert_pred, "distilbert", domain)
        cls_bert = evaluate_classification(distilbert_pred, "distilbert", domain)

        summary[domain] = {
            "bilstm": {
                "extraction_exact_f1": ext_bi["exact_match"]["f1"],
                "extraction_token_f1": ext_bi["token_level"]["f1"],
                "sentiment_accuracy": cls_bi["metrics"]["accuracy"],
                "sentiment_macro_f1": cls_bi["metrics"]["macro_f1"],
            },
            "distilbert": {
                "extraction_exact_f1": ext_bert["exact_match"]["f1"],
                "extraction_token_f1": ext_bert["token_level"]["f1"],
                "sentiment_accuracy": cls_bert["metrics"]["accuracy"],
                "sentiment_macro_f1": cls_bert["metrics"]["macro_f1"],
            },
        }

    # Save overall comparison
    summary_path = RESULTS_DIR / "metrics" / "overall_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Plot training curves
    plot_all_histories()

    logger.info("Evaluation complete! Summary saved to %s", summary_path)
    return summary


if __name__ == "__main__":
    evaluate_all()

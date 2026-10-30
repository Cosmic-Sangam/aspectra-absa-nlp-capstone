"""Lightweight explainability: token importance via leave-one-out."""
from typing import Dict, List
import torch


def leave_one_out_importance(predictor, text: str, aspect: str) -> List[Dict]:
    """Compute word importance by leave-one-out perturbation.

    For each word, remove it from the text and measure the change in
    predicted confidence for the aspect's sentiment.
    """
    baseline = predictor.classify_sentiment(text, aspect)
    baseline_conf = baseline["confidence"]
    baseline_sent = baseline["sentiment"]

    words = text.split()
    importance = []

    for i, word in enumerate(words):
        perturbed = " ".join(words[:i] + words[i + 1:])
        if not perturbed.strip():
            importance.append({"word": word, "importance": 0.0})
            continue
        result = predictor.classify_sentiment(perturbed, aspect)
        # Importance = drop in confidence for the baseline sentiment
        drop = baseline_conf - result["probabilities"].get(baseline_sent, 0.0)
        importance.append({"word": word, "importance": round(drop, 4)})

    # Sort by absolute importance
    importance.sort(key=lambda x: abs(x["importance"]), reverse=True)
    return importance


def get_explanation(predictor, text: str, aspect: str, top_k: int = 5) -> Dict:
    """Generate a human-readable explanation."""
    imp = leave_one_out_importance(predictor, text, aspect)
    sent_info = predictor.classify_sentiment(text, aspect)

    top_words = imp[:top_k]
    positive_influence = [w for w in top_words if w["importance"] > 0]
    negative_influence = [w for w in top_words if w["importance"] < 0]

    explanation = {
        "aspect": aspect,
        "predicted_sentiment": sent_info["sentiment"],
        "confidence": round(sent_info["confidence"], 4),
        "top_influential_words": top_words,
        "supporting_words": [w["word"] for w in positive_influence],
        "opposing_words": [w["word"] for w in negative_influence],
        "method": "leave-one-out perturbation",
        "note": "Importance > 0 means the word supports the prediction; < 0 means it opposes it.",
    }
    return explanation

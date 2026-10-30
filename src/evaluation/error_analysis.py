"""Error analysis utilities."""
import json
from pathlib import Path
from typing import List, Dict

from src.utils.config import RESULTS_DIR, SENT_ID2LABEL, setup_logging

logger = setup_logging("error_analysis")


def analyze_errors(predictions_path: Path) -> Dict:
    """Analyze classification errors from saved predictions."""
    with open(predictions_path) as f:
        predictions = json.load(f)

    errors = [p for p in predictions if not p["correct"]]
    correct = [p for p in predictions if p["correct"]]

    # Error breakdown by true label
    error_by_true = {}
    for e in errors:
        t = e["true"]
        error_by_true.setdefault(t, []).append(e)

    # Common confusion patterns
    confusion_pairs = {}
    for e in errors:
        key = f"{e['true']} → {e['pred']}"
        confusion_pairs[key] = confusion_pairs.get(key, 0) + 1

    # Low confidence errors vs high confidence errors
    low_conf_errors = [e for e in errors if e["confidence"] < 0.6]
    high_conf_errors = [e for e in errors if e["confidence"] >= 0.6]

    analysis = {
        "total_predictions": len(predictions),
        "total_correct": len(correct),
        "total_errors": len(errors),
        "error_rate": len(errors) / len(predictions) if predictions else 0,
        "errors_by_true_label": {k: len(v) for k, v in error_by_true.items()},
        "confusion_pairs": dict(sorted(confusion_pairs.items(), key=lambda x: -x[1])),
        "low_confidence_errors": len(low_conf_errors),
        "high_confidence_errors": len(high_conf_errors),
        "sample_errors": errors[:20],
    }

    # Save
    out_dir = RESULTS_DIR / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (predictions_path.stem.replace("predictions", "error_analysis") + ".json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info("Error analysis: %d/%d errors (%.1f%%)",
                len(errors), len(predictions),
                analysis["error_rate"] * 100)
    return analysis

"""Unit tests for ABSA pipeline and explainability."""
import json
import pytest

from src.inference.pipeline import ABSAPipeline
from src.inference.explainability import leave_one_out_importance, get_explanation


@pytest.fixture(scope="module")
def bilstm_pipeline():
    """Create a BiLSTM pipeline instance for testing."""
    return ABSAPipeline(model_type="bilstm", domain="restaurant", confidence_threshold=0.2)


def test_pipeline_initialization(bilstm_pipeline):
    """Test pipeline initialization."""
    assert bilstm_pipeline.model_type == "bilstm"
    assert bilstm_pipeline.domain == "restaurant"
    assert bilstm_pipeline.threshold == 0.2
    assert bilstm_pipeline.predictor is not None


def test_pipeline_empty_input(bilstm_pipeline):
    """Test handling of empty or whitespace input."""
    res_empty = bilstm_pipeline.analyze("")
    assert res_empty["aspects"] == []
    assert "error" in res_empty

    res_spaces = bilstm_pipeline.analyze("   ")
    assert res_spaces["aspects"] == []
    assert "error" in res_spaces


def test_pipeline_prediction_structure(bilstm_pipeline):
    """Test that pipeline output follows the expected schema."""
    text = "The food was delicious and the ambiance was wonderful."
    res = bilstm_pipeline.analyze(text)

    assert "review" in res
    assert res["review"] == text
    assert "aspects" in res
    assert "model" in res
    assert "domain" in res
    assert isinstance(res["aspects"], list)

    for aspect in res["aspects"]:
        assert "aspect" in aspect
        assert "sentiment" in aspect
        assert aspect["sentiment"] in ["positive", "negative", "neutral"]
        assert "confidence" in aspect
        assert 0.0 <= aspect["confidence"] <= 1.0


def test_pipeline_batch_analysis(bilstm_pipeline):
    """Test batch analysis."""
    texts = [
        "The pizza is great.",
        "Terrible service and rude staff.",
        "Average place.",
    ]
    results = bilstm_pipeline.analyze_batch(texts)
    assert len(results) == len(texts)
    for r in results:
        assert "aspects" in r


def test_pipeline_to_json(bilstm_pipeline):
    """Test JSON serialization."""
    text = "Great battery life."
    res = bilstm_pipeline.analyze(text)
    json_str = bilstm_pipeline.to_json(res)
    parsed = json.loads(json_str)
    assert parsed["review"] == text
    assert "aspects" in parsed


def test_pipeline_confidence_threshold():
    """Test that aspects below threshold are filtered out."""
    high_thresh = ABSAPipeline(model_type="bilstm", domain="restaurant", confidence_threshold=0.999)
    res = high_thresh.analyze("The food was good.")
    # With a 99.9% threshold, either empty or only super confident aspects remain
    for asp in res["aspects"]:
        assert asp["confidence"] >= 0.999


def test_explainability(bilstm_pipeline):
    """Test leave-one-out explainability."""
    text = "The food was delicious"
    aspect = "food"
    importance = leave_one_out_importance(bilstm_pipeline.predictor, text, aspect)

    assert isinstance(importance, list)
    assert len(importance) > 0
    assert "word" in importance[0]
    assert "importance" in importance[0]

    explanation = get_explanation(bilstm_pipeline.predictor, text, aspect, top_k=3)
    assert explanation["aspect"] == aspect
    assert "predicted_sentiment" in explanation
    assert "top_influential_words" in explanation
    assert len(explanation["top_influential_words"]) <= 3

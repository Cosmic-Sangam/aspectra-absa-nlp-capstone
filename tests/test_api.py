"""Unit tests for FastAPI endpoints and request/response models."""
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api.main import (
    health_check,
    list_models,
    predict,
    predict_batch,
    PredictRequest,
    BatchPredictRequest,
    PredictResponse,
    BatchPredictResponse,
)


def test_health_endpoint():
    """Test health check returns system status and supported models."""
    res = health_check()
    assert res["status"] == "healthy"
    assert "device" in res
    assert "bilstm" in res["supported_models"]
    assert "distilbert" in res["supported_models"]
    assert "restaurant" in res["supported_domains"]
    assert "laptop" in res["supported_domains"]


def test_models_endpoint():
    """Test models listing endpoint."""
    res = list_models()
    assert "models" in res
    assert "domains" in res
    model_ids = [m["id"] for m in res["models"]]
    assert "bilstm" in model_ids
    assert "distilbert" in model_ids


def test_predict_bilstm_restaurant():
    """Test single prediction on restaurant domain with BiLSTM."""
    req = PredictRequest(
        review="The pizza was delicious and the pasta was great.",
        model="bilstm",
        domain="restaurant",
        explain=False,
        confidence_threshold=0.2,
    )
    res = predict(req)
    assert isinstance(res, PredictResponse)
    assert res.model == "bilstm"
    assert res.domain == "restaurant"
    assert res.latency_ms > 0
    assert len(res.aspects) > 0
    for asp in res.aspects:
        assert asp.aspect
        assert asp.sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= asp.confidence <= 1.0


def test_predict_with_explainability():
    """Test prediction with explainability flag enabled."""
    req = PredictRequest(
        review="The battery life is exceptional.",
        model="bilstm",
        domain="laptop",
        explain=True,
        confidence_threshold=0.1,
    )
    res = predict(req)
    assert isinstance(res, PredictResponse)
    assert len(res.aspects) > 0
    assert res.aspects[0].explanation is not None
    assert "predicted_sentiment" in res.aspects[0].explanation
    assert "top_influential_words" in res.aspects[0].explanation


def test_predict_batch():
    """Test batch prediction endpoint."""
    req = BatchPredictRequest(
        reviews=[
            "Great screen and lightweight body.",
            "Terrible customer service and long wait times.",
        ],
        model="bilstm",
        domain="laptop",
        confidence_threshold=0.2,
    )
    res = predict_batch(req)
    assert isinstance(res, BatchPredictResponse)
    assert res.total_reviews == 2
    assert len(res.results) == 2
    for r in res.results:
        assert r.domain == "laptop"
        assert r.latency_ms >= 0


def test_predict_invalid_model():
    """Test predict with invalid model name raises HTTPException 400."""
    with pytest.raises(HTTPException) as exc_info:
        req = PredictRequest(
            review="Great screen.",
            model="unknown_transformer",
            domain="laptop",
        )
        predict(req)
    assert exc_info.value.status_code == 400


def test_predict_invalid_domain():
    """Test predict with invalid domain raises HTTPException 400."""
    with pytest.raises(HTTPException) as exc_info:
        req = PredictRequest(
            review="Great car engine.",
            model="bilstm",
            domain="automotive",
        )
        predict(req)
    assert exc_info.value.status_code == 400


def test_predict_empty_review_fails_validation():
    """Test Pydantic validation fails when review text is empty."""
    with pytest.raises(ValidationError):
        PredictRequest(
            review="",
            model="bilstm",
            domain="restaurant",
        )

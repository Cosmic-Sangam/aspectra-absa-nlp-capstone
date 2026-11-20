"""FastAPI backend for Aspect-Based Sentiment Analysis (ABSA)."""
import time
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.pipeline import ABSAPipeline
from src.inference.explainability import get_explanation
from src.utils.device import get_device

# Global pipeline cache: (model_type, domain) -> ABSAPipeline
pipelines: Dict[str, ABSAPipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown event."""
    yield
    pipelines.clear()


app = FastAPI(
    title="Aspect-Based Sentiment Analysis API",
    description="Production-grade ABSA API supporting BiLSTM and DistilBERT models with explainability.",
    version="1.0.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline(model_type: str = "distilbert", domain: str = "restaurant") -> ABSAPipeline:
    """Retrieve or lazily initialize pipeline."""
    key = f"{model_type}_{domain}"
    if key not in pipelines:
        pipelines[key] = ABSAPipeline(model_type=model_type, domain=domain)
    return pipelines[key]


# Request / Response Schemas
class PredictRequest(BaseModel):
    review: str = Field(..., description="The review text to analyze", min_length=1)
    model: str = Field("distilbert", description="Model architecture: 'distilbert' or 'bilstm'")
    domain: str = Field("restaurant", description="Domain: 'restaurant' or 'laptop'")
    explain: bool = Field(False, description="Whether to include token-level explainability")
    confidence_threshold: float = Field(0.3, description="Minimum confidence threshold", ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    reviews: List[str] = Field(..., description="List of review texts", min_length=1)
    model: str = Field("distilbert", description="Model architecture: 'distilbert' or 'bilstm'")
    domain: str = Field("restaurant", description="Domain: 'restaurant' or 'laptop'")
    confidence_threshold: float = Field(0.3, description="Minimum confidence threshold", ge=0.0, le=1.0)


class AspectResponse(BaseModel):
    aspect: str
    sentiment: str
    confidence: float
    char_start: int = -1
    char_end: int = -1
    explanation: Optional[Dict] = None


class PredictResponse(BaseModel):
    review: str
    model: str
    domain: str
    aspects: List[AspectResponse]
    latency_ms: float


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_reviews: int
    latency_ms: float


@app.get("/health")
def health_check():
    """Health check endpoint reporting device and loaded models."""
    device = str(get_device())
    return {
        "status": "healthy",
        "device": device,
        "loaded_pipelines": list(pipelines.keys()),
        "supported_models": ["bilstm", "distilbert"],
        "supported_domains": ["restaurant", "laptop"],
    }


@app.get("/models")
def list_models():
    """Return available models and supported features."""
    return {
        "models": [
            {
                "id": "distilbert",
                "name": "DistilBERT (Transformer)",
                "description": "Pre-trained Transformer fine-tuned for aspect extraction and sentiment classification.",
                "recommended": True,
            },
            {
                "id": "bilstm",
                "name": "BiLSTM (Bidirectional LSTM)",
                "description": "Lightweight sequence model with Word2Vec-style embeddings.",
                "recommended": False,
            },
        ],
        "domains": [
            {"id": "restaurant", "name": "Restaurant Reviews (SemEval 2014)"},
            {"id": "laptop", "name": "Laptop Customer Reviews (SemEval 2014)"},
        ],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Analyze a single review for aspect terms and sentiments."""
    start_time = time.time()

    model_type = request.model.lower()
    if model_type not in ["bilstm", "distilbert"]:
        raise HTTPException(status_code=400, detail="Model must be 'bilstm' or 'distilbert'")

    domain = request.domain.lower()
    if domain not in ["restaurant", "laptop"]:
        raise HTTPException(status_code=400, detail="Domain must be 'restaurant' or 'laptop'")

    pipeline = get_pipeline(model_type, domain)
    pipeline.threshold = request.confidence_threshold

    analysis = pipeline.analyze(request.review)

    aspects_out = []
    for a in analysis.get("aspects", []):
        aspect_dict = {
            "aspect": a["aspect"],
            "sentiment": a["sentiment"],
            "confidence": a["confidence"],
            "char_start": a.get("char_start", -1),
            "char_end": a.get("char_end", -1),
            "explanation": None,
        }
        if request.explain:
            try:
                aspect_dict["explanation"] = get_explanation(
                    pipeline.predictor, request.review, a["aspect"]
                )
            except Exception as e:
                aspect_dict["explanation"] = {"error": str(e)}

        aspects_out.append(AspectResponse(**aspect_dict))

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    return PredictResponse(
        review=request.review,
        model=model_type,
        domain=domain,
        aspects=aspects_out,
        latency_ms=elapsed_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    """Analyze a batch of reviews."""
    start_time = time.time()

    model_type = request.model.lower()
    if model_type not in ["bilstm", "distilbert"]:
        raise HTTPException(status_code=400, detail="Model must be 'bilstm' or 'distilbert'")

    domain = request.domain.lower()
    if domain not in ["restaurant", "laptop"]:
        raise HTTPException(status_code=400, detail="Domain must be 'restaurant' or 'laptop'")

    pipeline = get_pipeline(model_type, domain)
    pipeline.threshold = request.confidence_threshold

    results = []
    for rev in request.reviews:
        t0 = time.time()
        analysis = pipeline.analyze(rev)
        aspects_out = [
            AspectResponse(
                aspect=a["aspect"],
                sentiment=a["sentiment"],
                confidence=a["confidence"],
                char_start=a.get("char_start", -1),
                char_end=a.get("char_end", -1),
            )
            for a in analysis.get("aspects", [])
        ]
        t_lat = round((time.time() - t0) * 1000, 2)
        results.append(PredictResponse(
            review=rev,
            model=model_type,
            domain=domain,
            aspects=aspects_out,
            latency_ms=t_lat,
        ))

    total_elapsed = round((time.time() - start_time) * 1000, 2)
    return BatchPredictResponse(
        results=results,
        total_reviews=len(request.reviews),
        latency_ms=total_elapsed,
    )


# Mount compiled frontend if available (allows single-command deployment)
from pathlib import Path
from fastapi.staticfiles import StaticFiles

frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)

"""End-to-end ABSA pipeline: text → aspects with sentiments."""
import json
from typing import Dict, List, Optional

from src.inference.predictor import BiLSTMPredictor, DistilBertPredictor


class ABSAPipeline:
    """Unified pipeline for aspect-based sentiment analysis."""

    def __init__(
        self,
        model_type: str = "bilstm",
        domain: str = "restaurant",
        confidence_threshold: float = 0.3,
        device=None,
    ):
        self.model_type = model_type
        self.domain = domain
        self.threshold = confidence_threshold

        if model_type == "distilbert":
            self.predictor = DistilBertPredictor(domain=domain, device=device)
        else:
            self.predictor = BiLSTMPredictor(domain=domain, device=device)

    def analyze(self, text: str) -> Dict:
        """Analyze a review and return structured output."""
        if not text or not text.strip():
            return {"review": text, "aspects": [], "error": "Empty input"}

        result = self.predictor.predict(text.strip())

        # Apply confidence threshold
        filtered = [
            a for a in result["aspects"]
            if a["confidence"] >= self.threshold
        ]

        # Remove duplicates (same aspect term)
        seen = set()
        unique = []
        for a in filtered:
            key = a["aspect"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)

        result["aspects"] = unique
        result["model"] = self.model_type
        result["domain"] = self.domain
        return result

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple reviews."""
        return [self.analyze(t) for t in texts]

    def to_json(self, result: Dict) -> str:
        """Serialize result to JSON string."""
        return json.dumps(result, indent=2, ensure_ascii=False)

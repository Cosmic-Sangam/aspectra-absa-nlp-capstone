"""End-to-end ABSA pipeline: text → aspects with sentiments."""
import json
import re
from typing import Dict, List, Optional

from src.inference.predictor import BiLSTMPredictor, DistilBertPredictor

COMMON_ASPECTS = {
    "restaurant": [
        "wait time", "food quality", "food", "service", "staff", "waiter", "waitress",
        "ambiance", "atmosphere", "price", "drinks", "wine", "dessert", "place",
        "decor", "menu", "taste", "speed", "experience", "cleanliness"
    ],
    "laptop": [
        "battery life", "battery", "screen", "display", "keyboard", "trackpad",
        "performance", "speed", "price", "build quality", "design", "weight",
        "sound", "speaker", "fan noise", "fan", "heat", "software", "ram", "storage"
    ],
}


def find_candidate_aspects(text: str, domain: str = "restaurant") -> List[str]:
    """Fallback candidate aspect extractor for informal or implicit reviews."""
    text_lower = text.lower()
    found: List[str] = []

    # 1. Match common domain aspects
    domain_aspects = COMMON_ASPECTS.get(domain, COMMON_ASPECTS["restaurant"])
    for asp in sorted(domain_aspects, key=len, reverse=True):
        if re.search(r"\b" + re.escape(asp) + r"\b", text_lower):
            if not any(asp in existing for existing in found):
                found.append(asp)

    # 2. Heuristics for verbal or colloquial expressions
    if re.search(r"\bwait(?:ed|ing)?\b", text_lower) and "wait time" not in found:
        found.append("wait time")
    if re.search(r"\b(cook(?:ed)?|taste(?:d)?|delicious|tasty|fresh)\b", text_lower) and "food" not in found:
        found.append("food")
    if re.search(r"\b(slow|fast|quick|rude|friendly|polite|helpful)\b", text_lower) and "service" not in found:
        found.append("service")

    # 3. Extract noun phrases after articles (the/a/an)
    matches = re.findall(r"\b(?:the|a|an)\s+([a-z]+(?:\s+[a-z]+)?)\b", text_lower)
    stopwords = {"this", "that", "with", "from", "other", "very", "much", "more", "first", "next", "last", "same"}
    for m in matches:
        words = [w for w in m.split() if w not in stopwords]
        if words:
            phrase = " ".join(words)
            if len(phrase) > 2 and phrase not in found and not any(phrase in f for f in found):
                found.append(phrase)

    # 4. If still nothing found, provide 'overall experience'
    if not found:
        found.append("overall experience")

    return found[:3]


class ABSAPipeline:
    """Unified pipeline for aspect-based sentiment analysis."""

    def __init__(
        self,
        model_type: str = "distilbert",
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
        """Analyze a review and return structured output with intelligent fallback."""
        if not text or not text.strip():
            return {"review": text, "aspects": [], "error": "Empty input"}

        clean_text = text.strip()
        result = self.predictor.predict(clean_text)
        aspects = result.get("aspects", [])

        # Robust Fallback: If sequence tagger extracted 0 aspects, use candidate extraction
        if not aspects:
            candidates = find_candidate_aspects(clean_text, self.domain)
            for cand in candidates:
                sent_info = self.predictor.classify_sentiment(clean_text, cand)
                idx = clean_text.lower().find(cand.lower())
                c_start = idx if idx >= 0 else 0
                c_end = (idx + len(cand)) if idx >= 0 else len(clean_text)
                aspects.append({
                    "aspect": cand,
                    "sentiment": sent_info["sentiment"],
                    "confidence": round(sent_info["confidence"], 4),
                    "char_start": c_start,
                    "char_end": c_end,
                })

        # Apply confidence threshold
        filtered = [
            a for a in aspects
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

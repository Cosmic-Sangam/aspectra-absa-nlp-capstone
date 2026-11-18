"""End-to-end ABSA pipeline: text → aspects with sentiments."""
import json
import re
from typing import Dict, List, Optional, Tuple

from src.inference.predictor import BiLSTMPredictor, DistilBertPredictor

# Comprehensive domain patterns and aspects
DOMAIN_PATTERNS = {
    "restaurant": [
        (r"\bworth(?:\s+it)?\b", "worth it", "worth"),
        (r"\b(?:wait(?:ed|ing)?|long time|queue|delay)\b", "wait time", "wait"),
        (r"\b(?:food|pizza|pasta|sushi|burger|dish|meal|steak|dessert|salad|soup|meat|fish|bread|appetizer)\b", None, "food"),
        (r"\b(?:cook(?:ed)?|tast(?:e|y|ed|ing)|delicious|flavor|fresh|bland|yummy)\b", "food quality", "taste"),
        (r"\b(?:service|server|waiter|waitress|staff|bartender|hostess|manager)\b", "service", "service"),
        (r"\b(?:price|pricing|cost|expensive|cheap|overpriced|affordable|bill|check)\b", "price", "price"),
        (r"\b(?:ambian(?:ce|t)|atmosphere|vibe|decor|music|interior|lighting|patio|seating|room)\b", "ambiance", "ambiance"),
        (r"\b(?:drink|cocktail|beer|wine|beverage|coffee|tea|bar)\b", "drinks", "drinks"),
        (r"\b(?:clean(?:liness)?|dirty|filthy|hygien(?:e|ic)|smell)\b", "cleanliness", "clean"),
        (r"\b(?:portion|quantity|size)\b", "portion size", "portion"),
    ],
    "laptop": [
        (r"\bworth(?:\s+it)?\b", "worth it", "worth"),
        (r"\b(?:battery(?:\s+life)?|charge|charging|drain|adapter|charger)\b", "battery life", "battery"),
        (r"\b(?:screen|display|monitor|resolution|oled|panel|refresh rate|brightness)\b", "screen", "screen"),
        (r"\b(?:keyboard|keys|keypad|typing|backlight)\b", "keyboard", "keyboard"),
        (r"\b(?:trackpad|touchpad|mouse)\b", "trackpad", "trackpad"),
        (r"\b(?:performance|speed|fast|slow|lag|laggy|fps|multitask(?:ing)?|snappy|processor|cpu|gpu)\b", "performance", "performance"),
        (r"\b(?:heat|thermal|hot|warm|overheat(?:s|ing)?|cooling|fan(?:\s+noise)?)\b", "cooling / fan", "heat"),
        (r"\b(?:build(?:\s+quality)?|chassis|hinge|body|durab(?:le|ility)|lightweight|heavy|weight)\b", "build quality", "build"),
        (r"\b(?:speaker|audio|sound|volume|mic(?:rophone)?)\b", "sound quality", "speaker"),
        (r"\b(?:price|cost|expensive|cheap|value|deal|affordable)\b", "price", "price"),
        (r"\b(?:software|windows|os|driver|bloatware|updates)\b", "software", "software"),
    ],
}


def find_candidate_aspects(text: str, domain: str = "restaurant") -> List[Dict]:
    """Extract candidate aspects with accurate non-overlapping char spans."""
    text_lower = text.lower()
    patterns = DOMAIN_PATTERNS.get(domain, DOMAIN_PATTERNS["restaurant"])
    extracted: List[Dict] = []
    claimed_spans: List[Tuple[int, int]] = []

    def spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
        return max(s1, s2) < min(e1, e2)

    for regex_pat, normalized_name, fallback_keyword in patterns:
        for match in re.finditer(regex_pat, text_lower):
            s_idx, e_idx = match.start(), match.end()
            if any(spans_overlap(s_idx, e_idx, cs, ce) for cs, ce in claimed_spans):
                continue

            term = normalized_name if normalized_name else match.group(0)
            # Avoid duplicate aspects
            if any(term.lower() == e["aspect"].lower() for e in extracted):
                continue

            claimed_spans.append((s_idx, e_idx))
            extracted.append({
                "aspect": term,
                "char_start": s_idx,
                "char_end": e_idx,
            })

    # If no pattern matched, extract noun phrases after articles (the/a/an)
    if not extracted:
        matches = re.finditer(r"\b(?:the|a|an)\s+([a-z]+(?:\s+[a-z]+)?)\b", text_lower)
        stopwords = {"this", "that", "with", "from", "other", "very", "much", "more", "first", "next", "last", "same"}
        for m in matches:
            phrase = m.group(1).strip()
            words = [w for w in phrase.split() if w not in stopwords]
            if words:
                clean_phrase = " ".join(words)
                s_idx = text_lower.find(clean_phrase)
                if s_idx >= 0 and len(clean_phrase) > 2:
                    e_idx = s_idx + len(clean_phrase)
                    if not any(spans_overlap(s_idx, e_idx, cs, ce) for cs, ce in claimed_spans):
                        claimed_spans.append((s_idx, e_idx))
                        extracted.append({
                            "aspect": clean_phrase,
                            "char_start": s_idx,
                            "char_end": e_idx,
                        })

    # If still completely empty, fallback to overall experience
    if not extracted:
        extracted.append({
            "aspect": "overall experience",
            "char_start": 0,
            "char_end": min(len(text), 15),
        })

    return extracted[:4]


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
        """Analyze a review and return structured output with intelligent extraction."""
        if not text or not text.strip():
            return {"review": text, "aspects": [], "error": "Empty input"}

        clean_text = text.strip()
        result = self.predictor.predict(clean_text)
        aspects = result.get("aspects", [])

        # If model extraction returned zero aspects OR missed key contrastive aspects,
        # augment with candidate extractor to capture all sentiments (e.g. "worth it", "wait time")
        candidates = find_candidate_aspects(clean_text, self.domain)
        existing_aspects_lower = {a["aspect"].lower() for a in aspects}

        if not aspects:
            # Fallback mode
            for cand in candidates:
                sent_info = self.predictor.classify_sentiment(clean_text, cand["aspect"])
                aspects.append({
                    "aspect": cand["aspect"],
                    "sentiment": sent_info["sentiment"],
                    "confidence": round(sent_info["confidence"], 4),
                    "char_start": cand["char_start"],
                    "char_end": cand["char_end"],
                })
        else:
            # Augment mode: if contrast words exist ("worth it", "value") and weren't in BIO tags
            for cand in candidates:
                if cand["aspect"].lower() not in existing_aspects_lower and cand["aspect"].lower() in ["worth it", "value"]:
                    sent_info = self.predictor.classify_sentiment(clean_text, cand["aspect"])
                    aspects.append({
                        "aspect": cand["aspect"],
                        "sentiment": sent_info["sentiment"],
                        "confidence": round(sent_info["confidence"], 4),
                        "char_start": cand["char_start"],
                        "char_end": cand["char_end"],
                    })

        # Apply confidence threshold
        filtered = [
            a for a in aspects
            if a["confidence"] >= self.threshold
        ]

        # Remove duplicates
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

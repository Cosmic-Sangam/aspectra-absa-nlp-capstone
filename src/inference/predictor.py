"""Predictor classes — load trained models and run inference."""
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

from src.utils.config import (
    MODELS_DIR, PROCESSED_DIR, DISTILBERT_MODEL, MAX_SEQ_LEN,
    BIO_LABELS, BIO_ID2LABEL, SENTIMENT_LABELS, SENT_ID2LABEL,
)
from src.utils.device import get_device
from src.models.bilstm_extractor import BiLSTMAspectExtractor
from src.models.bilstm_classifier import BiLSTMSentimentClassifier


class BiLSTMPredictor:
    """Loads BiLSTM extractor + classifier and runs inference."""

    def __init__(self, domain: str = "restaurant", device=None):
        self.device = device or get_device()
        self.domain = domain

        # Load vocab
        with open(PROCESSED_DIR / "vocab.json") as f:
            self.vocab = json.load(f)

        vocab_size = len(self.vocab)

        # Load extractor
        self.extractor = BiLSTMAspectExtractor(vocab_size=vocab_size)
        ext_path = MODELS_DIR / "aspect_extraction" / "bilstm" / f"best_{domain}.pt"
        if ext_path.exists():
            self.extractor.load_state_dict(torch.load(ext_path, map_location=self.device, weights_only=True))
        self.extractor.to(self.device).eval()

        # Load classifier
        self.classifier = BiLSTMSentimentClassifier(vocab_size=vocab_size)
        cls_path = MODELS_DIR / "sentiment_classification" / "bilstm" / f"best_{domain}.pt"
        if cls_path.exists():
            self.classifier.load_state_dict(torch.load(cls_path, map_location=self.device, weights_only=True))
        self.classifier.to(self.device).eval()

    def extract_aspects(self, text: str) -> List[Dict]:
        tokens = text.split()
        ids = [self.vocab.get(t.lower(), 1) for t in tokens]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.extractor(input_ids=input_ids, lengths=lengths)
        logits = out["logits"][0, :len(ids)]
        preds = logits.argmax(dim=-1).cpu().tolist()

        aspects = []
        current_aspect: List[str] = []
        current_start = 0
        for i, (tok, pred) in enumerate(zip(tokens, preds)):
            label = BIO_ID2LABEL[pred]
            if label == "B-ASP":
                if current_aspect:
                    aspects.append({"term": " ".join(current_aspect), "token_start": current_start})
                current_aspect = [tok]
                current_start = i
            elif label == "I-ASP" and current_aspect:
                current_aspect.append(tok)
            else:
                if current_aspect:
                    aspects.append({"term": " ".join(current_aspect), "token_start": current_start})
                    current_aspect = []
        if current_aspect:
            aspects.append({"term": " ".join(current_aspect), "token_start": current_start})

        # Find char spans
        for asp in aspects:
            idx = text.lower().find(asp["term"].lower())
            asp["char_start"] = idx
            asp["char_end"] = idx + len(asp["term"]) if idx >= 0 else -1

        return aspects

    def classify_sentiment(self, text: str, aspect: str) -> Dict:
        combined = text.lower().split() + ["[SEP]"] + aspect.lower().split()
        ids = [self.vocab.get(t, 1) for t in combined]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.classifier(input_ids=input_ids, lengths=lengths)
        probs = F.softmax(out["logits"][0], dim=-1)
        pred = probs.argmax().item()
        return {
            "sentiment": SENT_ID2LABEL[pred],
            "confidence": probs[pred].item(),
            "probabilities": {SENT_ID2LABEL[i]: probs[i].item() for i in range(len(SENTIMENT_LABELS))},
        }

    def predict(self, text: str) -> Dict:
        aspects = self.extract_aspects(text)
        results = []
        for asp in aspects:
            sent_info = self.classify_sentiment(text, asp["term"])
            results.append({
                "aspect": asp["term"],
                "sentiment": sent_info["sentiment"],
                "confidence": round(sent_info["confidence"], 4),
                "char_start": asp.get("char_start", -1),
                "char_end": asp.get("char_end", -1),
            })
        return {"review": text, "aspects": results}


class DistilBertPredictor:
    """Loads DistilBERT extractor + classifier and runs inference."""

    def __init__(self, domain: str = "restaurant", device=None):
        from transformers import DistilBertTokenizerFast
        from src.models.distilbert_extractor import DistilBertAspectExtractor
        from src.models.distilbert_classifier import DistilBertSentimentClassifier

        self.device = device or get_device()
        self.domain = domain

        tok_dir = MODELS_DIR / "aspect_extraction" / "distilbert"
        if tok_dir.exists() and (tok_dir / "vocab.txt").exists():
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(tok_dir))
        else:
            try:
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL, local_files_only=True)
            except Exception:
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL)

        self.extractor = DistilBertAspectExtractor(model_name=DISTILBERT_MODEL)
        ext_path = MODELS_DIR / "aspect_extraction" / "distilbert" / f"best_{domain}.pt"
        if ext_path.exists():
            self.extractor.load_state_dict(torch.load(ext_path, map_location=self.device, weights_only=True))
        self.extractor.to(self.device).eval()

        self.classifier = DistilBertSentimentClassifier(model_name=DISTILBERT_MODEL)
        cls_path = MODELS_DIR / "sentiment_classification" / "distilbert" / f"best_{domain}.pt"
        if cls_path.exists():
            self.classifier.load_state_dict(torch.load(cls_path, map_location=self.device, weights_only=True))
        self.classifier.to(self.device).eval()

    def extract_aspects(self, text: str) -> List[Dict]:
        tokens = text.split()
        encoding = self.tokenizer(
            tokens, is_split_into_words=True,
            return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.extractor(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"][0]
        preds = logits.argmax(dim=-1).cpu().tolist()
        word_ids = encoding.word_ids(batch_index=0)

        # Map subword preds back to word-level
        word_preds = {}
        for i, wid in enumerate(word_ids):
            if wid is not None and wid not in word_preds:
                word_preds[wid] = preds[i]

        aspects = []
        current_aspect: List[str] = []
        for wid in sorted(word_preds):
            if wid >= len(tokens):
                break
            label = BIO_ID2LABEL.get(word_preds[wid], "O")
            if label == "B-ASP":
                if current_aspect:
                    aspects.append({"term": " ".join(current_aspect)})
                current_aspect = [tokens[wid]]
            elif label == "I-ASP" and current_aspect:
                current_aspect.append(tokens[wid])
            else:
                if current_aspect:
                    aspects.append({"term": " ".join(current_aspect)})
                    current_aspect = []
        if current_aspect:
            aspects.append({"term": " ".join(current_aspect)})

        for asp in aspects:
            idx = text.lower().find(asp["term"].lower())
            asp["char_start"] = idx
            asp["char_end"] = idx + len(asp["term"]) if idx >= 0 else -1

        return aspects

    def classify_sentiment(self, text: str, aspect: str) -> Dict:
        encoding = self.tokenizer(
            text, aspect,
            return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(out["logits"][0], dim=-1)
        pred = probs.argmax().item()
        return {
            "sentiment": SENT_ID2LABEL[pred],
            "confidence": probs[pred].item(),
            "probabilities": {SENT_ID2LABEL[i]: probs[i].item() for i in range(len(SENTIMENT_LABELS))},
        }

    def predict(self, text: str) -> Dict:
        aspects = self.extract_aspects(text)
        results = []
        for asp in aspects:
            sent_info = self.classify_sentiment(text, asp["term"])
            results.append({
                "aspect": asp["term"],
                "sentiment": sent_info["sentiment"],
                "confidence": round(sent_info["confidence"], 4),
                "char_start": asp.get("char_start", -1),
                "char_end": asp.get("char_end", -1),
            })
        return {"review": text, "aspects": results}

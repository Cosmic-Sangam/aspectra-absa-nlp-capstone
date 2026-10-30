"""Preprocessing: BIO tagging, tokenization, label alignment."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

from src.utils.config import (
    PROCESSED_DIR, BIO_LABELS, BIO_LABEL2ID,
    SENT_LABEL2ID, SENTIMENT_LABELS, MAX_SEQ_LEN, setup_logging,
)

logger = setup_logging("preprocessing")


# ── BIO conversion ──────────────────────────────────────────────────
def sentence_to_bio(text: str, aspects: List[Dict]) -> Tuple[List[str], List[str]]:
    """Convert a sentence + aspect annotations to token-level BIO tags.

    Uses simple whitespace tokenization. Character offsets map each token
    to its BIO label.
    """
    tokens: List[str] = []
    bio_tags: List[str] = []

    # Build a character-level label array
    char_labels = ["O"] * len(text)
    for asp in aspects:
        start, end = asp["from"], asp["to"]
        if start < 0 or end < 0 or start >= len(text):
            continue
        char_labels[start] = "B-ASP"
        for ci in range(start + 1, min(end, len(text))):
            if char_labels[ci] == "O":
                char_labels[ci] = "I-ASP"

    # Tokenize by whitespace and assign BIO tag per token
    pos = 0
    for token in text.split():
        idx = text.find(token, pos)
        if idx == -1:
            idx = pos
        tok_end = idx + len(token)
        # Majority label for this token span
        span_labels = char_labels[idx:tok_end]
        if "B-ASP" in span_labels:
            tag = "B-ASP" if span_labels[0] == "B-ASP" else "I-ASP"
        elif "I-ASP" in span_labels:
            tag = "I-ASP"
        else:
            tag = "O"
        tokens.append(token)
        bio_tags.append(tag)
        pos = tok_end

    return tokens, bio_tags


def build_vocab(sentences: List[Dict], min_freq: int = 1) -> Dict[str, int]:
    """Build a word → index vocabulary from parsed sentences."""
    freq: Dict[str, int] = {}
    for sent in sentences:
        for tok in sent["text"].lower().split():
            freq[tok] = freq.get(tok, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in sorted(freq.items()):
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


# ── Extraction dataset ──────────────────────────────────────────────
def prepare_extraction_data(sentences: List[Dict]) -> List[Dict]:
    """Create extraction samples: each has tokens + BIO tags."""
    samples = []
    for sent in sentences:
        tokens, bio_tags = sentence_to_bio(sent["text"], sent["aspects"])
        samples.append({
            "id": sent["id"],
            "text": sent["text"],
            "tokens": tokens,
            "bio_tags": bio_tags,
            "bio_ids": [BIO_LABEL2ID[t] for t in bio_tags],
        })
    return samples


# ── Classification dataset ──────────────────────────────────────────
def prepare_classification_data(sentences: List[Dict]) -> List[Dict]:
    """Create classification samples: one per (sentence, aspect) pair."""
    samples = []
    for sent in sentences:
        for asp in sent["aspects"]:
            polarity = asp["polarity"]
            if polarity not in SENT_LABEL2ID:
                # skip 'conflict' or unknown
                if polarity == "conflict":
                    continue
                logger.debug("Unknown polarity '%s' — skipping", polarity)
                continue
            samples.append({
                "id": sent["id"],
                "text": sent["text"],
                "aspect_term": asp["term"],
                "polarity": polarity,
                "label_id": SENT_LABEL2ID[polarity],
                "aspect_start": asp["from"],
                "aspect_end": asp["to"],
            })
    return samples


# ── Subword alignment (for Transformer models) ──────────────────────
def align_labels_to_subwords(
    tokens: List[str],
    bio_tags: List[str],
    tokenizer,
    max_len: int = MAX_SEQ_LEN,
) -> Dict:
    """Tokenize with a HF tokenizer and align BIO labels to subword pieces.

    Returns dict with input_ids, attention_mask, labels (padded).
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    aligned_labels = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            aligned_labels.append(-100)  # special tokens
        elif wid != prev_word_id:
            aligned_labels.append(BIO_LABEL2ID[bio_tags[wid]] if wid < len(bio_tags) else 0)
        else:
            # subsequent sub-word of the same token → I-ASP if original is B/I, else O
            if wid < len(bio_tags) and bio_tags[wid] in ("B-ASP", "I-ASP"):
                aligned_labels.append(BIO_LABEL2ID["I-ASP"])
            else:
                aligned_labels.append(-100)
        prev_word_id = wid

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": aligned_labels,
    }


# ── Save / load helpers ─────────────────────────────────────────────
def save_processed(data: List[Dict], name: str) -> Path:
    """Save processed data as JSON."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info("Saved %d samples → %s", len(data), path.name)
    return path


def load_processed(name: str) -> List[Dict]:
    """Load processed data from JSON."""
    path = PROCESSED_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_all():
    """Run full preprocessing pipeline on all parsed datasets."""
    for split in ["laptop_train", "laptop_test", "restaurant_train", "restaurant_test"]:
        src = PROCESSED_DIR / f"{split}.json"
        if not src.exists():
            logger.warning("Parsed file %s not found — skip", src)
            continue
        with open(src, "r", encoding="utf-8") as f:
            sentences = json.load(f)

        ext_data = prepare_extraction_data(sentences)
        save_processed(ext_data, f"{split}_extraction")

        cls_data = prepare_classification_data(sentences)
        save_processed(cls_data, f"{split}_classification")

        logger.info(
            "%s → %d extraction samples, %d classification samples",
            split, len(ext_data), len(cls_data),
        )

    # Build vocab from training data
    all_train = []
    for domain in ["laptop", "restaurant"]:
        p = PROCESSED_DIR / f"{domain}_train.json"
        if p.exists():
            with open(p) as f:
                all_train.extend(json.load(f))
    vocab = build_vocab(all_train)
    vocab_path = PROCESSED_DIR / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    logger.info("Vocabulary: %d tokens", len(vocab))


if __name__ == "__main__":
    preprocess_all()

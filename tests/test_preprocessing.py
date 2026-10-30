"""Unit tests for preprocessing and data formatting."""
import pytest
from src.data.preprocessing import (
    sentence_to_bio,
    build_vocab,
    prepare_extraction_data,
    prepare_classification_data,
)
from src.utils.config import BIO_LABEL2ID, SENT_LABEL2ID


def test_sentence_to_bio_single_aspect():
    text = "The screen is amazing."
    aspects = [{"term": "screen", "polarity": "positive", "from": 4, "to": 10}]
    tokens, bio_tags = sentence_to_bio(text, aspects)
    assert "screen" in tokens
    idx = tokens.index("screen")
    assert bio_tags[idx] == "B-ASP"
    assert bio_tags[0] == "O"


def test_sentence_to_bio_multi_word_aspect():
    text = "Great battery life indeed."
    aspects = [{"term": "battery life", "polarity": "positive", "from": 6, "to": 18}]
    tokens, bio_tags = sentence_to_bio(text, aspects)
    assert "battery" in tokens
    b_idx = tokens.index("battery")
    assert bio_tags[b_idx] == "B-ASP"
    assert bio_tags[b_idx + 1] == "I-ASP"


def test_sentence_to_bio_no_aspect():
    text = "Just a general sentence without aspects."
    aspects = []
    tokens, bio_tags = sentence_to_bio(text, aspects)
    assert all(tag == "O" for tag in bio_tags)


def test_build_vocab():
    sentences = [
        {"text": "Fast performance and great battery"},
        {"text": "Poor battery life"},
    ]
    vocab = build_vocab(sentences, min_freq=1)
    assert "<PAD>" in vocab
    assert "<UNK>" in vocab
    assert "battery" in vocab
    assert vocab["<PAD>"] == 0
    assert vocab["<UNK>"] == 1


def test_prepare_classification_data():
    sentences = [
        {
            "id": "1",
            "text": "The battery is good but screen is bad.",
            "aspects": [
                {"term": "battery", "polarity": "positive", "from": 4, "to": 11},
                {"term": "screen", "polarity": "negative", "from": 24, "to": 30},
            ],
        }
    ]
    samples = prepare_classification_data(sentences)
    assert len(samples) == 2
    assert samples[0]["polarity"] == "positive"
    assert samples[0]["label_id"] == SENT_LABEL2ID["positive"]
    assert samples[1]["polarity"] == "negative"
    assert samples[1]["label_id"] == SENT_LABEL2ID["negative"]

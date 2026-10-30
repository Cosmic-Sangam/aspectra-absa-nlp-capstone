"""PyTorch Dataset classes for ABSA extraction and classification."""
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset

from src.utils.config import PROCESSED_DIR, BIO_LABEL2ID, MAX_SEQ_LEN


class ExtractionDataset(Dataset):
    """Token-classification dataset for aspect extraction (BiLSTM)."""

    def __init__(self, data: List[Dict], vocab: Dict[str, int], max_len: int = MAX_SEQ_LEN):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample["tokens"][:self.max_len]
        bio_ids = sample["bio_ids"][:self.max_len]

        # Numericalize
        input_ids = [self.vocab.get(t.lower(), 1) for t in tokens]  # 1 = <UNK>
        length = len(input_ids)

        # Pad
        pad_len = self.max_len - length
        input_ids += [0] * pad_len
        bio_ids += [0] * pad_len  # O = 0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(bio_ids, dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
        }


class ClassificationDataset(Dataset):
    """Sentence-aspect pair dataset for sentiment classification (BiLSTM)."""

    def __init__(self, data: List[Dict], vocab: Dict[str, int], max_len: int = MAX_SEQ_LEN):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        aspect = sample["aspect_term"]

        # Combine: [CLS-like] text [SEP-like] aspect
        tokens = text.lower().split() + ["[SEP]"] + aspect.lower().split()
        tokens = tokens[:self.max_len]
        input_ids = [self.vocab.get(t, 1) for t in tokens]
        length = len(input_ids)

        pad_len = self.max_len - length
        input_ids += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(sample["label_id"], dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
        }


class TransformerExtractionDataset(Dataset):
    """Token-classification dataset for DistilBERT aspect extraction."""

    def __init__(self, data: List[Dict], tokenizer, max_len: int = MAX_SEQ_LEN):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample["tokens"]
        bio_tags = sample["bio_tags"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                aligned_labels.append(
                    BIO_LABEL2ID[bio_tags[wid]] if wid < len(bio_tags) else 0
                )
            else:
                if wid < len(bio_tags) and bio_tags[wid] in ("B-ASP", "I-ASP"):
                    aligned_labels.append(BIO_LABEL2ID["I-ASP"])
                else:
                    aligned_labels.append(-100)
            prev_word_id = wid

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


class TransformerClassificationDataset(Dataset):
    """Sentence-aspect pair dataset for DistilBERT sentiment classification."""

    def __init__(self, data: List[Dict], tokenizer, max_len: int = MAX_SEQ_LEN):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        aspect = sample["aspect_term"]

        encoding = self.tokenizer(
            text,
            aspect,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label_id"], dtype=torch.long),
        }

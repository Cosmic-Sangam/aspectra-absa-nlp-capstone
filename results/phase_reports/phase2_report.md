# Phase 2 Report: Model Architecture & Training

## Executive Summary
In Phase 2, both BiLSTM and DistilBERT architectures were constructed and trained across both domains (`restaurant` and `laptop`) for two subtasks:
1. **Aspect Term Extraction** (Sequence tagging using BIO tags)
2. **Aspect Sentiment Classification** (Contextual polarity classification: positive, negative, neutral)

All training was executed on GPU (`NVIDIA GeForce RTX 2050 4GB`).

---

## 1. Model Architectures

### BiLSTM Baseline
- **Embedding:** 128-dimensional learned embeddings initialized from vocabulary (5,856 tokens).
- **Encoder:** 2-layer Bidirectional LSTM, 128 hidden units per direction (256 total), dropout=0.3.
- **Extractor Head:** Time-distributed Linear classifier predicting BIO tag distributions (`[batch, seq_len, 3]`).
- **Classifier Head:** Concatenated sentence + `[SEP]` + aspect representation with Attention/Max-pooling into 3 classes.

### DistilBERT (Transformer)
- **Base Model:** `distilbert-base-uncased` (6 layers, 768 hidden, 12 attention heads, 66M parameters).
- **Extractor:** Subword sequence output with dropout (0.1) and Linear projection to BIO tags, ignoring padding (`-100`).
- **Classifier:** `[CLS]` pooled token representation passed through hidden dense layer + ReLU + classification head.

---

## 2. Training Configurations & Hyperparameters

- **Optimizer:** Adam (BiLSTM lr=1e-3, DistilBERT lr=2e-5 with AdamW)
- **Batch Size:** 32 (BiLSTM), 16 (DistilBERT)
- **Epochs:** 5 (BiLSTM), 3 (DistilBERT)
- **Device:** GPU (CUDA `cuda:0`)
- **Loss:** Masked CrossEntropyLoss for extraction, CrossEntropyLoss for classification

---

## 3. Training & Validation Performance

### Restaurant Domain
| Architecture | Task | Best Val Loss | Best Val Accuracy / F1 |
|---|---|---|---|
| BiLSTM Extractor | Aspect Extraction | 0.1787 | Loss: 0.1787 |
| BiLSTM Classifier | Sentiment | 0.8007 | 69.29% |
| DistilBERT Extractor | Aspect Extraction | 0.1072 | Loss: 0.1072 |
| DistilBERT Classifier | Sentiment | 0.4311 | **84.29%** |

### Laptop Domain
| Architecture | Task | Best Val Loss | Best Val Accuracy / F1 |
|---|---|---|---|
| BiLSTM Extractor | Aspect Extraction | 0.2086 | Loss: 0.2086 |
| BiLSTM Classifier | Sentiment | 0.8859 | 60.66% |
| DistilBERT Extractor | Aspect Extraction | 0.1403 | Loss: 0.1403 |
| DistilBERT Classifier | Sentiment | 0.6026 | **75.71%** |

Checkpoints saved under `models/aspect_extraction/` and `models/sentiment_classification/`.

# ABSA Dataset Documentation

This directory contains the datasets for **Aspect-Based Sentiment Analysis (ABSA)** based on the standard **SemEval-2014 Task 4** benchmark.

---

## 1. Dataset Origin & Description

- **Source:** SemEval-2014 Task 4 (Aspect-Based Sentiment Analysis)
- **Domains:**
  1. `restaurant`: Customer dining reviews covering food quality, service, ambiance, price.
  2. `laptop`: Customer hardware reviews covering battery life, screen, performance, price.
- **Format:**
  - Raw format: `.seg` triplets (Sentence text with `$T$` aspect marker, aspect term, polarity index).
  - Validated with 100% offset accuracy (`7694/7694` gold offsets verified).

---

## 2. Dataset Splits and Statistics

| Split | Sentences | Aspect Terms | Avg Aspects / Sent |
|---|---|---|---|
| `restaurant_train` | 2,009 | 3,608 | 1.80 |
| `restaurant_test` | 610 | 1,120 | 1.84 |
| `laptop_train` | 1,482 | 2,328 | 1.57 |
| `laptop_test` | 418 | 638 | 1.53 |
| **Total** | **4,519** | **7,694** | **1.70** |

---

## 3. Processed Files Structure (`data/processed/`)

- `*_extraction.json`: Samples with word-level tokens and BIO tags (`B-ASP`, `I-ASP`, `O`).
- `*_classification.json`: Sentence + aspect pairs with categorical polarity labels (`positive`, `negative`, `neutral`).
- `vocab.json`: Built word vocabulary (5,856 tokens) with `<PAD>` (index 0) and `<UNK>` (index 1).
- `dataset_stats.json`: Complete serialized frequency counts and token distributions.

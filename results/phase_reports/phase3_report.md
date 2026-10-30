# Phase 3 Report: Evaluation, Explainability & Error Analysis

## Executive Summary
Phase 3 completed the evaluation suite across all 4 trained models on the official SemEval-2014 test sets, implemented leave-one-out perturbation explainability, and produced confusion matrices and error analyses.

---

## 1. Test Set Evaluation Results

### Restaurant Domain
| Metric | BiLSTM Baseline | DistilBERT Transformer | Absolute Gain |
|---|---|---|---|
| **Extraction Token F1** | 76.83% | **85.54%** | +8.71% |
| **Extraction Exact F1** | 77.62% | **86.07%** | +8.45% |
| **Sentiment Accuracy** | 68.48% | **84.29%** | +15.81% |
| **Sentiment Macro-F1** | 47.71% | **76.98%** | +29.27% |

### Laptop Domain
| Metric | BiLSTM Baseline | DistilBERT Transformer | Absolute Gain |
|---|---|---|---|
| **Extraction Token F1** | 70.80% | **82.95%** | +12.15% |
| **Extraction Exact F1** | 67.00% | **81.29%** | +14.29% |
| **Sentiment Accuracy** | 60.66% | **75.71%** | +15.05% |
| **Sentiment Macro-F1** | 49.68% | **71.56%** | +21.88% |

---

## 2. Explainability Analysis

The pipeline implements leave-one-out word perturbation:
$$\Delta \text{Confidence} = \text{Conf}_{\text{baseline}}(s, a) - \text{Conf}_{\text{perturbed}}(s \setminus \{w_i\}, a)$$
- Words with positive drops support the predicted polarity (e.g., "crispy", "delicious", "exceptional").
- Words with negative drops oppose the polarity or provide contrasting nuance (e.g., "slow", "cheap").

---

## 3. Key Error Patterns

- **Multi-word aspects:** BiLSTM occasionally splits multi-word aspect spans into separate terms or predicts only head nouns. DistilBERT preserves subword phrase boundaries significantly better.
- **Neutral Polarity Confusion:** Both models experience the highest error rate on neutral reviews (factual statements without strong sentiment adjectives), often misclassifying them as positive or negative.
- **Negation and Contrast:** Complex conjunction structures ("The food was fine, but neither the manager nor the staff apologized") pose challenges for BiLSTM, whereas DistilBERT captures cross-clause self-attention.

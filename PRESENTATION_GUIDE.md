# Project Presentation & PowerPoint Guide (PPT Reference)

This document contains the complete slide-by-slide content, architecture diagrams, mathematical formulations, and presentation talking points for presenting this **End-to-End Aspect-Based Sentiment Analysis (ABSA)** system to your professor or evaluation committee.

---

## Slide 1: Title Slide
- **Title:** End-to-End Aspect-Based Sentiment Analysis (ABSA) for E-Commerce & Customer Reviews
- **Subtitle:** Comparative Study of Bidirectional LSTM Baselines vs. Fine-Tuned DistilBERT Transformers with Explainable NLP
- **Course / Topic:** Natural Language Processing (Semester 5)
- **Presenter:** [Your Name / Roll Number]
- **Key Deliverables:** Multi-Model Pipeline, REST API, Stripe-Style Dashboard, Explainability Engine

> **Presenter Talking Point:**  
> *"Good morning professor. Today I am presenting our End-to-End Aspect-Based Sentiment Analysis system. Traditional sentiment analysis only tells you if an entire review is positive or negative. Our system breaks down complex sentences to extract individual aspects—like food quality, service, battery life, or screen display—and classifies the sentiment for each specific aspect individually with explainable AI."*

---

## Slide 2: Problem Statement & Motivation
- **Document-Level Limitation:** A review saying *"The pizza was phenomenal, but the staff was rude"* is classified as Neutral by conventional sentiment analyzers, losing actionable feedback.
- **ABSA Objective:**
  1. **Subtask 1 (Aspect Term Extraction - ATE):** Identify token boundaries of target entities (e.g. `pizza`, `staff`).
  2. **Subtask 2 (Aspect-Level Sentiment Classification - ALSC):** Determine the polarity (`Positive`, `Negative`, `Neutral`) conditioned on the target entity.
- **Industry Applications:** Amazon product review aggregation, restaurant Yelp intelligence, customer support ticketing, automated competitive intelligence.

---

## Slide 3: Dataset & Verification (SemEval-2014 Task 4)
- **Official Benchmark:** SemEval-2014 Task 4 (Subtask 1 & Subtask 2).
- **Domains Evaluated:**
  - **Restaurant Domain:** 2,009 train sentences (3,608 aspects), 610 test sentences (1,120 aspects).
  - **Laptop Domain:** 1,482 train sentences (2,328 aspects), 418 test sentences (638 aspects).
- **Annotation & Offsets:**
  - Verified 100% character offsets (`7,694 / 7,694` aspects verified).
  - Sequence tagging encoded via **BIO Scheme** (`B-ASP` for beginning, `I-ASP` for inside, `O` for outside).
  - Custom vocabulary built: 5,856 unique tokens with `<PAD>` and `<UNK>` tokens.

---

## Slide 4: System Architecture & Pipeline Workflow

```
+-------------------------------------------------------------------------+
|                           Raw Input Review                              |
|   "The pizza was delicious and crispy, but the service was terrible."   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|                Subtask 1: Aspect Term Extraction (BIO)                 |
|   Tokens:  ["The", "pizza", "was", "crispy", "...", "service", "..."]   |
|   BIO:     [  O  ,  B-ASP ,   O  ,    O    ,  ... ,  B-ASP   , ... ]   |
|   Spans Extracted: [ "pizza", "service" ]                               |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|            Subtask 2: Aspect-Level Sentiment Classification            |
|   Input Pair 1: ("Review Text", "pizza")   --> Output: Positive (96%)   |
|   Input Pair 2: ("Review Text", "service") --> Output: Negative (94%)   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|               Subtask 3: Token Explainability Engine                   |
|   Leave-One-Out Perturbation: measures confidence drop per token        |
|   Supporting tokens: "delicious" (+0.45), "crispy" (+0.31)              |
+------------------------------------+------------------------------------+
```

---

## Slide 5: Model Architectures Compared

### Model A: BiLSTM Baseline
- **Word Embeddings:** 128-dimensional learned embeddings over a 5,856-word vocabulary.
- **Encoder:** 2-layer Bidirectional LSTM (128 units/direction = 256 hidden dimension), Dropout = 0.3.
- **Extractor Head:** Time-distributed Linear layer projecting hidden states to 3 BIO logits.
- **Classifier Head:** Input formatted as `[Sentence] [SEP] [Aspect]`, encoded by BiLSTM with Max-pooling across time, followed by Linear layer to 3 sentiment classes.

### Model B: DistilBERT Transformer (Fine-Tuned)
- **Architecture:** Pre-trained `distilbert-base-uncased` (6 layers, 768 hidden dimension, 12 attention heads, 66M parameters).
- **Subword Handling:** WordPiece tokenizer with Fast Tokenizer offsets mapping subword logits back to word-level spans.
- **Extractor Head:** Sequence output with Dropout (0.1) and Linear classifier, ignoring subword padding tokens (`ignore_index = -100`).
- **Classifier Head:** Aspect and sentence concatenated via `[CLS] Sentence [SEP] Aspect [SEP]`. The pooled `[CLS]` embedding is passed through a Linear layer + ReLU + Dropout + 3-way classification head.

---

## Slide 6: Mathematical Formulations

### 1. BIO Extraction Loss (Masked Cross-Entropy)
$$\mathcal{L}_{\text{ext}} = -\frac{1}{\sum_{i=1}^{N} m_i} \sum_{i=1}^{N} \sum_{c \in \{B, I, O\}} y_{i,c} \log \hat{y}_{i,c}$$
*(where $m_i \in \{0, 1\}$ masks out padding tokens).*

### 2. Aspect-Conditioned Sentiment Classification
$$\mathbf{h}_{[\text{CLS}]} = \text{DistilBERT}([\text{CLS}] \oplus \mathbf{x} \oplus [\text{SEP}] \oplus \mathbf{a} \oplus [\text{SEP}])$$
$$\mathbf{z} = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{h}_{[\text{CLS}]} + \mathbf{b}_1) + \mathbf{b}_2$$
$$\hat{\mathbf{y}}_{\text{sent}} = \text{softmax}(\mathbf{z})$$

### 3. Leave-One-Out Token Attribution (Explainability)
$$I(w_k; a, s) = P(s \mid \mathbf{x}, a) - P(s \mid \mathbf{x} \setminus \{w_k\}, a)$$
- $I > 0$: Token $w_k$ provides positive evidence supporting sentiment $s$.
- $I < 0$: Token $w_k$ opposes sentiment $s$.

---

## Slide 7: Empirical Results & Benchmark Comparison

All results evaluated on the official unseen SemEval-2014 test sets:

| Domain | Architecture | Aspect Exact F1 | Aspect Token F1 | Sentiment Accuracy | Sentiment Macro-F1 |
|---|---|---|---|---|---|
| **Restaurant** | **DistilBERT** | **86.07%** | **85.54%** | **84.29%** | **76.98%** |
| Restaurant | BiLSTM | 77.62% | 76.83% | 68.48% | 47.71% |
| *Margin* | *Transformer* | *+8.45%* | *+8.71%* | *+15.81%* | *+29.27%* |
| **Laptop** | **DistilBERT** | **81.29%** | **82.95%** | **75.71%** | **71.56%** |
| Laptop | BiLSTM | 67.00% | 70.80% | 60.66% | 49.68% |
| *Margin* | *Transformer* | *+14.29%* | *+12.15%* | *+15.05%* | *+21.88%* |

---

## Slide 8: Key Insights & Error Analysis

1. **Self-Attention Advantage:** DistilBERT successfully connects sentiment adjectives across long distances to the correct aspect noun, even across contrastive conjunctions (*"while"*, *"although"*, *"but"*).
2. **Multi-Word Aspect Spans:** BiLSTM tends to split phrases (e.g. predicting *"battery"* instead of *"battery life"*). DistilBERT's exact match F1 is **14.29% higher** on the laptop domain.
3. **Neutral Polarity Challenge:** Both models encounter the highest misclassification rate on neutral reviews (factual statements without explicit polarity markers).
4. **Latency vs. Accuracy Trade-Off:** BiLSTM runs in ~15ms on CPU, making it ideal for low-power edge devices, while DistilBERT delivers superior accuracy in ~60ms on GPU.

---

## Slide 9: System Demo & UI Showcase
- **Backend:** FastAPI service with OpenAPI 3.0 specs, Pydantic validation, CORS middleware, and GPU acceleration.
- **Frontend:** Inspired by Stripe's developer-grade UI design:
  - Responsive cards with subtle borders and shadows.
  - Aspect chips with color codes (Emerald for Positive, Rose for Negative, Slate for Neutral).
  - Real-time confidence progress meters.
  - Interactive token-level attribution badges.
  - Single review and batch CSV/JSON export capabilities.

---

## Slide 10: Conclusion & Future Extensions
- **Conclusion:** Successfully built and benchmarked an end-to-end ABSA system showing significant transformer improvements (+15.8% accuracy, +29.2% macro-F1) over bidirectional RNN baselines.
- **Future Directions:**
  1. Implicit Aspect Detection (inferring aspects like *price* from *"costs an arm and a leg"*).
  2. Generative Instruction Tuning (LLM-based joint tuple generation using Llama 3 / Mistral).
  3. Cross-lingual zero-shot aspect sentiment transfer.

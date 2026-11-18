# ACADEMIC PROJECT REPORT

## ASPECTRA: DECOUPLED TWO-STAGE ASPECT-BASED SENTIMENT ANALYSIS WITH ATTENTION AND TOKEN PERTURBATION EXPLAINABILITY

---

### METADATA
- **Course**: Natural Language Processing (Semester 5)
- **Domain**: Deep Learning, Transformers, Opinion Mining, Information Extraction
- **Dataset**: SemEval-2014 Task 4 (Subtask 1 & 2)
- **Architectures**: DistilBERT (`distilbert-base-uncased`), Bidirectional LSTM (BiLSTM)
- **Evaluation Benchmark**: Exact Match Span F1, Token F1, Sentiment Accuracy, Macro-F1

---

## 1. ABSTRACT & EXECUTIVE SUMMARY

Traditional sentiment analysis categorizes an entire document or sentence into a single polarity: positive, negative, or neutral. While effective for simple, homogeneous statements, real-world customer opinions (e.g., product feedback, Yelp reviews) are fundamentally multi-faceted. A review frequently praises one feature while criticizing another.

In this work, we present **Aspectra**, an end-to-end NLP system engineered for fine-grained **Aspect-Based Sentiment Analysis (ABSA)**. Aspectra implements a decoupled two-stage sequential architecture:
1. **Aspect Term Extraction (ATE)**: Identifying target opinion entities via token-level BIO sequence tagging (`B-ASP`, `I-ASP`, `O`).
2. **Aspect-Level Sentiment Classification (ALSC)**: Predicting three-class polarity conditioned on each target aspect through self-attention representations.

To benchmark effectiveness, we trained and evaluated two distinct model paradigms on the human-annotated **SemEval-2014 Task 4** dataset:
- A fine-tuned **DistilBERT** Transformer architecture (66M parameters).
- A custom 2-layer **Bidirectional LSTM (BiLSTM)** sequence model with dense embeddings.

Empirical results reveal that DistilBERT achieves **86.07% exact span F1** on restaurant reviews and **81.29% exact span F1** on laptop reviews, substantially outperforming the BiLSTM baseline (77.62% and 67.00% respectively). Furthermore, to remove "black box" deep learning opacity, we introduce an algorithmic **Leave-One-Out (LOO)** perturbation attribution engine that measures exact token-level influence for every prediction. The system is deployed via dual interfaces: an interactive **Streamlit Studio** and a **FastAPI + TypeScript** web application.

---

## 2. PROBLEM STATEMENT & MOTIVATION

### 2.1 The Failure Mode of Monolithic Sentiment Analysis
Consider the following authentic customer review:
> *"The margherita pizza was delicious and crispy, but the waiter was extremely rude and the bill was overpriced."*

A traditional lexicon-based (e.g., VADER, TextBlob) or document-level classification model aggregates these conflicting clauses into an arbitrary average score (typically neutral or weakly positive due to lexical weighting). 

This introduces three critical operational failures:
1. **Information Loss**: Business managers cannot pinpoint whether customer dissatisfaction stems from food quality, customer service, or pricing.
2. **False Generalization**: Strong negative experiences on secondary aspects are masked by praise on primary aspects.
3. **Lack of Actionability**: Automated customer feedback aggregation tools cannot build structured feature-level rating dashboards.

### 2.2 Objectives of Aspectra
- Formulate ABSA as a modular two-stage pipeline separating entity span detection from polarity assignment.
- Compare pre-trained Transformer distillation against recurrent architectures under identical tokenization and evaluation constraints.
- Implement token-level interpretability (XAI) to explain predictions transparently.
- Provide a responsive, production-ready interface with dynamic confidence thresholding.

---

## 3. DATASET & CORPUS ANALYSIS

We utilize the gold-standard **SemEval-2014 Task 4** benchmark, consisting of authentic customer reviews from two distinct domains:

### 3.1 Corpus Statistics
| Domain | Training Sentences | Test Sentences | Total Aspects | Aspect Density |
| :--- | :---: | :---: | :---: | :---: |
| **Restaurant Reviews** | 2,009 | 610 | 4,728 | ~1.81 aspects/sentence |
| **Laptop Reviews** | 1,482 | 418 | 2,966 | ~1.56 aspects/sentence |
| **Combined** | **3,491** | **1,028** | **7,694** | **~1.70 aspects/sentence** |

### 3.2 Polarity Distribution
The dataset exhibits natural class imbalance, reflecting genuine user review distributions:
- **Positive**: ~58.3%
- **Negative**: ~22.8%
- **Neutral**: ~18.9%
- **Conflict**: (<1.5% — excluded during preprocessing due to negligible frequency and ambiguous multi-polarity labels).

### 3.3 Span Annotation & Character Offsets
Every aspect term is annotated with exact character-level start and end pointers:
```json
{
  "id": "1004",
  "text": "The bread is top notch as well.",
  "aspects": [
    {
      "term": "bread",
      "polarity": "positive",
      "from": 4,
      "to": 9
    }
  ]
}
```
All character offsets were programmatically audited and verified with zero index alignment errors.

---

## 4. NLP PREPROCESSING & REPRESENTATION PIPELINE

### 4.1 BIO Sequence Tagging (Stage 1 Formulation)
To enable token classification, character offsets are transformed into a token-level BIO schema:
- **`B-ASP` (Begin Aspect)**: Assigned to the initial token of an aspect phrase.
- **`I-ASP` (Inside Aspect)**: Assigned to any subsequent tokens belonging to a multi-word aspect span (e.g., `battery` [`B-ASP`] `life` [`I-ASP`]).
- **`O` (Outside)**: Assigned to all non-aspect tokens.

### 4.2 Subword Alignment for Transformers
DistilBERT utilizes a WordPiece tokenizer (30,522 vocabulary size). A single word may split into multiple subword pieces (e.g., `"unlimited"` $\to$ `["un", "##limited"]`).
To prevent loss of label fidelity:
1. The first subword inherits the word's ground-truth label (`B-ASP` or `I-ASP`).
2. Subsequent subwords are mapped to `I-ASP` if the parent token is an aspect, or ignored via label index `-100`.
3. Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) are assigned label `-100`, instructing PyTorch `nn.CrossEntropyLoss(ignore_index=-100)` to ignore them during backpropagation.

---

## 5. SYSTEM ARCHITECTURE & MODEL METHODOLOGY

### 5.1 Architecture Pipeline Diagram
```
[Raw User Review]
       │
       ▼
[Stage 1: ATE (Aspect Term Extraction)]
       ├── DistilBERT Token Classifier (BIO tagging)
       └── Regex Colloquial Fallback Pattern Engine
       │
       ▼
[Confidence Threshold Filtering (Slider: 0.0 - 1.0)]
       │
       ▼
[Stage 2: ALSC (Aspect-Level Sentiment)]
       ├── Format: [CLS] Sentence [SEP] Aspect [SEP]
       └── Softmax 3-Class Probabilities (Pos / Neg / Neu)
       │
       ▼
[Explainable AI Engine (Leave-One-Out Token Perturbation)]
       │
       ▼
[Delivery: Streamlit Studio & FastAPI REST Backend]
```

### 5.2 Model Architectures

#### 1. DistilBERT (`distilbert-base-uncased`)
- **Parameters**: ~66 Million (40% smaller and 60% faster than BERT-base while retaining 97% of its language understanding).
- **Layers**: 6 Transformer encoder blocks, 768 hidden dimension, 12 attention heads.
- **Self-Attention Formulation**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Stage 1 Head**: Linear layer projecting `(batch, 128, 768)` $\to$ `(batch, 128, 3)`.
- **Stage 2 Head**: `[CLS]` token extracted from final hidden state $\to$ `Linear(768, 768)` $\to$ `ReLU()` $\to$ `Dropout(0.1)` $\to$ `Linear(768, 3)`.

#### 2. Bidirectional LSTM (BiLSTM Baseline)
- **Vocabulary**: 5,856 tokens constructed from training corpus.
- **Embedding Layer**: 300-dimensional dense vectors trained end-to-end.
- **Recurrent Core**: 2-layer BiLSTM, hidden dimension 256 per direction (total 512).
- **Packed Sequences**: Implemented using `torch.nn.utils.rnn.pack_padded_sequence` to bypass computational overhead on padding tokens.

### 5.3 Training Parameters
- **Optimizer**: AdamW with decoupled weight decay ($\lambda = 0.01$).
- **Learning Rate**: $2 \times 10^{-5}$ for DistilBERT, $1 \times 10^{-3}$ for BiLSTM.
- **Gradient Clipping**: Norm clipped at $1.0$ to prevent exploding gradients.
- **Early Stopping**: Monitored validation loss with patience of 3 epochs.
- **Hardware Acceleration**: NVIDIA GeForce RTX 2050 GPU with CUDA.

---

## 6. EXPERIMENTAL RESULTS & EVALUATION

### 6.1 Benchmark Results Table
Rigorous evaluation on unseen test sets yielded the following verified metrics:

| Domain | Model Architecture | Exact Span F1 | Token-Level F1 | Sentiment Accuracy | Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Restaurant** | **DistilBERT** | **86.07%** | **85.54%** | **84.29%** | **76.98%** |
| Restaurant | BiLSTM | 77.62% | 76.83% | 68.48% | 47.71% |
| **Laptop** | **DistilBERT** | **81.29%** | **82.95%** | **75.71%** | **71.56%** |
| Laptop | BiLSTM | 67.00% | 70.80% | 60.66% | 49.68% |

### 6.2 Analytical Observations
1. **Pre-training Advantage**: DistilBERT outclasses BiLSTM across both domains, primarily because pre-trained Transformer weights encode syntactic dependencies and semantic nuances that recurrent models cannot learn from 3,500 domain samples alone.
2. **Domain Disparity**: The Laptop domain consistently yields lower performance than the Restaurant domain. Laptop reviews contain technical abbreviations (e.g., *SSD*, *OLED*, *RAM*, *GUI*) and colloquial functional complaints (*"heats up after 20 minutes"*), representing implicit aspect references.
3. **Macro-F1 Sensitivity**: Macro-F1 weights each class equally regardless of frequency. Because the neutral class represents only ~19% of the corpus, subtle neutral statements are more prone to confusion, which penalizes the macro score.

---

## 7. EXPLAINABLE AI (XAI): LEAVE-ONE-OUT PERTURBATION

To foster user trust and facilitate clinical error analysis, Aspectra features a **Leave-One-Out (LOO)** perturbation algorithm:

### 7.1 Algorithmic Formulation
For a review $x = \{w_1, w_2, \dots, w_n\}$, an aspect $a$, and baseline predicted class $s$:
$$I(w_k) = P(s \mid x, a) - P(s \mid x \setminus \{w_k\}, a)$$

- **Positive Attribution ($I > 0$)**: Removing $w_k$ causes predicted confidence to collapse. This proves that $w_k$ serves as primary supporting evidence for the sentiment.
- **Negative Attribution ($I < 0$)**: Removing $w_k$ boosts the predicted confidence. This indicates that $w_k$ acted as counter-evidence or a dampening modifier.

```
Example Inference:
"The battery life is exceptional and the display is bright"
Target Aspect: "battery life" -> Sentiment: POSITIVE (Confidence: 96.1%)
Top Influential Tokens:
  1. exceptional  (+64.3% importance drop)  [Major Driver]
  2. bright       (+5.2% importance drop)   [Context Spillover]
  3. and          (-0.8% importance drop)   [Connective Modifier]
```

---

## 8. DEPLOYMENT & SYSTEM INTERFACES

Aspectra is architected for deployment flexibility across production environments:

### 8.1 Streamlit Studio (`streamlit_app.py`)
- Single-command execution: `streamlit run streamlit_app.py`.
- Integrates sidebar hyperparameter tuning (domain switching, architecture toggling, confidence thresholding).
- Implements interactive batch inference with downloadable CSV export.

### 8.2 FastAPI REST Service (`api/main.py`)
- Asynchronous REST endpoints:
  - `POST /predict`: Single review analysis with optional XAI payloads.
  - `POST /predict/batch`: High-throughput array processing.
  - `GET /health`: Model status and CUDA device telemetry.
- Auto-generated Swagger documentation at `/docs`.

---

## 9. VISUAL INTERFACE & SCREENSHOTS

*(Insert your dashboard screenshots here)*

### 9.1 Single Review Studio
<!-- Place Screenshot 1 Here -->
> **Figure 9.1**: Aspectra Single Review Studio showing aspect chips, confidence progress indicators, and predicted polarities.

### 9.2 Leave-One-Out Token Explainability
<!-- Place Screenshot 2 Here -->
> **Figure 9.2**: Token attribution highlighting supporting and opposing tokens with importance percentages.

### 9.3 Batch Review Analytics
<!-- Place Screenshot 3 Here -->
> **Figure 9.3**: Batch processing dashboard processing multiple reviews with tabulated aspect extraction.

### 9.4 Benchmark Comparison Matrix
<!-- Place Screenshot 4 Here -->
> **Figure 9.4**: Comparative performance table on SemEval-2014 test sets.

---

## 10. LIMITATIONS & FUTURE WORK

### 10.1 Identified Limitations
1. **Pipeline Error Cascading**: Because extraction precedes classification, errors in Stage 1 prevent Stage 2 from evaluating the true aspect.
2. **Computational Overhead of XAI**: Leave-One-Out perturbation requires $n$ sequential model forward passes (where $n$ is sentence length), increasing latency on long paragraphs.
3. **Monolingual Constraints**: Models are currently tuned solely on English corpora.

### 10.2 Future Extensions
1. **Joint Multi-Task Learning**: Implementing unified sequence tagging and classification using span-based pointers.
2. **Conditional Random Fields (CRF)**: Integrating a CRF layer atop the extraction head to enforce valid label transitions ($I\text{-ASP}$ cannot follow $O$).
3. **Multilingual Adaptation**: Fine-tuning `xlm-roberta-base` for cross-lingual ABSA across European and Asian languages.

---

## 11. CONCLUSION

In this project, we designed, implemented, and audited **Aspectra**, a production-oriented ABSA framework. By evaluating DistilBERT and BiLSTM on SemEval-2014, we empirically confirmed the significant superiority of pre-trained Transformers for fine-grained opinion mining (**86.07% vs. 77.62% F1**). The incorporation of Leave-One-Out explainability, dynamic confidence filtering, and dual web deployment modes elevates Aspectra from an academic exercise into a robust, interpretable decision-support platform.

---
*End of Academic Report.*

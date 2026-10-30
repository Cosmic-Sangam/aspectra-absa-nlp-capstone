# ABSA Project Development Log

## Phase 1: Environment, Architecture, Dataset Acquisition & Preprocessing
- **Date**: 2026-10-30
- **Environment**:
  - OS: Windows 11 (10.0.26300)
  - Python: 3.12.7
  - CPU: Intel64 16 logical cores
  - GPU: NVIDIA GeForce RTX 2050 (4096 MiB VRAM, 3965 MiB free)
  - Free Disk: ~103 GB
- **Virtual Environment**: Created at `.venv/` with Python 3.12.7.
- **Dependencies Installed**:
  - PyTorch 2.5.1 + CUDA 12.1 (`torch`, `torchvision`)
  - Transformers 5.22.0, scikit-learn 1.9.1, pandas 3.1.0, lxml 6.1.3
  - FastAPI 0.162.0, uvicorn, pydantic, httpx
  - Streamlit 1.66.1, matplotlib, seaborn, plotly, pytest
- **Dataset Acquisition**:
  - Downloaded SemEval-2014 Task 4 (Laptops and Restaurants, train + test gold) from verified academic repository (`songyouwei/ABSA-PyTorch`).
  - Total sentences: 4,519
  - Total aspect annotations: 7,694
  - Character offset verification: 100% exact match (0 mismatches, 0 missing).
  - Saved dataset manifest: `data/dataset_manifest.json`
- **Data Preprocessing**:
  - Implemented BIO tagging for token extraction.
  - Implemented sentence-aspect pair extraction for 3-class sentiment classification (positive, neutral, negative).
  - Generated subword token alignment with DistilBERT tokenizer.
  - Built vocabulary: 5,856 tokens.
- **Verification**:
  - `python -m compileall src`: 100% clean compilation.
  - `pytest -v`: 13/13 tests passed in 0.44s.

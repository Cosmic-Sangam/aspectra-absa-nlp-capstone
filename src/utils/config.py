"""Project-wide configuration."""
from pathlib import Path
import json
import logging

# Project root — resolves to the 'absa' folder regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Key directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset domains
DOMAINS = ["laptop", "restaurant"]

# Label maps
BIO_LABELS = ["O", "B-ASP", "I-ASP"]
BIO_LABEL2ID = {l: i for i, l in enumerate(BIO_LABELS)}
BIO_ID2LABEL = {i: l for i, l in enumerate(BIO_LABELS)}

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENT_LABEL2ID = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
SENT_ID2LABEL = {i: l for i, l in enumerate(SENTIMENT_LABELS)}

# Training defaults
DEFAULT_SEED = 42
DEFAULT_LR = 2e-5
DEFAULT_TRANSFORMER_LR = 2e-5
DEFAULT_BILSTM_LR = 1e-3
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
MAX_SEQ_LEN = 128
BILSTM_HIDDEN = 256
BILSTM_EMBED = 300
BILSTM_DROPOUT = 0.5

# Model names
DISTILBERT_MODEL = "distilbert-base-uncased"


def get_model_config() -> dict:
    """Load or return default model config."""
    config_path = MODELS_DIR / "model_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "distilbert_model": DISTILBERT_MODEL,
        "max_seq_len": MAX_SEQ_LEN,
        "bilstm_hidden": BILSTM_HIDDEN,
        "bilstm_embed": BILSTM_EMBED,
        "bilstm_dropout": BILSTM_DROPOUT,
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "default_epochs": DEFAULT_EPOCHS,
        "seed": DEFAULT_SEED,
    }


def setup_logging(name: str = "absa", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        fh = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
        fh.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

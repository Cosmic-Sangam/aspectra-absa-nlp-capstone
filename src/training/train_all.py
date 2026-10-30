"""Train all models sequentially."""
import sys
import logging
from src.utils.config import setup_logging
from src.training.train_extractor import train_bilstm_extractor, train_distilbert_extractor
from src.training.train_classifier import train_bilstm_classifier, train_distilbert_classifier

logger = setup_logging("train_all")


def train_all(domains=None, skip_transformer: bool = False):
    """Train all models for all domains."""
    if domains is None:
        domains = ["restaurant", "laptop"]

    for domain in domains:
        logger.info("=" * 60)
        logger.info("Training on domain: %s", domain)
        logger.info("=" * 60)

        # BiLSTM extractor
        logger.info("--- BiLSTM Aspect Extractor ---")
        try:
            train_bilstm_extractor(domain=domain, epochs=5)
        except Exception as e:
            logger.error("BiLSTM extractor failed: %s", e)

        # BiLSTM classifier
        logger.info("--- BiLSTM Sentiment Classifier ---")
        try:
            train_bilstm_classifier(domain=domain, epochs=5)
        except Exception as e:
            logger.error("BiLSTM classifier failed: %s", e)

        if not skip_transformer:
            # DistilBERT extractor
            logger.info("--- DistilBERT Aspect Extractor ---")
            try:
                train_distilbert_extractor(domain=domain, epochs=3)
            except Exception as e:
                logger.error("DistilBERT extractor failed: %s", e)

            # DistilBERT classifier
            logger.info("--- DistilBERT Sentiment Classifier ---")
            try:
                train_distilbert_classifier(domain=domain, epochs=3)
            except Exception as e:
                logger.error("DistilBERT classifier failed: %s", e)


if __name__ == "__main__":
    skip = "--skip-transformer" in sys.argv
    train_all(skip_transformer=skip)

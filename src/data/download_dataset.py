"""Download SemEval-2014 Task 4 ABSA dataset from verified academic mirrors."""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import requests
from lxml import etree

from src.utils.config import RAW_DIR, DATA_DIR, setup_logging

logger = setup_logging("download")

# Prioritised list of verified mirrors for SemEval-2014 Task 4 ABSA data
SOURCES = {
    "laptop_train": [
        "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/Laptops_Train.xml.seg",
        "https://raw.githubusercontent.com/TechBison/ABSA_LSTM-CRF/master/Laptop_Train_v2.xml",
    ],
    "laptop_test": [
        "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/Laptops_Test_Gold.xml.seg",
    ],
    "restaurant_train": [
        "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/Restaurants_Train.xml.seg",
        "https://raw.githubusercontent.com/HSLCY/ABSA-BERT-pair/master/data/semeval2014/Restaurants_Train.xml",
        "https://raw.githubusercontent.com/TechBison/ABSA_LSTM-CRF/master/Restaurants_Train_v2.xml",
    ],
    "restaurant_test": [
        "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/Restaurants_Test_Gold.xml.seg",
        "https://raw.githubusercontent.com/HSLCY/ABSA-BERT-pair/master/data/semeval2014/Restaurants_Test_Gold.xml",
    ],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 30, retries: int = 3) -> bool:
    """Download *url* to *dest* with streaming and retry logic. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            logger.info("Attempt %d: GET %s", attempt, url)
            resp = requests.get(url, timeout=timeout, stream=True)
            if resp.status_code != 200:
                logger.warning("HTTP %d for %s", resp.status_code, url)
                continue
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type and "xml" not in url.lower():
                logger.warning("Got HTML instead of data for %s", url)
                continue
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size = tmp.stat().st_size
            if size < 500:
                logger.warning("File too small (%d bytes) — likely error page", size)
                tmp.unlink(missing_ok=True)
                continue
            if dest.exists():
                dest.unlink()
            tmp.rename(dest)
            logger.info("Saved %s (%d bytes)", dest.name, size)
            return True
        except requests.RequestException as e:
            logger.warning("Request error on attempt %d: %s", attempt, e)
            time.sleep(2 * attempt)
    return False


def _validate_content(path: Path) -> bool:
    """Check that file is valid XML or valid .seg format."""
    if not path.exists() or path.stat().st_size < 500:
        return False
    try:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if content.startswith("<?xml") or content.startswith("<"):
            tree = etree.parse(str(path))
            root = tree.getroot()
            sentences = root.findall(".//sentence")
            return len(sentences) > 0
        else:
            # .seg format: 3 lines per sample ($T$ in text, aspect, polarity)
            lines = content.split("\n")
            return len(lines) >= 3 and "$T$" in lines[0]
    except Exception as e:
        logger.warning("Validation error on %s: %s", path.name, e)
        return False


def download_dataset() -> dict:
    """Download all four dataset files and return manifest dict."""
    manifest = {
        "dataset_name": "SemEval-2014 Task 4 ABSA",
        "retrieval_date": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

    file_specs = {
        "laptop_train": ("laptop", "Laptops_Train.xml.seg"),
        "laptop_test": ("laptop", "Laptops_Test_Gold.xml.seg"),
        "restaurant_train": ("restaurant", "Restaurants_Train.xml.seg"),
        "restaurant_test": ("restaurant", "Restaurants_Test_Gold.xml.seg"),
    }

    for key, urls in SOURCES.items():
        domain, fname = file_specs[key]
        dest_dir = RAW_DIR / domain
        dest = dest_dir / fname

        # Check existing
        if dest.exists() and _validate_content(dest):
            logger.info("Cached valid file found: %s", dest.name)
            manifest["files"][key] = {
                "path": str(dest),
                "filename": dest.name,
                "domain": domain,
                "split": "test" if "test" in key else "train",
                "size_bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "source": "cached",
                "valid": True,
            }
            continue

        downloaded = False
        for url in urls:
            if download_file(url, dest):
                if _validate_content(dest):
                    manifest["files"][key] = {
                        "path": str(dest),
                        "filename": dest.name,
                        "domain": domain,
                        "split": "test" if "test" in key else "train",
                        "size_bytes": dest.stat().st_size,
                        "sha256": _sha256(dest),
                        "source": url,
                        "valid": True,
                    }
                    downloaded = True
                    break
                else:
                    logger.warning("Validation failed for %s from %s", dest.name, url)
                    dest.unlink(missing_ok=True)

        if not downloaded:
            manifest["files"][key] = {"valid": False, "error": "All sources failed"}

    manifest_path = DATA_DIR / "dataset_manifest.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", manifest_path)
    return manifest


if __name__ == "__main__":
    m = download_dataset()
    for k, v in m["files"].items():
        print(f"{k}: valid={v.get('valid')} size={v.get('size_bytes')} source={v.get('source')}")

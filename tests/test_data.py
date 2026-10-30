"""Tests for dataset download and parsing."""
import json
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import RAW_DIR, PROCESSED_DIR, DATA_DIR


class TestDatasetFiles:
    """Test that dataset files exist and are valid."""

    def test_raw_laptop_train_exists(self):
        files = list((RAW_DIR / "laptop").glob("*Train*"))
        assert len(files) > 0, "Laptop training file not found"

    def test_raw_laptop_test_exists(self):
        files = list((RAW_DIR / "laptop").glob("*Test*"))
        assert len(files) > 0, "Laptop test file not found"

    def test_raw_restaurant_train_exists(self):
        files = list((RAW_DIR / "restaurant").glob("*Train*"))
        assert len(files) > 0, "Restaurant training file not found"

    def test_raw_restaurant_test_exists(self):
        files = list((RAW_DIR / "restaurant").glob("*Test*"))
        assert len(files) > 0, "Restaurant test file not found"

    def test_dataset_manifest_exists(self):
        manifest_path = DATA_DIR / "dataset_manifest.json"
        assert manifest_path.exists(), "Dataset manifest not found"

    def test_dataset_manifest_valid(self):
        manifest_path = DATA_DIR / "dataset_manifest.json"
        if not manifest_path.exists():
            pytest.skip("Manifest not yet created")
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "files" in manifest
        for key, info in manifest["files"].items():
            assert info.get("valid", False), f"{key} not valid in manifest"

    def test_raw_files_not_html(self):
        """Ensure downloaded files are not HTML error pages."""
        for domain_dir in [RAW_DIR / "laptop", RAW_DIR / "restaurant"]:
            if not domain_dir.exists():
                continue
            for f in domain_dir.glob("*"):
                content = f.read_text(encoding="utf-8", errors="replace")[:200]
                assert "<!DOCTYPE html" not in content.lower(), \
                    f"{f.name} appears to be an HTML page"

    def test_raw_files_have_content(self):
        """Files should have substantial content."""
        for domain_dir in [RAW_DIR / "laptop", RAW_DIR / "restaurant"]:
            if not domain_dir.exists():
                continue
            for f in domain_dir.glob("*"):
                assert f.stat().st_size > 1000, f"{f.name} is too small ({f.stat().st_size} bytes)"

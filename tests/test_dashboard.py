"""Tests for Dashboard frontend integration and build."""
from pathlib import Path


def test_frontend_dist_built():
    """Verify that frontend production build exists."""
    dist_dir = Path("frontend/dist")
    assert dist_dir.exists(), "frontend/dist does not exist"
    assert (dist_dir / "index.html").exists(), "frontend/dist/index.html is missing"


def test_frontend_source_files_present():
    """Verify critical frontend source files exist."""
    src_dir = Path("frontend/src")
    assert (src_dir / "main.ts").exists()
    assert (src_dir / "style.css").exists()

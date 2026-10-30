"""Parse SemEval-2014 XML files into structured data."""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

from lxml import etree

from src.utils.config import RAW_DIR, PROCESSED_DIR, DATA_DIR, setup_logging

logger = setup_logging("parser")


def parse_xml_file(xml_path: Path) -> List[Dict]:
    """Parse a SemEval-2014 XML file into a list of sentence dicts.

    Each dict has:
        - id: str
        - text: str
        - aspects: list of {term, polarity, from, to}
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    sentences = []

    for sent_elem in root.findall(".//sentence"):
        sid = sent_elem.get("id", "")
        text_elem = sent_elem.find("text")
        if text_elem is None or not text_elem.text:
            continue
        text = text_elem.text.strip()

        aspects = []
        # SemEval-2014 uses <aspectTerms><aspectTerm .../></aspectTerms>
        for at in sent_elem.findall(".//aspectTerm"):
            term = at.get("term", "")
            polarity = at.get("polarity", "").lower()
            start = int(at.get("from", -1))
            end = int(at.get("to", -1))
            aspects.append({
                "term": term,
                "polarity": polarity,
                "from": start,
                "to": end,
            })

        # Some datasets use <Opinions><Opinion .../></Opinions>
        for op in sent_elem.findall(".//Opinion"):
            term = op.get("target", "")
            if term == "NULL":
                continue
            polarity = op.get("polarity", "").lower()
            start = int(op.get("from", -1))
            end = int(op.get("to", -1))
            aspects.append({
                "term": term,
                "polarity": polarity,
                "from": start,
                "to": end,
            })

        sentences.append({"id": sid, "text": text, "aspects": aspects})

    return sentences


def parse_seg_file(seg_path: Path) -> List[Dict]:
    """Parse a .seg plain-text file (one sentence per line with $T$ markers).

    Format: sentence text where aspect is replaced by $T$
    Followed by: aspect term
    Followed by: polarity label (as integer: -1, 0, 1)
    """
    sentences_map: Dict[str, Dict] = {}
    lines = seg_path.read_text(encoding="utf-8").strip().split("\n")
    i = 0
    idx = 0
    while i < len(lines) - 2:
        raw_text = lines[i].strip()
        aspect_term = lines[i + 1].strip()
        polarity_int = lines[i + 2].strip()
        i += 3

        # Map polarity: -1 -> negative, 0 -> neutral, 1 -> positive
        pol_map = {"-1": "negative", "0": "neutral", "1": "positive"}
        polarity = pol_map.get(polarity_int, "neutral")

        # Reconstruct full text
        full_text = raw_text.replace("$T$", aspect_term)

        # Find the aspect position in the full text
        start = full_text.lower().find(aspect_term.lower())
        end = start + len(aspect_term) if start >= 0 else -1

        # Use full_text as key to group multiple aspects per sentence
        if full_text not in sentences_map:
            sentences_map[full_text] = {
                "id": str(idx),
                "text": full_text,
                "aspects": [],
            }
            idx += 1

        sentences_map[full_text]["aspects"].append({
            "term": aspect_term,
            "polarity": polarity,
            "from": start,
            "to": end,
        })

    return list(sentences_map.values())


def parse_dataset_file(path: Path) -> List[Dict]:
    """Auto-detect format and parse."""
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if content.startswith("<?xml") or content.startswith("<"):
        return parse_xml_file(path)
    else:
        return parse_seg_file(path)


def verify_offsets(sentences: List[Dict]) -> Dict:
    """Verify character offsets match the actual text."""
    stats = {"total": 0, "correct": 0, "mismatch": 0, "missing": 0}
    for sent in sentences:
        for asp in sent["aspects"]:
            stats["total"] += 1
            start, end = asp["from"], asp["to"]
            if start < 0 or end < 0:
                stats["missing"] += 1
                continue
            extracted = sent["text"][start:end]
            if extracted.lower() == asp["term"].lower():
                stats["correct"] += 1
            else:
                stats["mismatch"] += 1
                logger.debug(
                    "Offset mismatch in '%s': expected '%s' got '%s'",
                    sent["id"], asp["term"], extracted,
                )
    return stats


def get_dataset_stats(sentences: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    total_aspects = sum(len(s["aspects"]) for s in sentences)
    polarities = {}
    for s in sentences:
        for a in s["aspects"]:
            p = a["polarity"]
            polarities[p] = polarities.get(p, 0) + 1

    multi_aspect = sum(1 for s in sentences if len(s["aspects"]) > 1)
    no_aspect = sum(1 for s in sentences if len(s["aspects"]) == 0)

    return {
        "num_sentences": len(sentences),
        "total_aspects": total_aspects,
        "polarity_distribution": polarities,
        "multi_aspect_sentences": multi_aspect,
        "no_aspect_sentences": no_aspect,
    }


def parse_all_datasets() -> Dict:
    """Parse all four dataset files and save processed data."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {}

    file_map = {
        "laptop_train": RAW_DIR / "laptop" / "Laptops_Train.xml",
        "laptop_test": RAW_DIR / "laptop" / "Laptops_Test_Gold.xml",
        "restaurant_train": RAW_DIR / "restaurant" / "Restaurants_Train.xml",
        "restaurant_test": RAW_DIR / "restaurant" / "Restaurants_Test_Gold.xml",
    }

    for key, path in file_map.items():
        if not path.exists():
            # Try .seg extension
            seg_path = path.with_suffix(".xml.seg")
            if seg_path.exists():
                path = seg_path
            else:
                logger.warning("File not found: %s", path)
                continue

        sentences = parse_dataset_file(path)
        stats = get_dataset_stats(sentences)
        offset_stats = verify_offsets(sentences)
        stats["offset_verification"] = offset_stats
        all_stats[key] = stats

        # Save parsed JSON
        out_path = PROCESSED_DIR / f"{key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sentences, f, indent=2, ensure_ascii=False)
        logger.info("Parsed %s: %d sentences, %d aspects → %s",
                     key, stats["num_sentences"], stats["total_aspects"], out_path.name)

    # Save combined stats
    stats_path = PROCESSED_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    return all_stats


if __name__ == "__main__":
    stats = parse_all_datasets()
    print(json.dumps(stats, indent=2))

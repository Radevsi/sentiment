from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# Mirrors the punctuation/digit removal in the authors' public extraction code.
_PAPER_STRIP_RE = re.compile(r"['\"»•;!#$%&()*+,\-/:;<=>?@\[\]\^_`{|}~0-9]")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "language",
        "year_start",
        "year_end",
        "target_substrings",
        "model_id",
        "anchor_pairs",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    return config


def normalize_ngram(text: str) -> str:
    """Apply the paper's punctuation cleanup and collapse whitespace."""
    return " ".join(_PAPER_STRIP_RE.sub("", text).lower().split())

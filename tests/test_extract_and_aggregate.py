from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

import numpy as np

from finance_sentiment.extract import extract_to_database
from finance_sentiment.score import cosine_to_axis


def test_extract_aggregates_google_match_counts(tmp_path: Path) -> None:
    source = tmp_path / "sample.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write("La finance est bonne ici\t1900\t2\t1\n")
        handle.write("la finance est bonne ici\t1900\t3\t2\n")
        handle.write("la finance est mauvaise ici\t1901\t4\t1\n")
        handle.write("la navigation est bonne ici\t1900\t99\t5\n")
        handle.write("la finance est bonne ici\t1800\t99\t5\n")

    config = {
        "language": "fre",
        "year_start": 1870,
        "year_end": 2009,
        "target_substrings": ["financ"],
    }
    database = tmp_path / "test.sqlite"
    extract_to_database([source], config, database, commit_every=1)

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT year, text, match_count FROM contexts ORDER BY year, text"
        ).fetchall()
    assert rows == [
        (1900, "la finance est bonne ici", 5),
        (1901, "la finance est mauvaise ici", 4),
    ]


def test_cosine_to_axis() -> None:
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 1.0]])
    axis = np.array([1.0, 0.0])
    scores = cosine_to_axis(vectors, axis)
    np.testing.assert_allclose(scores, [1.0, -1.0, 2**-0.5])

from __future__ import annotations

import argparse
import gzip
import sqlite3
from pathlib import Path
from typing import Iterable, TextIO

from .common import load_config, normalize_ngram


SCHEMA = """
CREATE TABLE IF NOT EXISTS contexts (
    language TEXT NOT NULL,
    year INTEGER NOT NULL,
    text TEXT NOT NULL,
    match_count INTEGER NOT NULL,
    PRIMARY KEY (language, year, text)
);
CREATE INDEX IF NOT EXISTS contexts_text_idx ON contexts(text);
"""


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_matches(
    paths: Iterable[Path],
    *,
    targets: list[str],
    year_start: int,
    year_end: int,
) -> Iterable[tuple[int, str, int]]:
    targets = [target.lower() for target in targets]
    for path in paths:
        with open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 4:
                    continue
                raw_text, raw_year, raw_match_count = fields[:3]
                lowered = raw_text.lower()
                if not any(target in lowered for target in targets):
                    continue
                try:
                    year = int(raw_year)
                    match_count = int(raw_match_count)
                except ValueError:
                    continue
                if year_start <= year <= year_end and match_count > 0:
                    text = normalize_ngram(raw_text)
                    if text:
                        yield year, text, match_count


def extract_to_database(
    paths: list[Path], config: dict, database: Path, commit_every: int = 25_000
) -> int:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(SCHEMA)
    statement = """
        INSERT INTO contexts(language, year, text, match_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(language, year, text)
        DO UPDATE SET match_count = match_count + excluded.match_count
    """
    pending: list[tuple[str, int, str, int]] = []
    total = 0
    for year, text, match_count in iter_matches(
        paths,
        targets=config["target_substrings"],
        year_start=int(config["year_start"]),
        year_end=int(config["year_end"]),
    ):
        pending.append((config["language"], year, text, match_count))
        total += 1
        if len(pending) >= commit_every:
            connection.executemany(statement, pending)
            connection.commit()
            pending.clear()
    if pending:
        connection.executemany(statement, pending)
        connection.commit()
    connection.close()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter and aggregate Google Books 5-grams into SQLite."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    observed = extract_to_database(args.inputs, config, args.database)
    with sqlite3.connect(args.database) as connection:
        unique = connection.execute(
            "SELECT COUNT(DISTINCT text) FROM contexts WHERE language = ?",
            (config["language"],),
        ).fetchone()[0]
        language_years = connection.execute(
            "SELECT COUNT(DISTINCT year) FROM contexts WHERE language = ?",
            (config["language"],),
        ).fetchone()[0]
    print(
        f"Processed {observed:,} matching records; "
        f"stored {unique:,} unique contexts across {language_years} years."
    )


if __name__ == "__main__":
    main()

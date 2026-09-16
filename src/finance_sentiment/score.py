from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from .common import load_config
from .extract import SCHEMA


SCORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_scores (
    model_id TEXT NOT NULL,
    representation TEXT NOT NULL,
    anchor_signature TEXT NOT NULL,
    text TEXT NOT NULL,
    cosine REAL NOT NULL,
    PRIMARY KEY (model_id, representation, anchor_signature, text)
);
"""


def chunks(values: Iterable[str], size: int) -> Iterable[list[str]]:
    chunk: list[str] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def anchor_signature(config: dict) -> str:
    import hashlib
    import json

    payload = json.dumps(config["anchor_pairs"], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class PaperBertEncoder:
    """Modern implementation of the authors' sum-last-four-[CLS] convention."""

    def __init__(self, config: dict, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.config = config
        self.device = torch.device(device)
        options = {"revision": config.get("model_revision", "main")}
        if config.get("local_files_only"):
            options["local_files_only"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_id"], **options)
        self.model = AutoModel.from_pretrained(config["model_id"], **options)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        if self.config.get("lowercase_for_model", True):
            texts = [text.lower() for text in texts]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=int(self.config.get("max_length", 64)),
            return_tensors="pt",
        )
        if "token_type_ids" in encoded:
            encoded["token_type_ids"] = torch.full_like(
                encoded["token_type_ids"],
                int(self.config.get("token_type_id", 1)),
            )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded, output_hidden_states=True, return_dict=True)
            cls = torch.stack([layer[:, 0, :] for layer in output.hidden_states[-4:]], dim=0).sum(dim=0)
        return cls.float().cpu().numpy()


def build_axis(encoder: PaperBertEncoder, config: dict) -> np.ndarray:
    positives = [pair[0] for pair in config["anchor_pairs"]]
    negatives = [pair[1] for pair in config["anchor_pairs"]]
    return (encoder.encode(positives) - encoder.encode(negatives)).sum(axis=0)


def cosine_to_axis(vectors: np.ndarray, axis: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(axis)
    if np.any(denominator == 0):
        raise ValueError("Cannot calculate cosine with a zero-length vector.")
    return (vectors @ axis) / denominator


def unscored_texts(
    connection: sqlite3.Connection, config: dict, signature: str
) -> Iterable[str]:
    query = """
        SELECT DISTINCT c.text
        FROM contexts c
        LEFT JOIN context_scores s
          ON s.text = c.text
         AND s.model_id = ?
         AND s.representation = ?
         AND s.anchor_signature = ?
        WHERE c.language = ? AND s.text IS NULL
        ORDER BY c.text
    """
    cursor = connection.execute(
        query,
        (
            config["model_id"],
            config["representation"],
            signature,
            config["language"],
        ),
    )
    for row in cursor:
        yield row[0]


def write_annual(
    connection: sqlite3.Connection, config: dict, signature: str, output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    query = """
        SELECT c.year,
               SUM(c.match_count * s.cosine) / SUM(c.match_count) AS score,
               SUM(c.match_count) AS total_match_count,
               COUNT(*) AS context_year_rows
        FROM contexts c
        JOIN context_scores s ON s.text = c.text
        WHERE c.language = ?
          AND s.model_id = ?
          AND s.representation = ?
          AND s.anchor_signature = ?
        GROUP BY c.year
        ORDER BY c.year
    """
    rows = connection.execute(
        query,
        (
            config["language"],
            config["model_id"],
            config["representation"],
            signature,
        ),
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        standardization = config.get("published_global_standardization")
        writer.writerow(
            [
                "year",
                "language",
                "raw_score",
                "published_global_standardized_score",
                "total_match_count",
                "context_year_rows",
            ]
        )
        for year, score, total_count, count_rows in rows:
            standardized = ""
            if standardization:
                standardized = (score - standardization["mean"]) / standardization[
                    "sample_std"
                ]
            writer.writerow(
                [
                    year,
                    config["language"],
                    score,
                    standardized,
                    total_count,
                    count_rows,
                ]
            )


def write_extremes(
    connection: sqlite3.Connection,
    config: dict,
    signature: str,
    output: Path,
    limit: int = 50,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    base = """
        SELECT ? AS pole, s.cosine, SUM(c.match_count) AS total_match_count, s.text
        FROM context_scores s
        JOIN contexts c ON c.text = s.text
        WHERE c.language = ?
          AND s.model_id = ?
          AND s.representation = ?
          AND s.anchor_signature = ?
        GROUP BY s.text, s.cosine
        ORDER BY s.cosine {order}
        LIMIT ?
    """
    parameters = (
        config["language"],
        config["model_id"],
        config["representation"],
        signature,
        limit,
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pole", "cosine", "total_match_count", "text"])
        for pole, order in (("positive", "DESC"), ("negative", "ASC")):
            for row in connection.execute(
                base.format(order=order), (pole, *parameters)
            ):
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode contexts and calculate the annual sentiment index."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--annual-output", required=True, type=Path)
    parser.add_argument("--extremes-output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = load_config(args.config)
    signature = anchor_signature(config)
    encoder = PaperBertEncoder(config, args.device)
    axis = build_axis(encoder, config)

    connection = sqlite3.connect(args.database)
    connection.executescript(SCHEMA)
    connection.executescript(SCORE_SCHEMA)
    insert = """
        INSERT OR REPLACE INTO context_scores(
            model_id, representation, anchor_signature, text, cosine
        ) VALUES (?, ?, ?, ?, ?)
    """
    processed = 0
    # Avoid mutating context_scores while a SELECT over that table is active.
    # The French target is small enough to materialize this list in memory.
    remaining = list(unscored_texts(connection, config, signature))
    for batch in chunks(remaining, args.batch_size):
        vectors = encoder.encode(batch)
        scores = cosine_to_axis(vectors, axis)
        connection.executemany(
            insert,
            [
                (
                    config["model_id"],
                    config["representation"],
                    signature,
                    text,
                    float(score),
                )
                for text, score in zip(batch, scores, strict=True)
            ],
        )
        connection.commit()
        processed += len(batch)
        if processed % (args.batch_size * 20) == 0:
            print(f"Scored {processed:,} new contexts")

    write_annual(connection, config, signature, args.annual_output)
    write_extremes(connection, config, signature, args.extremes_output)
    connection.close()
    print(f"Scored {processed:,} new contexts; wrote {args.annual_output}")


if __name__ == "__main__":
    main()

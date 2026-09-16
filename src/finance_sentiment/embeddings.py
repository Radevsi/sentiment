"""Resumable vectors, with a durable row-to-text map and bounded RAM use."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import re
import sqlite3
import time

import numpy as np

from .corpus import statistics
from .score import PaperBertEncoder
from .tracking import log, write_json


def fsync_file(path: Path):
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def embed(root: Path, *, device="cuda", batch_size=64, dtype="float32", offline=False, encoder_factory=PaperBertEncoder):
    if batch_size < 1 or dtype not in ("float32", "float16"):
        raise ValueError("Positive batch size and float32/float16 storage required")
    identity = json.loads((root / "run.json").read_text())
    config = identity["config"]
    if not re.fullmatch(r"[0-9a-f]{40}", config.get("model_revision", "")):
        raise ValueError("Pin model_revision to a 40-character Hugging Face commit before embedding")
    with sqlite3.connect(root / "counts.sqlite") as connection:
        stats = statistics(connection, len(identity["manifest"]["shards"]))
        if not stats["retrieval_complete"]:
            raise ValueError("Finish retrieval before embedding (partial scans are not complete corpora)")
        if not stats["unique_contexts"]:
            raise ValueError("No matching contexts to embed")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS vector_rows(row_id INTEGER PRIMARY KEY, text TEXT UNIQUE NOT NULL);
            CREATE TABLE IF NOT EXISTS embedding_progress(singleton INTEGER PRIMARY KEY CHECK(singleton=1), next_row INTEGER NOT NULL);
        """)
        if not connection.execute("SELECT 1 FROM vector_rows LIMIT 1").fetchone():
            with connection:
                connection.execute("""INSERT INTO vector_rows(row_id,text)
                    SELECT ROW_NUMBER() OVER (ORDER BY text)-1,text FROM (SELECT DISTINCT text FROM contexts)""")
        total = connection.execute("SELECT COUNT(*) FROM vector_rows").fetchone()[0]
        packages = {}
        for name in ("torch", "transformers", "numpy", "tokenizers"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "not installed"
        metadata = dict(format_version=1, rows=total, dimensions=768, dtype=dtype,
                        model_id=config["model_id"], model_revision=config["model_revision"],
                        representation=config["representation"], normalized=False,
                        preprocessing={key: config.get(key) for key in ("lowercase_for_model", "max_length", "token_type_id")},
                        packages=packages)
        if config.get("representation") != "sum_last_four_cls":
            raise ValueError("Only sum_last_four_cls is implemented")
        info = root / "embeddings.json"
        if info.exists() and json.loads(info.read_text()) != metadata:
            raise ValueError("Embedding settings or package versions changed; restore the original environment or use a new run")
        path = root / "embeddings.npy"
        progress = connection.execute("SELECT next_row FROM embedding_progress WHERE singleton=1").fetchone()
        start_row = progress[0] if progress else 0
        if start_row and not path.exists():
            raise ValueError("Checkpoint exists but embeddings.npy is missing")
        if path.exists():
            matrix = np.load(path, mmap_mode="r+")
            if matrix.shape != (total, 768) or matrix.dtype != np.dtype(dtype):
                raise ValueError("Vector file shape/dtype does not match run")
        else:
            # Atomic creation avoids a half-written .npy header after an interrupted allocation.
            temporary = root / "embeddings.npy.partial"
            matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=(total, 768))
            matrix.flush()
            del matrix
            fsync_file(temporary)
            temporary.replace(path)
            matrix = np.load(path, mmap_mode="r+")
        write_json(info, metadata)
        if start_row == total:
            log(f"2 embedding already complete: {total:,} vectors")
            return
        encoder = encoder_factory({**config, "local_files_only": offline}, device)
        started = last_log = time.monotonic()
        for offset in range(start_row, total, batch_size):
            texts = [r[0] for r in connection.execute(
                "SELECT text FROM vector_rows WHERE row_id>=? ORDER BY row_id LIMIT ?", (offset, batch_size))]
            values = encoder.encode(texts)
            if values.shape != (len(texts), 768) or not np.isfinite(values).all():
                raise ValueError("Invalid encoder output")
            stored = values.astype(dtype)
            if not np.isfinite(stored).all():
                raise ValueError("Storage dtype overflow; use float32")
            end = offset + len(texts)
            matrix[offset:end] = stored
            matrix.flush()
            fsync_file(path)  # Vectors must be durable BEFORE advancing the SQLite receipt.
            with connection:
                connection.execute("INSERT OR REPLACE INTO embedding_progress VALUES (1,?)", (end,))
            now = time.monotonic()
            if now-last_log >= 30 or end == total:
                rate = (end-start_row)/max(now-started, .001)
                log(f"2 embedding {end:,}/{total:,}; elapsed {now-started:.1f}s; "
                    f"{rate:.1f} contexts/s; ETA estimate {(total-end)/max(rate,.001):.0f}s")
                last_log = now
        write_json(root / "stats.json", {**stats, "embedded_contexts": total, "embedding_complete": True})


def open_vectors(root: Path):
    """Portable read API: returns (read-only mmap, read-only SQLite connection, metadata)."""
    metadata = json.loads((root / "embeddings.json").read_text())
    database = (root / "counts.sqlite").resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(database, uri=True)
    receipt = connection.execute("SELECT next_row FROM embedding_progress WHERE singleton=1").fetchone()
    if not receipt or receipt[0] != metadata["rows"]:
        connection.close()
        raise ValueError("Embedding is incomplete; do not publish partial vector files")
    matrix = np.load(root / "embeddings.npy", mmap_mode="r")
    if matrix.shape != (metadata["rows"], metadata["dimensions"]) or matrix.dtype != np.dtype(metadata["dtype"]):
        connection.close()
        raise ValueError("Vector file does not match metadata")
    return matrix, connection, metadata

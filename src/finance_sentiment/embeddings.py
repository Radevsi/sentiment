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
from .embedding_workers import encoded_batches
from .score import PaperBertEncoder, load_bert_dependencies
from .tracking import log, write_json


def fsync_file(path: Path):
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def completed_embeddings(connection):
    """Includes durable out-of-order batches, while retaining the legacy contiguous prefix."""
    row = connection.execute("SELECT next_row FROM embedding_progress WHERE singleton=1").fetchone()
    total = row[0] if row else 0
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name='embedding_batches'").fetchone():
        total += connection.execute("SELECT COALESCE(SUM(end_row-start_row),0) FROM embedding_batches").fetchone()[0]
    return total


def missing_batches(connection, total, batch_size):
    progress = connection.execute("SELECT next_row FROM embedding_progress WHERE singleton=1").fetchone()
    cursor = progress[0] if progress else 0
    if not 0 <= cursor <= total:
        raise ValueError("Invalid embedding progress")
    saved = connection.execute("SELECT start_row,end_row FROM embedding_batches ORDER BY start_row").fetchall()
    gaps = []
    for start, end in saved:
        if not cursor <= start < end <= total:
            raise ValueError("Invalid/overlapping embedding batch receipts")
        gaps.append((cursor, start))
        cursor = end
    gaps.append((cursor, total))
    return ((start, min(start+batch_size, end)) for lo, end in gaps for start in range(lo, end, batch_size))


def record_batch(connection, start, end):
    # Caller flushes/fsyncs vector bytes BEFORE atomically recording their completion.
    with connection:
        connection.execute("INSERT INTO embedding_batches VALUES (?,?)", (start, end))
        row = connection.execute("SELECT next_row FROM embedding_progress WHERE singleton=1").fetchone()
        prefix = row[0] if row else 0
        for lo, hi in connection.execute("SELECT start_row,end_row FROM embedding_batches ORDER BY start_row"):
            if lo != prefix:
                break
            prefix = hi
        connection.execute("INSERT OR REPLACE INTO embedding_progress VALUES (1,?)", (prefix,))
        connection.execute("DELETE FROM embedding_batches WHERE end_row<=?", (prefix,))


def embed(root: Path, *, device="cuda", devices=None, batch_size=64, dtype="float32", offline=False, encoder_factory=PaperBertEncoder):
    if batch_size < 1 or dtype not in ("float32", "float16"):
        raise ValueError("Positive batch size and float32/float16 storage required")
    devices = list(devices) if devices is not None else [device]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Supply distinct devices")
    if encoder_factory is PaperBertEncoder:
        log("2 embedding preflight: checking BERT dependency imports")
        load_bert_dependencies()
    if len(devices) > 1 and encoder_factory is PaperBertEncoder:
        if any(not re.fullmatch(r"cuda:\d+", name) for name in devices):
            raise ValueError("Multiple devices must be explicit CUDA indices, e.g. cuda:0 cuda:1")
        if len({int(name.split(":")[1]) for name in devices}) != len(devices):
            raise ValueError("Supply distinct CUDA indices")
        import torch
        if not torch.cuda.is_available() or any(int(name.split(":")[1]) >= torch.cuda.device_count() for name in devices):
            raise ValueError("Requested GPU is not visible; check CUDA_VISIBLE_DEVICES and your GPU allocation")
    identity = json.loads((root / "run.json").read_text())
    config = identity["config"]
    if not re.fullmatch(r"[0-9a-f]{40}", config.get("model_revision", "")):
        raise ValueError("Pin model_revision to a 40-character Hugging Face commit before embedding")
    log("2 embedding setup: verifying corpus and preparing the context-to-vector row map")
    with sqlite3.connect(root / "counts.sqlite") as connection:
        stats = statistics(connection, len(identity["manifest"]["shards"]))
        if not stats["retrieval_complete"]:
            raise ValueError("Finish retrieval before embedding (partial scans are not complete corpora)")
        if not stats["unique_contexts"]:
            raise ValueError("No matching contexts to embed")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS vector_rows(row_id INTEGER PRIMARY KEY, text TEXT UNIQUE NOT NULL);
            CREATE TABLE IF NOT EXISTS embedding_progress(singleton INTEGER PRIMARY KEY CHECK(singleton=1), next_row INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS embedding_batches(start_row INTEGER PRIMARY KEY, end_row INTEGER NOT NULL CHECK(end_row>start_row));
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
        path = root / "embeddings.npy"
        already_done = completed_embeddings(connection)
        if info.exists():
            previous = json.loads(info.read_text())
            if previous != metadata:
                same_settings = {k: v for k, v in previous.items() if k != "packages"} == {
                    k: v for k, v in metadata.items() if k != "packages"}
                if already_done or not same_settings:
                    raise ValueError("Embedding settings or package versions changed; restore the original environment or use a new run")
                log("2 embedding: refreshing package metadata after environment repair; no vectors have been committed")
        ranges = missing_batches(connection, total, batch_size)
        if already_done and not path.exists():
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
        if already_done == total:
            log(f"2 embedding already complete: {total:,} vectors")
            return
        write_json(root / "embedding_execution.json", dict(devices=devices, batch_size_per_device=batch_size,
                                                          resumed_contexts=already_done))
        log(f"2 embedding: {len(devices)} device(s) {devices}; {batch_size} contexts per device/batch; "
            f"{already_done:,}/{total:,} already saved")
        started = last_log = time.monotonic()
        done = already_done
        device_rows = {name: 0 for name in devices}
        results = encoded_batches(connection, ranges, {**config, "local_files_only": offline}, devices, encoder_factory)
        try:
            for offset, end, values, source_device in results:
                if values.shape != (end-offset, 768) or not np.isfinite(values).all():
                    raise ValueError("Invalid encoder output")
                stored = values.astype(dtype)
                if not np.isfinite(stored).all():
                    raise ValueError("Storage dtype overflow; use float32")
                matrix[offset:end] = stored
                matrix.flush()
                fsync_file(path)
                record_batch(connection, offset, end)
                done += end-offset
                device_rows[source_device] += end-offset
                now = time.monotonic()
                if now-last_log >= 30 or done == total:
                    rate = (done-already_done)/max(now-started, .001)
                    log(f"2 embedding {done:,}/{total:,}; elapsed {now-started:.1f}s; "
                        f"aggregate {rate:.1f} contexts/s; ETA estimate {(total-done)/max(rate,.001):.0f}s; "
                        f"saved this run by device: {device_rows}")
                    last_log = now
        finally:
            results.close()
        if completed_embeddings(connection) != total:
            raise RuntimeError("Embedding workers exited before all contexts were saved")
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

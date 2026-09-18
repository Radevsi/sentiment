"""Metadata-only discovery and bounded-memory, shard-atomic corpus ingestion."""
from __future__ import annotations

import base64
import gzip
import hashlib
import http.client
import io
import json
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import sqlite3
import time
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

from .common import normalize_ngram
from .extract import SCHEMA
from .tracking import digest, log, stage, write_json

BUCKET = "https://storage.googleapis.com/storage/v1/b/books/o"
EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    name TEXT PRIMARY KEY, source_signature TEXT NOT NULL,
    matching_records INTEGER NOT NULL, scanned_records INTEGER NOT NULL,
    compressed_bytes INTEGER NOT NULL, completed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def discover(language: str, selection: str = "all") -> dict:
    prefix = f"ngrams/books/googlebooks-{language}-all-5gram-20120701-"
    records, token = [], None
    while True:
        params = {"prefix": prefix, "maxResults": 1000}
        if token:
            params["pageToken"] = token
        with urlopen(BUCKET + "?" + urlencode(params), timeout=60) as response:
            page = json.load(response)
        for item in page.get("items", []):
            if not item["name"].endswith(".gz"):
                continue
            if selection == "paper-parts" and not item["name"].endswith("_.gz"):
                continue
            records.append({
                "name": item["name"].rsplit("/", 1)[-1],
                "url": "https://storage.googleapis.com/books/" + quote(item["name"], safe="/") + "?generation=" + item["generation"],
                "size": int(item["size"]), "md5": item["md5Hash"],
                "generation": item["generation"],
            })
        token = page.get("nextPageToken")
        if not token:
            break
    if not records:
        raise ValueError(f"No 2012 five-gram shards found for {language}")
    return {"language": language, "release": "20120701", "ngram_length": 5,
            "selection": selection, "shards": sorted(records, key=lambda row: row["name"])}


class MeteredReader:
    def __init__(self, response, total: int, label: str = "shard"):
        self.response, self.total = response, total
        self.label = label
        self.count = 0
        self.md5 = hashlib.md5()
        self.start = self.last = time.monotonic()

    def read(self, size=-1):
        data = self.response.read(size)
        self.count += len(data)
        self.md5.update(data)
        now = time.monotonic()
        if now - self.last >= 30:
            rate = self.count / max(now-self.start, .001)
            log(f"{self.label}: download+decompress+filter {self.count:,}/{self.total:,} compressed bytes; "
                f"{rate/1e6:.2f} MB/s; elapsed {now-self.start:.0f}s; "
                f"shard ETA estimate {(self.total-self.count)/max(rate,1):.0f}s")
            self.last = now
        return data


def filter_stream(shard: dict, config: dict, destination: Path) -> tuple[int, int]:
    """Retain only matches; verify compressed length, MD5 and gzip CRC before import."""
    matched = scanned = 0
    targets = [s.lower() for s in config["target_substrings"]]
    single_target = targets[0] if len(targets) == 1 else None
    request = Request(shard["url"], headers={"Accept-Encoding": "identity"})
    with urlopen(request, timeout=120) as response:
        meter = MeteredReader(response, shard["size"], shard["name"])
        with gzip.GzipFile(fileobj=meter, mode="rb") as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", errors="strict") as source:
                with gzip.open(destination, "wt", encoding="utf-8", compresslevel=3) as out:
                    for line in source:
                        scanned += 1
                        fields = line.rstrip("\n").split("\t")
                        if len(fields) != 4:
                            raise ValueError(f"Unexpected 2012 record at line {scanned}")
                        raw, year, count, _volume_count = fields
                        lowered = raw.lower()
                        if not (single_target in lowered if single_target is not None
                                else any(target in lowered for target in targets)):
                            continue
                        year, count = int(year), int(count)
                        if config["year_start"] <= year <= config["year_end"] and count > 0:
                            text = normalize_ngram(raw)
                            if text:
                                out.write(json.dumps([year, text, count], ensure_ascii=False) + "\n")
                                matched += 1
        if meter.count != shard["size"] or base64.b64encode(meter.md5.digest()).decode() != shard["md5"]:
            raise ValueError(f"Source length/checksum mismatch: {shard['name']}")
    return matched, scanned


def import_shard(connection, shard, config, filtered, matched, scanned):
    """Counts and receipt commit together; any interruption rolls back the whole shard."""
    with connection:
        if connection.execute("SELECT 1 FROM sources WHERE name=?", (shard["name"],)).fetchone():
            return
        with gzip.open(filtered, "rt", encoding="utf-8") as handle:
            rows = ((config["language"], *json.loads(line)) for line in handle)
            connection.executemany("""
                INSERT INTO contexts(language,year,text,match_count) VALUES (?,?,?,?)
                ON CONFLICT(language,year,text) DO UPDATE SET match_count=match_count+excluded.match_count
            """, rows)
        connection.execute("INSERT INTO sources(name,source_signature,matching_records,scanned_records,compressed_bytes) VALUES (?,?,?,?,?)",
                           (shard["name"], digest(shard), matched, scanned, shard["size"]))


def bind_run(root: Path, config: dict, manifest: dict) -> None:
    if manifest["language"] != config["language"] or manifest["release"] != "20120701" or manifest["ngram_length"] != 5:
        raise ValueError("Manifest must match config language and 2012 five-grams")
    names = [s["name"] for s in manifest["shards"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Manifest must have nonempty unique shard names")
    identity = {"pipeline_version": 1, "config": config, "manifest": manifest}
    path = root / "run.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Run configuration/manifest changed; use a new run directory")
    else:
        if (root / "counts.sqlite").exists():
            raise ValueError("Untracked counts.sqlite exists; use an empty run directory")
        write_json(path, identity)


def shard_paths(root: Path, shard: dict) -> tuple[Path, Path]:
    directory = root / ".matching-shards"
    key = digest(shard)
    return directory / f"{key}.jsonl.gz", directory / f"{key}.ready.json"


def prepare_shard(root: Path, shard: dict, config: dict, retries: int) -> tuple[Path, int, int]:
    """Workers only write their own filtered files; the parent alone writes SQLite."""
    filtered, receipt = shard_paths(root, shard)
    filtered.parent.mkdir(parents=True, exist_ok=True)
    signature = digest({"shard": shard, "config": config})
    if receipt.exists() and filtered.exists():
        saved = json.loads(receipt.read_text())
        if saved["signature"] != signature:
            raise ValueError(f"Filtered checkpoint does not match {shard['name']}")
        log(f"reuse verified filtered checkpoint: {shard['name']}")
        return filtered, saved["matched"], saved["scanned"]
    temporary = filtered.with_suffix(filtered.suffix + ".partial")
    with stage(f"1 filter / {shard['name']}"):
        for attempt in range(1, retries+1):
            try:
                matched, scanned = filter_stream(shard, config, temporary)
                break
            except (OSError, EOFError, ValueError, http.client.HTTPException) as error:
                log(f"{shard['name']}: stream attempt {attempt}/{retries} failed: {error}")
                if attempt == retries:
                    raise
                time.sleep(min(2**attempt, 30))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(filtered)
        write_json(receipt, {"signature": signature, "matched": matched, "scanned": scanned})
    return filtered, matched, scanned


def remove_filtered(root: Path, shard: dict) -> None:
    filtered, receipt = shard_paths(root, shard)
    receipt.unlink(missing_ok=True)
    filtered.unlink(missing_ok=True)
    filtered.with_suffix(filtered.suffix + ".partial").unlink(missing_ok=True)


def prepare_worker(sender, root, shard, config, retries):
    # The parent handles Ctrl-C and terminates its workers before releasing the run lock.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        sender.send((True, prepare_shard(root, shard, config, retries)))
    except Exception as error:
        sender.send((False, error))
    finally:
        sender.close()


def prepared_shards(root, remaining, config, retries, workers):
    if workers == 1:
        for shard in remaining:
            yield shard, prepare_shard(root, shard, config, retries)
        return
    # Spawn prevents inheritance of the parent's open SQLite connection or run lock.
    # At most `workers` jobs are in flight. A slow importer cannot queue the whole corpus on disk.
    context = multiprocessing.get_context("spawn")
    pending = []
    try:
        todo = iter(remaining)
        exhausted = False
        while pending or not exhausted:
            while len(pending) < workers and not exhausted:
                shard = next(todo, None)
                if shard is None:
                    exhausted = True
                else:
                    receiver, sender = context.Pipe(duplex=False)
                    process = context.Process(target=prepare_worker, args=(sender, root, shard, config, retries))
                    try:
                        process.start()
                    except BaseException:
                        receiver.close()
                        if process.pid is not None:
                            process.terminate()
                            process.join()
                        raise
                    finally:
                        sender.close()
                    pending.append((shard, process, receiver))
            if not pending:
                continue
            wait([receiver for _, _, receiver in pending] + [process.sentinel for _, process, _ in pending])
            for i, (shard, process, receiver) in enumerate(pending):
                if receiver.poll():
                    try:
                        success, result = receiver.recv()
                    except EOFError as error:
                        raise RuntimeError(f"Worker exited without a result: {shard['name']}; rerun to resume") from error
                    process.join()
                    process.close()
                    receiver.close()
                    pending.pop(i)
                    if not success:
                        raise result
                    yield shard, result
                    break
                if process.exitcode is not None:
                    raise RuntimeError(f"Worker exited with code {process.exitcode}: {shard['name']}; rerun to resume")
    finally:
        for _, process, receiver in pending:
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
            receiver.close()


def retrieve(root: Path, config: dict, manifest: dict, limit: int | None = None, retries: int = 3, workers: int = 1) -> None:
    if workers < 1 or retries < 1 or (limit is not None and limit < 1):
        raise ValueError("workers, retries, and limit must be positive")
    bind_run(root, config, manifest)
    size = sum(shard["size"] for shard in manifest["shards"])
    log(f"1 retrieval: {manifest['selection']}; {len(manifest['shards'])} source shards; "
        f"{size/1e9:.2f} GB compressed; targets={config['target_substrings']}; "
        f"years={config['year_start']}-{config['year_end']}; {workers} worker(s); completed shards will be skipped")
    database = root / "counts.sqlite"
    with sqlite3.connect(database) as connection:
        # Rollback journal, not WAL: single writer and cluster filesystem compatibility.
        connection.executescript(SCHEMA + EXTRA_SCHEMA)
        completed = {name for name, in connection.execute("SELECT name FROM sources")}
        for shard in manifest["shards"]:
            if shard["name"] in completed:
                remove_filtered(root, shard)
        # Legacy single-worker temporary data has no verified completion receipt.
        (root / "matching-shard.jsonl.gz.partial").unlink(missing_ok=True)
        remaining = [s for s in manifest["shards"] if s["name"] not in completed]
        if limit is not None:
            remaining = remaining[:limit]
        if remaining:
            prepared = prepared_shards(root, remaining, config, retries, min(workers, len(remaining)))
            try:
                for i, (shard, (filtered, matched, scanned)) in enumerate(prepared, 1):
                    with stage(f"1 import / {shard['name']}"):
                        import_shard(connection, shard, config, filtered, matched, scanned)
                    remove_filtered(root, shard)
                    log(f"committed {len(completed)+i}/{len(manifest['shards'])} shards; "
                        f"{shard['name']}: retained {matched:,} matching records / {scanned:,} scanned")
            finally:
                # Also terminate workers if the importer fails or the user interrupts.
                prepared.close()
        summary = statistics(connection, len(manifest["shards"]))
        write_json(root / "stats.json", summary)
        log(json.dumps(summary))


def statistics(connection, expected: int) -> dict:
    done, matches, scanned, size = connection.execute(
        "SELECT COUNT(*),COALESCE(SUM(matching_records),0),COALESCE(SUM(scanned_records),0),COALESCE(SUM(compressed_bytes),0) FROM sources"
    ).fetchone()
    rows, unique, years, occurrences = connection.execute(
        "SELECT COUNT(*),COUNT(DISTINCT text),COUNT(DISTINCT year),COALESCE(SUM(match_count),0) FROM contexts"
    ).fetchone()
    return dict(completed_shards=done, expected_shards=expected, retrieval_complete=done==expected,
                matching_source_records=matches, scanned_source_records=scanned, compressed_bytes_processed=size,
                context_year_rows=rows, unique_contexts=unique, years=years, total_match_count=occurrences,
                float32_vector_bytes=unique*768*4, float16_vector_bytes=unique*768*2)

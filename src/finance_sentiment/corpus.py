"""Metadata-only discovery and bounded-memory, shard-atomic corpus ingestion."""
from __future__ import annotations

import base64
import gzip
import hashlib
import http.client
import io
import json
from pathlib import Path
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
    def __init__(self, response, total: int):
        self.response, self.total = response, total
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
            log(f"compressed stream {self.count:,}/{self.total:,} bytes; "
                f"{rate/1e6:.2f} MB/s; shard ETA estimate {(self.total-self.count)/max(rate,1):.0f}s")
            self.last = now
        return data


def filter_stream(shard: dict, config: dict, destination: Path) -> tuple[int, int]:
    """Retain only matches; verify compressed length, MD5 and gzip CRC before import."""
    matched = scanned = 0
    targets = [s.lower() for s in config["target_substrings"]]
    request = Request(shard["url"], headers={"Accept-Encoding": "identity"})
    with urlopen(request, timeout=120) as response:
        meter = MeteredReader(response, shard["size"])
        with gzip.GzipFile(fileobj=meter, mode="rb") as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", errors="strict") as source:
                with gzip.open(destination, "wt", encoding="utf-8", compresslevel=3) as out:
                    for line in source:
                        scanned += 1
                        fields = line.rstrip("\n").split("\t")
                        if len(fields) != 4:
                            raise ValueError(f"Unexpected 2012 record at line {scanned}")
                        raw, year, count, _volume_count = fields
                        if not any(target in raw.lower() for target in targets):
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


def retrieve(root: Path, config: dict, manifest: dict, limit: int | None = None, retries: int = 3) -> None:
    bind_run(root, config, manifest)
    size = sum(shard["size"] for shard in manifest["shards"])
    log(f"1 retrieval: {manifest['selection']}; {len(manifest['shards'])} source shards; "
        f"{size/1e9:.2f} GB compressed; targets={config['target_substrings']}; "
        f"years={config['year_start']}-{config['year_end']}; completed shards will be skipped")
    database = root / "counts.sqlite"
    with sqlite3.connect(database) as connection:
        # Rollback journal, not WAL: single writer and cluster filesystem compatibility.
        connection.executescript(SCHEMA + EXTRA_SCHEMA)
        completed = {name for name, in connection.execute("SELECT name FROM sources")}
        remaining = [s for s in manifest["shards"] if s["name"] not in completed]
        if limit is not None:
            remaining = remaining[:limit]
        for i, shard in enumerate(remaining, 1):
            with stage(f"1 retrieval / shard {len(completed)+i}/{len(manifest['shards'])}: {shard['name']}"):
                temporary = root / "matching-shard.jsonl.gz.partial"
                for attempt in range(1, retries+1):
                    try:
                        matched, scanned = filter_stream(shard, config, temporary)
                        break
                    except (OSError, EOFError, ValueError, http.client.HTTPException) as error:
                        log(f"stream attempt {attempt}/{retries} failed: {error}")
                        if attempt == retries:
                            raise
                        time.sleep(min(2**attempt, 30))
                import_shard(connection, shard, config, temporary, matched, scanned)
                temporary.unlink(missing_ok=True)
                log(f"retained {matched:,} matching records / {scanned:,} scanned")
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

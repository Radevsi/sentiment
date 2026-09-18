"""All-row UMAP export and loopback-only, dependency-free WebGL map server.

This module only reads the existing run. No corpus or model is downloaded.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .embeddings import open_vectors, fsync_file
from .tracking import digest, log, run_lock, stage, write_json

FORMAT = 1
STATIC = Path(__file__).with_name("map_static")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "umap-learn", "pynndescent", "numba", "llvmlite", "scikit-learn", "scipy")}


@contextlib.contextmanager
def heartbeat(label):
    """Keep long native/JIT calls visibly alive without fabricating progress/ETA."""
    started = time.monotonic()
    stop = threading.Event()

    def report():
        while not stop.wait(30):
            log(f"{label}; elapsed {time.monotonic()-started:.0f}s; still working; ETA unavailable")

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def export_metadata(connection, path, total):
    """Stream row-aligned aggregates; never rely on SQLite's incidental row order."""
    path.unlink(missing_ok=True)
    hasher = hashlib.sha256()
    started = time.monotonic()
    with sqlite3.connect(path) as out:
        out.execute("CREATE TABLE points (row_id INTEGER PRIMARY KEY, text TEXT NOT NULL, "
                    "total_match_count INTEGER NOT NULL, year_count INTEGER NOT NULL, "
                    "first_year INTEGER NOT NULL, last_year INTEGER NOT NULL)")
        cursor = connection.execute("""
            SELECT v.row_id,v.text,SUM(c.match_count),COUNT(DISTINCT c.year),MIN(c.year),MAX(c.year)
            FROM vector_rows v LEFT JOIN contexts c ON c.text=v.text
            GROUP BY v.row_id,v.text ORDER BY v.row_id
        """)
        count = 0
        for row in cursor:
            if row[0] != count or row[2] is None or row[3] < 1:
                raise ValueError("Invalid vector-to-text mapping or missing context counts")
            out.execute("INSERT INTO points VALUES (?,?,?,?,?,?)", row)
            hasher.update(json.dumps(row, ensure_ascii=False).encode() + b"\n")
            count += 1
            if count % 10000 == 0 or count == total:
                elapsed = time.monotonic()-started
                rate = count / max(elapsed, .001)
                log(f"1 metadata {count:,}/{total:,}; elapsed {elapsed:.1f}s; {rate:.0f} rows/s; "
                    f"ETA estimate {max(0,total-count)/max(rate,.001):.0f}s")
        if count != total:
            raise ValueError("Vector row count does not match matrix")
    fsync_file(path)
    return hasher.hexdigest()


def fit_projection(matrix, parameters):
    from umap import UMAP
    return UMAP(**parameters).fit_transform(matrix)


def validate_points(path, rows):
    if path.stat().st_size != rows * 2 * 4:
        raise ValueError("Coordinate payload size does not match all rows")
    points = np.memmap(path, mode="r", dtype="<f4", shape=(rows, 2))
    if not np.isfinite(points).all():
        raise ValueError("Projection contains nonfinite coordinates")
    return points


def validate_artifact(output):
    info = json.loads((output / "map.json").read_text())
    if info.get("format_version") != FORMAT or info.get("selection") != "all":
        raise ValueError("Unsupported or partial map artifact")
    rows = info["rows"]
    if not isinstance(rows, int) or rows < 3:
        raise ValueError("Invalid map row count")
    for name in ("points.f32", "metadata.sqlite"):
        if sha256(output / name) != info["files"][name]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    validate_points(output / "points.f32", rows)
    with contextlib.closing(sqlite3.connect((output / "metadata.sqlite").resolve().as_uri()+"?mode=ro", uri=True)) as conn:
        count, lo, hi = conn.execute("SELECT COUNT(*),MIN(row_id),MAX(row_id) FROM points").fetchone()
        if (count, lo, hi) != (rows, 0, rows-1):
            raise ValueError("Artifact row mapping is not contiguous")
    return info


def build(root, output, *, neighbors=30, min_dist=.1, epochs=500, seed=42, expected_rows=None):
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Use a separate output directory outside the existing run")
    if neighbors < 2 or not math.isfinite(min_dist) or not 0 <= min_dist <= 1 or epochs < 10 or not 0 <= seed < 2**32:
        raise ValueError("Require neighbors >=2, min_dist in [0,1], epochs >=10 and a uint32 seed")
    started = time.monotonic()
    with run_lock(output), stage("map build overall"):
        with stage("1/3 validate saved vectors and aggregate metadata"), heartbeat("1/3 validating/exporting"):
            matrix, connection, metadata = open_vectors(root)
            try:
                rows = len(matrix)
                if rows < 3 or neighbors >= rows:
                    raise ValueError("Require at least 3 rows and neighbors smaller than row count")
                if expected_rows is not None and rows != expected_rows:
                    raise ValueError(f"Expected {expected_rows:,} rows, found {rows:,}")
                identity = json.loads((root / "run.json").read_text())
                config = identity["config"]
                for key in ("model_id", "model_revision", "representation"):
                    if metadata[key] != config[key]:
                        raise ValueError(f"Run and vector identity disagree: {key}")
                for start in range(0, rows, 8192):
                    block = matrix[start:start+8192]
                    if not np.isfinite(block).all() or np.any(np.linalg.norm(block, axis=1) == 0):
                        raise ValueError("Cosine projection requires finite nonzero vectors")
                mapping_hash = export_metadata(connection, output / "metadata.partial.sqlite", rows)
            finally:
                connection.close()
            params = dict(n_components=2, metric="cosine", n_neighbors=neighbors,
                          min_dist=min_dist, n_epochs=epochs, random_state=seed, transform_seed=seed,
                          n_jobs=1, low_memory=True, force_approximation_algorithm=True,
                          init="random", verbose=True)
            source = dict(run_name=root.name, run_sha256=digest(identity), config=config,
                          embeddings=metadata, vectors_sha256=sha256(root / "embeddings.npy"),
                          mapping_and_counts_sha256=mapping_hash)
            request = dict(format_version=FORMAT, selection="all", rows=rows,
                           source=source, projection=params, packages=versions())
            signature = digest(request)
            receipt_path = output / "request.json"
            if receipt_path.exists() and json.loads(receipt_path.read_text()) != request:
                raise ValueError("Inputs, parameters or package versions changed; choose a new output directory")
            if (output / "map.json").exists():
                info = validate_artifact(output)
                if info["signature"] != signature:
                    raise ValueError("Map signature differs from current request")
                (output / "metadata.partial.sqlite").unlink()
                log(f"All {rows:,} points already complete and verified; reused cache")
                return info
            write_json(receipt_path, request)
            (output / "metadata.partial.sqlite").replace(output / "metadata.sqlite")
        with stage("2/3 all-row UMAP projection"), heartbeat("2/3 UMAP"):
            checkpoint = output / "projection.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved["signature"] != signature or saved["sha256"] != sha256(output / "points.f32"):
                    raise ValueError("Invalid projection checkpoint; choose a new output directory")
                log("2/3 using completed projection checkpoint")
            else:
                log(f"2/3 fitting ALL {rows:,} saved {matrix.shape[1]}D vectors; CPU, seed={seed}; no BERT inference")
                values = np.asarray(fit_projection(matrix, params), dtype="<f4")
                if values.shape != (rows, 2) or not np.isfinite(values).all():
                    raise ValueError("UMAP returned invalid coordinates; no checkpoint published")
                values.tofile(output / "points.partial.f32")
                fsync_file(output / "points.partial.f32")
                (output / "points.partial.f32").replace(output / "points.f32")
                write_json(checkpoint, dict(signature=signature, sha256=sha256(output / "points.f32")))
            points = validate_points(output / "points.f32", rows)
        with stage("3/3 publish and verify compact map artifact"):
            info = {**request, "signature": signature,
                    "coordinate_format": "little-endian float32 interleaved x,y; offset i equals row_id i",
                    "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                    "files": {name: sha256(output / name) for name in ("points.f32", "metadata.sqlite")},
                    "build_elapsed_seconds": time.monotonic()-started}
            write_json(output / "map.json", info)
            validate_artifact(output)
        log(f"Ready: {rows:,} / {rows:,} points; total elapsed {time.monotonic()-started:.1f}s; {output}")
        return info


def make_server(output, port=8765):
    output = output.resolve()
    info = validate_artifact(output)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, content_type="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            try:
                if url.path in ("/", "/viewer.js", "/style.css"):
                    name, mime = {"/": ("index.html", "text/html; charset=utf-8"),
                                  "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
                                  "/style.css": ("style.css", "text/css; charset=utf-8")}[url.path]
                    self.reply((STATIC / name).read_bytes(), mime)
                elif url.path == "/api/map":
                    self.reply(json.dumps(info, ensure_ascii=False).encode())
                elif url.path == "/points.f32":
                    self.reply((output / "points.f32").read_bytes(), "application/octet-stream")
                elif url.path in ("/api/point", "/api/find"):
                    with contextlib.closing(sqlite3.connect((output / "metadata.sqlite").as_uri()+"?mode=ro", uri=True)) as conn:
                        conn.row_factory = sqlite3.Row
                        if url.path == "/api/point":
                            row_id = int(query.get("id", [""])[0])
                            result = conn.execute("SELECT * FROM points WHERE row_id=?", (row_id,)).fetchone()
                            if result is None:
                                self.reply(b'{"error":"Unknown row_id"}', status=404)
                                return
                            payload = dict(result)
                        else:
                            text = query.get("text", [""])[0].strip()
                            if not 2 <= len(text) <= 200:
                                raise ValueError("Literal search requires 2 to 200 characters")
                            payload = [dict(row) for row in conn.execute(
                                "SELECT * FROM points WHERE instr(text,?)>0 ORDER BY row_id LIMIT 50", (text,))]
                        self.reply(json.dumps(payload, ensure_ascii=False).encode())
                else:
                    self.reply(b'{"error":"Not found"}', status=404)
            except (ValueError, OverflowError) as error:
                self.reply(json.dumps({"error": str(error)}).encode(), status=400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    project = sub.add_parser("build", help="Project every stored vector; never embeds or downloads")
    project.add_argument("--run-dir", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)
    project.add_argument("--neighbors", type=int, default=30)
    project.add_argument("--min-dist", type=float, default=.1)
    project.add_argument("--epochs", type=int, default=500)
    project.add_argument("--seed", type=int, default=42)
    project.add_argument("--expected-rows", type=int)
    server = sub.add_parser("serve", help="Serve an already computed map on IPv4 loopback")
    server.add_argument("--output", type=Path, required=True)
    server.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "build":
        build(args.run_dir, args.output, neighbors=args.neighbors, min_dist=args.min_dist,
              epochs=args.epochs, seed=args.seed, expected_rows=args.expected_rows)
    else:
        with stage("1/1 verify and start loopback map server"):
            httpd = make_server(args.output, args.port)
        log(f"Map: http://127.0.0.1:{httpd.server_port} (Ctrl-C stops server)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()


if __name__ == "__main__":
    main()

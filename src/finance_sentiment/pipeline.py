"""Cluster entry point. No corpus downloads occur on installation or import."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from .common import load_config
from .corpus import discover, retrieve, statistics
from .tracking import run_lock, stage, write_json, log


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description="Retrieve, track and embed 2012 finance five-grams")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="Fetch file sizes/checksums only, never corpus contents")
    manifest.add_argument("--language", default="fre")
    manifest.add_argument("--selection", choices=["all", "paper-parts"], default="all")
    manifest.add_argument("--output", type=Path, required=True)
    fetch = commands.add_parser("fetch", help="Stream all source bytes; keep matching data only")
    fetch.add_argument("--config", required=True, type=Path)
    fetch.add_argument("--manifest", required=True, type=Path)
    fetch.add_argument("--run-dir", required=True, type=Path)
    fetch.add_argument("--limit", type=positive_int, help="Process at most this many unfinished shards (pilot); rerun without limit to finish")
    fetch.add_argument("--retries", type=positive_int, default=3)
    fetch.add_argument("--workers", type=positive_int, default=1,
                       help="Independent shard-filter processes; one coordinated database writer (default: 1)")
    vectors = commands.add_parser("embed", help="Save one BERT vector per distinct context; no anchors needed")
    vectors.add_argument("--run-dir", required=True, type=Path)
    device_options = vectors.add_mutually_exclusive_group()
    device_options.add_argument("--device", default="cuda")
    device_options.add_argument("--devices", nargs="+", help="One worker per GPU, e.g. cuda:0 cuda:1 cuda:2 cuda:3")
    vectors.add_argument("--batch-size", type=positive_int, default=64, help="Contexts per batch per device")
    vectors.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    vectors.add_argument("--offline", action="store_true", help="Use a previously populated model cache")
    status = commands.add_parser("status")
    status.add_argument("--run-dir", required=True, type=Path)
    prepare = commands.add_parser("prepare-model", help="Download pinned BERT weights/tokenizer, not the corpus")
    prepare.add_argument("--config", required=True, type=Path)
    projector = commands.add_parser("projector", help="Export a bounded sample for a local embedding viewer")
    projector.add_argument("--run-dir", required=True, type=Path)
    projector.add_argument("--output", required=True, type=Path)
    projector.add_argument("--limit", type=positive_int, default=5000)
    args = parser.parse_args()
    with stage(f"overall / {args.command}"):
        if args.command == "manifest":
            data = discover(args.language, args.selection)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, data)
            size = sum(s["size"] for s in data["shards"])
            log(f"{len(data['shards'])} shards; {size:,} compressed bytes ({size/1e9:.2f} GB / {size/2**30:.2f} GiB)")
        elif args.command == "prepare-model":
            from .score import PaperBertEncoder
            PaperBertEncoder(load_config(args.config), "cpu")
            log("Pinned model/tokenizer cache ready; use the same HF_HOME on the GPU node")
        elif args.command == "status":
            identity = json.loads((args.run_dir / "run.json").read_text())
            with sqlite3.connect((args.run_dir / "counts.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as conn:
                stats = statistics(conn, len(identity["manifest"]["shards"]))
                if conn.execute("SELECT 1 FROM sqlite_master WHERE name='embedding_progress'").fetchone():
                    from .embeddings import completed_embeddings
                    stats["embedded_contexts"] = completed_embeddings(conn)
                    stats["embedding_complete"] = stats["embedded_contexts"] == stats["unique_contexts"]
                print(json.dumps(stats, indent=2))
        else:
            with run_lock(args.run_dir):
                if args.command == "fetch":
                    retrieve(args.run_dir, load_config(args.config), json.loads(args.manifest.read_text()), args.limit, args.retries, args.workers)
                elif args.command == "embed":
                    from .embeddings import embed
                    embed(args.run_dir, device=args.device, devices=args.devices, batch_size=args.batch_size, dtype=args.dtype, offline=args.offline)
                elif args.command == "projector":
                    from .search import export_projector
                    export_projector(args.run_dir, args.output, args.limit)


if __name__ == "__main__":
    main()

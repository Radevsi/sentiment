"""Local retrieval API and CPU query CLI; an HTTP service can wrap these later."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import normalize_ngram
from .embeddings import open_vectors
from .score import PaperBertEncoder


def nearest(matrix, vector, k=10, block_size=8192):
    if k < 1 or block_size < 1:
        raise ValueError("k and block_size must be positive")
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if not norm or not np.isfinite(norm):
        raise ValueError("Query must be a finite nonzero vector")
    vector = vector / norm
    best = []
    for start in range(0, len(matrix), block_size):
        block = np.asarray(matrix[start:start+block_size], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1)
        scores = (block @ vector) / np.maximum(norms, 1e-30)
        scores[norms == 0] = -np.inf
        count = min(k, len(scores))
        ids = np.argpartition(scores, len(scores)-count)[-count:]
        best.extend((float(scores[i]), int(start+i)) for i in ids)
        best = sorted(best, key=lambda x: (-x[0], x[1]))[:k]
    return best


def query(root: Path, text: str, *, k=10, device="cpu", offline=False):
    cleaned = normalize_ngram(text)
    if not cleaned:
        raise ValueError("Query is empty after preprocessing")
    matrix, conn, _ = open_vectors(root)
    try:
        config = json.loads((root / "run.json").read_text())["config"]
        encoder = PaperBertEncoder({**config, "local_files_only": offline}, device)
        vector = encoder.encode([cleaned])[0]
        results = []
        for score, row_id in nearest(matrix, vector, k):
            context = conn.execute("SELECT text FROM vector_rows WHERE row_id=?", (row_id,)).fetchone()[0]
            count = conn.execute("SELECT SUM(match_count) FROM contexts WHERE text=?", (context,)).fetchone()[0]
            results.append(dict(row_id=row_id, cosine=score, text=context, total_match_count=count))
        return {"query": text, "cleaned_query": cleaned, "neighbors": results}
    finally:
        conn.close()


def export_projector(root: Path, output: Path, limit=5000):
    matrix, conn, metadata = open_vectors(root)
    try:
        output.mkdir(parents=True, exist_ok=True)
        # Reproducible uniform sample; alphabetical prefixes would bias the visualizer.
        rng = np.random.default_rng(42)
        ids = np.sort(rng.choice(len(matrix), min(limit, len(matrix)), replace=False))
        with (output / "vectors.tsv").open("w") as vectors, (output / "metadata.tsv").open("w", encoding="utf-8", newline="") as labels:
            writer = csv.writer(labels, delimiter="\t")
            writer.writerow(["row_id", "text"])
            for i in ids:
                np.savetxt(vectors, matrix[int(i):int(i)+1], delimiter="\t", fmt="%.8g")
                writer.writerow([int(i), conn.execute("SELECT text FROM vector_rows WHERE row_id=?", (int(i),)).fetchone()[0]])
        from .tracking import write_json
        write_json(output / "sample.json", {"seed": 42, "sample_rows": len(ids), "corpus_rows": len(matrix), "model_revision": metadata["model_revision"]})
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Embed a query and find nearest stored contexts (CPU default)")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(query(args.run_dir, args.text, k=args.k, device=args.device, offline=args.offline), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

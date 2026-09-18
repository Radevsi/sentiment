from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from finance_sentiment.corpus import retrieve, import_shard, shard_paths, SCHEMA, EXTRA_SCHEMA
from finance_sentiment.embeddings import embed, open_vectors
from finance_sentiment.search import nearest, export_projector
from finance_sentiment.tracking import run_lock


@pytest.fixture
def corpus(tmp_path):
    config = dict(language="fre", year_start=1870, year_end=2009, target_substrings=["financ"],
                  model_id="fake", model_revision="a"*40, representation="sum_last_four_cls",
                  lowercase_for_model=True, token_type_id=1, max_length=64)
    shards = []
    samples = [
        "La finance est bonne ici\t1900\t2\t1\nla finance est bonne ici\t1900\t3\t1\n"
        "la finance est bonne ici\t1901\t4\t1\nles financiers sont mauvais ici\t2009\t7\t1\n"
        "la navigation est bonne ici\t1900\t99\t1\nla finance est bonne ici\t1869\t99\t1\n",
        "la finance est bonne ici\t1900\t5\t1\nle financement est utile ici\t1870\t6\t1\n"
        "la finance est bonne ici\t2010\t88\t1\n",
    ]
    for i, sample in enumerate(samples):
        path = tmp_path / f"shard-{i}.gz"
        path.write_bytes(gzip.compress(sample.encode()))
        shards.append(dict(name=path.name, url=path.as_uri(), size=path.stat().st_size,
                           md5=base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode(), generation="1"))
    manifest = dict(language="fre", release="20120701", ngram_length=5, selection="all", shards=shards)
    root = tmp_path / "run"
    root.mkdir()
    return root, config, manifest


def test_retrieval_filters_aggregates_and_resumes_without_double_count(corpus):
    root, config, manifest = corpus
    retrieve(root, config, manifest, limit=1)
    assert not json.loads((root / "stats.json").read_text())["retrieval_complete"]
    with pytest.raises(ValueError, match="Finish retrieval"):
        embed(root)
    retrieve(root, config, manifest)
    retrieve(root, config, manifest)
    stats = json.loads((root / "stats.json").read_text())
    assert stats["retrieval_complete"]
    assert stats["unique_contexts"] == 3
    assert stats["matching_source_records"] == 6
    assert stats["total_match_count"] == 27
    with sqlite3.connect(root / "counts.sqlite") as conn:
        assert conn.execute("SELECT match_count FROM contexts WHERE year=1900").fetchone()[0] == 10
    assert not (root / "matching-shard.jsonl.gz.partial").exists()


def test_changed_config_refused(corpus):
    root, config, manifest = corpus
    retrieve(root, config, manifest, limit=1)
    with pytest.raises(ValueError, match="changed"):
        retrieve(root, {**config, "target_substrings": ["banqu"]}, manifest)


def test_corrupt_source_not_committed_then_can_resume(corpus):
    root, config, manifest = corpus
    source = Path(manifest["shards"][0]["url"].removeprefix("file://"))
    original = source.read_bytes()
    source.write_bytes(original[:-5])
    with pytest.raises((EOFError, OSError)):
        retrieve(root, config, manifest, retries=1)
    with sqlite3.connect(root / "counts.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM contexts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0
    source.write_bytes(original)
    retrieve(root, config, manifest, retries=1)
    assert json.loads((root / "stats.json").read_text())["retrieval_complete"]


def test_checksum_mismatch_refused(corpus):
    root, config, manifest = corpus
    manifest["shards"][0]["md5"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        retrieve(root, config, manifest, retries=1)
    with sqlite3.connect(root / "counts.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM contexts").fetchone()[0] == 0


def test_import_error_rolls_back_entire_shard(corpus, tmp_path):
    root, config, manifest = corpus
    filtered = tmp_path / "bad.jsonl.gz"
    with gzip.open(filtered, "wt") as out:
        out.write('[1900,"la finance",2]\nnot-json\n')
    with sqlite3.connect(root / "counts.sqlite") as conn:
        conn.executescript(SCHEMA + EXTRA_SCHEMA)
        with pytest.raises(json.JSONDecodeError):
            import_shard(conn, manifest["shards"][0], config, filtered, 2, 2)
        assert conn.execute("SELECT COUNT(*) FROM contexts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0


class FakeEncoder:
    seen = []

    def __init__(self, config, device):
        pass

    def encode(self, texts):
        self.seen.extend(texts)
        return np.array([[len(text), *([1]*767)] for text in texts], dtype=np.float32)


class InterruptEncoder(FakeEncoder):
    calls = 0

    def encode(self, texts):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated interruption")
        return super().encode(texts)


def test_embedding_resume_mapping_and_portable_read(corpus, tmp_path):
    root, config, manifest = corpus
    retrieve(root, config, manifest)
    with pytest.raises(RuntimeError, match="interruption"):
        embed(root, batch_size=1, encoder_factory=InterruptEncoder)
    with pytest.raises(ValueError, match="incomplete"):
        open_vectors(root)
    FakeEncoder.seen = []
    embed(root, batch_size=1, encoder_factory=FakeEncoder)
    assert len(FakeEncoder.seen) == 2  # Already saved row must not be recomputed.
    embed(root, batch_size=1, encoder_factory=FakeEncoder)
    assert len(FakeEncoder.seen) == 2
    matrix, conn, info = open_vectors(root)
    texts = [r[0] for r in conn.execute("SELECT text FROM vector_rows ORDER BY row_id")]
    assert info["rows"] == 3
    np.testing.assert_equal(matrix[:, 0], [len(t) for t in texts])
    conn.close()
    export_projector(root, tmp_path / "projector", limit=2)
    assert np.loadtxt(tmp_path / "projector/vectors.tsv").shape == (2, 768)
    assert len((tmp_path / "projector/metadata.tsv").read_text().splitlines()) == 3
    with pytest.raises(ValueError, match="changed"):
        embed(root, dtype="float16", encoder_factory=FakeEncoder)


def test_nearest_matches_brute_force_across_blocks():
    rng = np.random.default_rng(41)
    matrix = rng.normal(size=(29, 6)).astype(np.float32)
    vector = rng.normal(size=6).astype(np.float32)
    scores = matrix @ vector / (np.linalg.norm(matrix, axis=1)*np.linalg.norm(vector))
    actual = nearest(matrix, vector, k=7, block_size=4)
    assert [i for _, i in actual] == np.argsort(scores)[-7:][::-1].tolist()
    np.testing.assert_allclose([s for s, _ in actual], np.sort(scores)[-7:][::-1], rtol=1e-5)


def test_single_writer_lock(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with run_lock(tmp_path):
                pass


def read_counts(root):
    with sqlite3.connect(root / "counts.sqlite") as conn:
        return conn.execute("SELECT language,year,text,match_count FROM contexts ORDER BY language,year,text").fetchall()


def test_parallel_matches_serial_and_resumes_existing_run(corpus, tmp_path):
    root, config, manifest = corpus
    retrieve(root, config, manifest)
    parallel = tmp_path / "parallel"
    parallel.mkdir()
    retrieve(parallel, config, manifest, workers=2)
    assert read_counts(parallel) == read_counts(root)
    assert not list((parallel / ".matching-shards").iterdir())
    retrieve(parallel, config, manifest, workers=3)
    assert read_counts(parallel) == read_counts(root)
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    retrieve(resumed, config, manifest, limit=1)
    retrieve(resumed, config, manifest, workers=2)
    assert read_counts(resumed) == read_counts(root)


def test_verified_filtered_checkpoint_survives_import_failure(corpus, monkeypatch):
    root, config, manifest = corpus
    import finance_sentiment.corpus as module
    actual_import = module.import_shard
    def fail(*args):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(module, "import_shard", fail)
    with pytest.raises(OSError, match="disk failure"):
        retrieve(root, config, manifest, workers=2)
    import multiprocessing
    assert not multiprocessing.active_children()
    ready = list((root / ".matching-shards").glob("*.ready.json"))
    assert ready
    # Remove the original of a fully verified shard: restart must reuse its filtered checkpoint.
    for shard in manifest["shards"]:
        _, receipt = shard_paths(root, shard)
        if receipt.exists():
            from urllib.parse import urlparse, unquote
            Path(unquote(urlparse(shard["url"]).path)).unlink()
    monkeypatch.setattr(module, "import_shard", actual_import)
    retrieve(root, config, manifest, workers=2)
    assert json.loads((root / "stats.json").read_text())["total_match_count"] == 27


def test_parallel_bad_checksum_does_not_commit_bad_shard(corpus):
    root, config, manifest = corpus
    manifest["shards"][1]["md5"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        retrieve(root, config, manifest, retries=1, workers=2)
    with sqlite3.connect(root / "counts.sqlite") as conn:
        assert not conn.execute("SELECT 1 FROM sources WHERE name=?", (manifest["shards"][1]["name"],)).fetchone()
    # The first shard can be committed or awaiting import, depending on completion order.
    assert all(row[3] < 10 for row in read_counts(root))


def test_multitarget_filter_matches_union(corpus):
    root, config, manifest = corpus
    config["target_substrings"] = ["FINANC", "navigation"]
    retrieve(root, config, manifest, workers=2)
    assert json.loads((root / "stats.json").read_text())["total_match_count"] == 126


def crash_worker(*args):
    import os
    os._exit(9)


def test_abrupt_worker_exit_fails_visibly_instead_of_hanging(corpus, monkeypatch):
    root, config, manifest = corpus
    monkeypatch.setattr("finance_sentiment.corpus.prepare_worker", crash_worker)
    with pytest.raises(RuntimeError, match="Worker exited"):
        retrieve(root, config, manifest, workers=2)
    import multiprocessing
    assert not multiprocessing.active_children()


def test_interrupt_import_terminates_workers(corpus, monkeypatch):
    root, config, manifest = corpus
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr("finance_sentiment.corpus.import_shard", interrupt)
    with pytest.raises(KeyboardInterrupt):
        retrieve(root, config, manifest, workers=2)
    import multiprocessing
    assert not multiprocessing.active_children()

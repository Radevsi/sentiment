# French retrieval and embedding on a cluster

The first deliverable is a portable corpus of finance contexts, annual counts,
and reusable BERT vectors. No sentiment anchors, W&B account, cloud bucket, or
HTTP server are required. All corpus retrieval and BERT execution happen on
the cluster. The local repository contains code and small source manifests only.

## First: inspect the cluster

After pushing this repository, SSH to the cluster, clone/pull it, and run from
the repository directory (Python standard library only):

```bash
python3 scripts/cluster/inspect.py
# If you already know a project path, include it:
python3 scripts/cluster/inspect.py /actual/project/path
```

Share the output to determine the next commands. No administrator access is
needed. Missing commands are expected. The script does not recursively scan
directories, download data, submit jobs, or read credentials. `df` reports free
space on the filesystem, **not your user or lab quota**. Empty `quota` output
does not establish that you have unlimited space. GPFS/Lustre/project quotas
may require site-specific commands or an administrator's answer.

We need these facts before a long run:

1. The approved persistent project/work directory and your user/group byte
   **and inode quotas**, including current usage.
2. Whether scratch is purged, its retention period, whether it survives jobs,
   and whether the same path is visible from login, transfer, CPU and GPU nodes.
3. Scheduler (Slurm/PBS/other, or no scheduler), CPU and GPU queues, account/QOS,
   wall-time limits, permitted job memory, GPU type/VRAM, and Python/CUDA modules.
4. Where outbound HTTPS to Google Storage and Hugging Face is permitted. Some
   compute nodes cannot access the internet; retrieval may need a transfer node.
5. Where long CPU/network jobs are permitted. An SSH login node is often only
   for editing/submitting jobs. `tmux` does not allocate compute resources.
6. Whether the chosen filesystem supports reliable POSIX file locks and SQLite
   rollback journals. Use one writer. Do not use SQLite WAL on a network mount.

If the inspection cannot reveal quotas/policies, ask the lab: “Which persistent
directory can I use for a personal dataset, what are my user/project quotas,
does scratch expire, and which node or queue allows long HTTPS downloads and
GPU inference?” No need to provide anyone's password or SSH private key.

In Slurm, a **partition is a job queue / group of compute nodes**, not the folder
where files are stored. Choosing a GPU partition does not grant storage quota.
See the [Slurm quick start](https://slurm.schedmd.com/quickstart.html).

## Source size and scope

The checked-in manifests were read from Google's public object metadata on
2026-09-16. They include immutable object generations, compressed lengths, and
MD5 hashes. No corpus shard was downloaded to obtain this inventory.

| French 2012 five-grams | Files | Compressed bytes | Decimal GB | GiB |
| --- | ---: | ---: | ---: | ---: |
| All shards (cluster default) | 678 | 55,345,512,134 | 55.35 | 51.54 |
| `_.gz` selection used by existing reproduction | 35 | 2,286,745,612 | 2.29 | 2.13 |

These are five-gram files only, not the combined 1–5-gram corpus, book text, or
the size of the finance subset. No reliable finance-only count is available
before scanning. `status` reports distinct cleaned contexts, context-year rows,
matching source records, and summed occurrence counts separately. Those are
different quantities. The scan filters years to **1870–2009 inclusive**.

The source of the inventory is the
[Google Storage object-list API](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-fre-all-5gram-20120701-&maxResults=1000).
The metadata can be refreshed without downloading n-grams:

```bash
finance-pipeline manifest --language fre --selection all --output /chosen/path/fre-manifest.json
```

Use a new run directory if switching manifests, terms, years, or model settings.
`paper-parts` filters filenames ending in `_.gz`; select the checked-in
`manifests/fre-2012-5-paper-parts.json` for the existing reproduction's scope.
**The all-shard result is a broader sample, not an exact reproduction.** Google's
exports include syntactic/POS variants; summing their records is not the same
thing as counting distinct original passages. The existing cleanup and substring
rules are preserved, including their limitations. Compare annual results only
with an explicit record of which source selection was used.

Google's [Ngram documentation](https://books.google.com/ngrams/info) describes
these downloadable files. We stream the bulk exports rather than query the
Viewer API. To find `financ` in any position we must scan all selected shards;
scanning only shards starting with `f` would miss contexts. Google five-grams
are short sequences of tokens, not necessarily complete sentences, and do not
provide the original book passages.

## Storage arrangement

Keep the Git checkout small. Prefer an approved persistent project/work path
outside the checkout for results. Git ignore rules protect against accidental
commits; they do not move files or change quotas. For this server the repository
is `~/projects/sentiment` and the data root is `~/projects/sentiment_data`.
Use a separate run directory per language and experiment:

```text
~/projects/sentiment/                         # Repository and .venv only
~/projects/sentiment_data/
  runs/
    fre/finance-2012-all-v1/                  # French counts, vectors, checkpoints
    ger/finance-2012-all-v1/                  # Future German run
    spa/finance-2012-all-v1/                  # Future Spanish run
  cache/huggingface/                         # Shared model/tokenizer cache
  logs/fre/                                 # Logs grouped by language
```

`FINANCE_DATA` is the shared data root; `FINANCE_RUN` is the one run currently
being processed. The scripts already accept an arbitrary `--run-dir` and check
that config and manifest languages agree. Each additional language needs its
own config (including appropriate target stems) and 2012 source manifest;
changing the folder name alone does not change the language being retrieved.
The German/Spanish folders above are examples, not preconfigured experiments.
If an earlier run already exists at `fre-financ-2012-all`, keep using that path
or move its entire directory while all jobs are stopped; do not start a second
copy of the same retrieval just to adopt the new naming scheme.

Hugging Face hosts the pretrained multilingual BERT weights, tokenizer and
model configuration used by the embedding stage. The n-gram data comes from
Google Storage. `HF_HOME` selects where the model downloads are cached; setting
it does not itself download anything, and `fetch` never uses that cache.
All languages using the same pinned multilingual model reuse the same cached
files. Embedding runs on the server, without sending contexts to a hosted model
API. See the [Hugging Face cache guide](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

The pipeline streams compressed bytes and decompresses in memory, writes only
matching records into a temporary compressed file, commits those matches to
SQLite, then deletes that temporary file. **It does not retain the 55.35 GB raw
corpus or a fully decompressed copy.** It still transfers/processes those bytes.
Adding a new target later requires rescanning; run multiple terms together in
one config if their union is desired. That config's corpus is the union, not
separate per-term statistics. For separate term corpora use separate runs.

Allow space for the database, its indexes/journal, one matching-shard temporary
file per worker (plus any verified checkpoints from interrupted runs), model
cache, software environment, and vectors. Exact filtered storage
cannot be promised before retrieval. Largest compressed source shard is
9.75 GB, but it is streamed rather than staged on disk. A one-shard pilot checks
operation, not a statistically valid estimate of this alphabetically partitioned
corpus. `--limit 1` intentionally prevents the embedding stage until retrieval
is complete.

768 float32 values take **3,072 bytes per distinct context** (3.072 GB per million),
plus text/count metadata. Float16 storage halves vector size but rounds values;
float32 is the fidelity default. Inference uses model float32 in both cases.
SQLite and `.npy` prioritize checkpoint safety and easy access rather than maximum
compression. Once jobs finish, optionally compress a copy of the complete run
for transport; do not compress/delete live working files. Keep a durable copy
before relying on purged scratch. No automatic deletion of final results occurs.

## Installation and preparation

Use Python 3.10+ and the site's recommended environment/modules. From the repo:

```bash
cd "$HOME/projects/sentiment"
bash scripts/cluster/setup.sh .venv
source .venv/bin/activate
```

Setup isolates the environment from the server's existing Python packages and
checks dependencies with `pip check`. If an older setup printed missing
dependencies for unrelated packages such as `spektral` or `scikit-learn`, rerun
the updated setup command at the same `.venv` path. This disables inherited
system packages without deleting the environment or touching the data directory.
Developer tests are optional: install `pip install -e '.[dev]'` if needed.

Retrieval needs only lightweight dependencies. Before embedding, use the
cluster-approved CUDA-compatible PyTorch installation, then:

```bash
python -m pip install --progress-bar on -e '.[ml]'
export FINANCE_DATA="$HOME/projects/sentiment_data"
export HF_HOME="$FINANCE_DATA/cache/huggingface"
mkdir -p "$FINANCE_DATA/logs"
finance-pipeline prepare-model --config configs/french_finance_cluster.json
python -m pip freeze > "$FINANCE_DATA/logs/environment.txt"
git rev-parse HEAD > "$FINANCE_DATA/logs/code-commit.txt"
```

Create the logs directory first. `prepare-model` downloads weights/tokenizer
using native Hugging Face progress bars on an internet-enabled node. It uses
CPU RAM; it does not embed the corpus. Use the same `HF_HOME` on GPU nodes.
`--offline` makes embedding require cached files. On a GPU allocation verify
`python -c 'import torch; print(torch.cuda.is_available())'` prints `True`.
We cannot choose the correct CUDA build until the cluster details are known.
See [Hugging Face cache/offline documentation](https://huggingface.co/docs/transformers/installation).

## Run and resume

Set the shared root and current language/run. These settings select French;
set matching config and manifest paths when adding another language:

```bash
export FINANCE_REPO="$HOME/projects/sentiment"
export FINANCE_DATA="$HOME/projects/sentiment_data"
export FINANCE_LANGUAGE=fre
export FINANCE_VENV="$FINANCE_REPO/.venv"
export FINANCE_RUN="$FINANCE_DATA/runs/$FINANCE_LANGUAGE/finance-2012-all-v1"
export FINANCE_CONFIG="$FINANCE_REPO/configs/french_finance_cluster.json"
export FINANCE_MANIFEST="$FINANCE_REPO/manifests/fre-2012-5-all.json"
export HF_HOME="$FINANCE_DATA/cache/huggingface"
mkdir -p "$FINANCE_DATA/logs/$FINANCE_LANGUAGE"
cd "$FINANCE_REPO"
source "$FINANCE_VENV/bin/activate"
```

On a host where long retrieval is permitted:

```bash
finance-pipeline fetch \
  --config "$FINANCE_CONFIG" \
  --manifest "$FINANCE_MANIFEST" \
  --run-dir "$FINANCE_RUN" \
  --workers 4

# On a GPU allocation, after retrieval completes:
finance-pipeline embed --run-dir "$FINANCE_RUN" --device cuda --batch-size 64 --offline

finance-pipeline status --run-dir "$FINANCE_RUN"
```

`--workers 4` processes up to four independent shards on separate CPU cores.
The default remains 1. Use no more workers than your available/allocated CPUs;
four is a starting point for the shared GPU server, not a promise of 4x speed.
Only the coordinator writes SQLite, and at most that many filter jobs are in
flight, so slow database imports do not accumulate a full-corpus backlog.
Small shards incur process-start overhead; network/storage limits and the last
large shard can also limit scaling. Each worker uses a separate temporary file.

The reported MB/s measures **download + decompression + filtering** in compressed
bytes per second, not raw network bandwidth. If a single process is near 100%
CPU in `top`, one CPU core is saturated even if the rest of the server is idle.
Progress lines identify the shard, so rates from different workers can be
distinguished; a worker can still show 3.5 MB/s while several run concurrently.

To upgrade an existing serial run, push/pull the updated code, stop the old
process with Ctrl-C, and repeat the same fetch command and run directory with
`--workers 4`. Ideally stop just after a shard completes to avoid repeating its
scan. Do not launch a second fetch before the first exits. Completed source
receipts remain compatible; workers are an execution setting, not a new corpus.
No new dependency installation is needed for this parallel-processing change.

If GPU memory is insufficient, repeat `embed` with batch size 32 or 16. Settings
that affect vector identity/dtype and library versions cannot change mid-run.
Batch size and device can change; small floating-point differences are possible
across hardware. Record the environment alongside results. The pinned model is
`google-bert/bert-base-multilingual-cased` at commit
`3f076fdb1ab68d5b2880cb87a0886f315b8146f8`.

**Slurm:** submit the provided scripts with the approved account, queue,
wall-time, memory and GPU options. Values below are placeholders; resource sizes
are starting requests, not measured requirements:

```bash
sbatch --account=ACCOUNT --partition=CPU_QUEUE --time=24:00:00 --mem=8G \
  --output="$FINANCE_DATA/logs/$FINANCE_LANGUAGE/fetch-%j.log" scripts/cluster/fetch.sbatch

# After fetch succeeds (or use --dependency=afterok:FETCH_JOB_ID):
sbatch --account=ACCOUNT --partition=GPU_QUEUE --gres=gpu:1 \
  --time=08:00:00 --mem=16G --output="$FINANCE_DATA/logs/$FINANCE_LANGUAGE/embed-%j.log" \
  scripts/cluster/embed.sbatch
```

Some sites use `--gpus=1` or a named GPU type instead of `--gres=gpu:1`, require
module loads in the job, or disallow outbound HTTPS on CPU queues. Adapt after
inspection. Batch jobs survive SSH logout; tmux is not needed for them. A job
time limit still applies. Resubmit the same command to resume; automatic
resubmission/requeue is intentionally not assumed.

The fetch batch script uses `SLURM_CPUS_PER_TASK` workers by default. You can
override it with `FINANCE_WORKERS`, within the allocated CPU count.

**Unscheduled server or approved long-running host:** `tmux new -s finance`,
activate the environment, and run the commands. Detach with Ctrl-b then d;
reconnect with `tmux attach -t finance`. A tmux session does not move work from
a login node to a compute node, survive a reboot, or override allocation limits.

Each shard is verified using compressed length, MD5 and gzip CRC before import.
The counts and completion receipt commit in a single SQLite transaction, so
rerunning cannot double-count completed shards. A broken stream retries up to
three times with backoff; it then exits visibly. An unfinished shard restarts
from its beginning (gzip does not provide arbitrary decompressed seek/resume).
Completed shards are skipped. Embeddings are flushed to disk before saving the
batch checkpoint. A crash can redo the last unsaved batch safely. Do not run the
legacy `finance-extract` command on these databases or edit them by hand.

With the updated fetcher, fully filtered and verified shards awaiting import
have durable receipts under `.matching-shards/`. Those results are reused after
an interruption without downloading the source again; they are deleted after
their database transaction commits. Partially filtered shards restart from the
beginning. Ctrl-C or an import failure stops the worker processes before the
coordinator releases its run lock. Abrupt worker failures are reported rather
than leaving the coordinator waiting indefinitely.

Logs report timestamps, named stages, stage/overall duration, compressed-byte
throughput and current-shard ETA, or contexts/sec and embedding ETA. There is
no speculative total retrieval ETA before measuring the actual scan rate.

## Outputs and later reuse

Copy the complete run directory only after jobs finish:

```bash
# Run on your own machine; replace user/host and both paths:
rsync -avP user@cluster:projects/sentiment_data/runs/fre/finance-2012-all-v1/ /local/destination/fre/finance-2012-all-v1/
```

| File | Contents |
| --- | --- |
| `run.json` | Exact config and source manifest, including generations/checksums |
| `counts.sqlite` | Annual counts, source receipts, vector row-to-text map, embedding checkpoint |
| `embeddings.npy` | Row-aligned 768-dimensional vectors, memory-mappable |
| `embeddings.json` | Shape, dtype, model revision, pooling/preprocessing, package versions |
| `stats.json` | Summary from the most recent completed stage (`status` reads live receipts) |

Every distinct cleaned five-gram is encoded once, independent of year. Yearly
`match_count` stays in SQLite, without expanding repeated occurrences. We retain
the existing `[CLS]` sum of the last four hidden layers, lowercasing, type ID 1,
and length limit 64. This follows the existing port of the authors' script;
modern Transformers is not a demonstrated bit-exact replication of their older
software. Positive/negative anchors and annual sentiment scoring can be added
after this stage. Existing `finance-score` remains available but currently
re-encodes contexts instead of consuming the new cache.

Later, on a CPU machine with the ML extra and cached model:

```bash
finance-query --run-dir /local/copied/run --text 'la finance aide notre société' --k 10
```

The query uses identical cleanup/model/pooling and blockwise cosine search, so
it does not load all vectors into RAM. CPU inference for a single short query
is feasible; benchmark latency and keep the model loaded in a future service.
Exact search scans every vector and can dominate latency for large corpora; an
approximate nearest-neighbor index is a later optimization. The library exposes
`finance_sentiment.embeddings.open_vectors` and `finance_sentiment.search.query`
as a portable Python API. No public web endpoint is deployed in this v1.

For exploratory visualization without W&B:

```bash
finance-pipeline projector --run-dir "$FINANCE_RUN" \
  --output "$FINANCE_DATA/projector/$FINANCE_LANGUAGE/finance-2012-all-v1" --limit 5000
```

This writes reproducibly sampled `vectors.tsv` and `metadata.tsv` suitable for
the [TensorFlow Embedding Projector](https://projector.tensorflow.org/) or a local
viewer. It does not upload them. A two/three-dimensional projection loses
information; use full-dimensional cosine similarity for actual comparisons.
BERT's paper-specific representation is not a guarantee of general semantic
search quality, and finance-related neighbors are not sentiment labels.

## Validation boundary

Local tests use tiny synthetic gzip shards and a deterministic fake encoder to
verify filtering, counts, integrity rejection, atomic rollback, duplicate-free
resume, vector checkpoints, row mapping, and cosine retrieval. They do not
download the corpus or model. Real BERT/CUDA throughput, cluster filesystem
semantics, quotas, and comparison with the published annual series still need
validation on the selected cluster.

# French finance contexts, embeddings, and sentiment

## New: cluster retrieval and reusable vectors

For the current download-and-embedding workflow, start with
**[the cluster walkthrough](docs/cluster.md)**. It includes SSH/tmux and Slurm
instructions, storage/quota discovery, source sizes, resumable streaming,
portable BERT vectors, CPU nearest-neighbor queries, and visualizer export.

First run on the remote cluster, from this checkout:

```bash
python3 scripts/cluster/inspect.py
```

This only inspects the machine. Share its output before choosing storage and
job settings. There is no local corpus/model download needed to prepare code.

The checked-in Google metadata inventories **678 French 2012 five-gram files
(55.35 GB compressed)**. The new default scans them all but stores only records
containing `financ` from 1870–2009. The existing reproduction selection is
**35 files (2.29 GB)**, selectable separately. No finance-only context count is
known until retrieval. The full-shard sample is a methodological extension.

The new entry points are `finance-pipeline` and `finance-query`;
`configs/french_finance_cluster.json` pins model/preprocessing without needing
sentiment anchors. For another language or target, copy the config, create the
corresponding manifest, and use a separate run directory. Retrieval installs
without PyTorch; embedding/query/scoring require the `ml` extra.
Pass `--workers 4` to `finance-pipeline fetch` to scan independent shards on
four CPU processes with one database writer. Existing serial runs resume in
the same directory; stop the old process before restarting with more workers.
For four-GPU embedding use `finance-pipeline embed --devices cuda:0 cuda:1 cuda:2 cuda:3`
with the same `--run-dir`; `--batch-size` is per GPU. The coordinator saves one
aligned vector file and checkpoints completed batches, including out-of-order
results. The [cluster walkthrough](docs/cluster.md) covers GPU selection and resume.

## Original annual-sentiment workflow

The remainder documents the older full-raw-download/scalar-scoring workflow.
Use the cluster workflow above for safe interrupted retrieval and saved vectors.
The legacy extraction command is additive: do not rerun it against a populated
database, and do not mix it with the new tracked run directories.

This repository reproduces the French portion of Jha, Liu, and Manela,
*Does Finance Benefit Society? A Language Embedding Approach*. It does **not**
measure religion sentiment.

The v1 deliberately produces a narrow, auditable result:

1. Read the 2012 French Google Books 5-gram files.
2. Retain 1870-2009 records containing the French finance stem `financ`.
3. Sum Google Books `match_count` for the same cleaned 5-gram and year.
4. Encode each distinct 5-gram with `bert-base-multilingual-cased`.
5. Represent a sentence by the `[CLS]` token with its last four hidden layers
   summed, matching the authors' public scoring script.
6. Construct the finance-positive-minus-negative direction from the authors'
   five French sentence pairs.
7. Compute each 5-gram's cosine similarity to that direction and form the
   frequency-weighted annual mean.

This v1 targets the paper's French finance result and measurement pipeline.

## What the v1 outputs

- `output/french_finance_annual.csv`: one raw cosine score and one published-
  scale standardized score per year.
- `output/french_finance_extremes.csv`: highest- and lowest-scoring
  contexts for face-validity review.
- `data/work/french_finance.sqlite`: cleaned counts and cached context
  scores, so interrupted runs can resume.

The raw 768-dimensional vectors are not retained because the annual index only
needs each context's scalar cosine score.

## Setup

Create a Python environment with CUDA-enabled PyTorch appropriate for the
machine, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev,download,ml]'
```

Place the **2012** French 5-gram `.gz` shards under `data/raw/fre/`. To use the
same downloader mechanism as the authors' archive, first inspect the manifest:

```bash
python scripts/download_google_ngrams.py --language fre --ngram-length 5 --dry-run
```

By default the downloader preserves the authors' `partsonly=true` choice for
French: only 2012 shard filenames ending in `_.gz` are selected. This unusual
subset is necessary to target their published sample. Pass `--all-shards` only
for a deliberate robustness extension; it will not be an exact reproduction.

Then download only after confirming the storage requirement:

```bash
python scripts/download_google_ngrams.py --language fre --ngram-length 5 --output data/raw/fre
```

## Run

```bash
finance-extract \
  --config configs/french_finance.json \
  --database data/work/french_finance.sqlite \
  data/raw/fre/*.gz

finance-score \
  --config configs/french_finance.json \
  --database data/work/french_finance.sqlite \
  --annual-output output/french_finance_annual.csv \
  --extremes-output output/french_finance_extremes.csv \
  --device cuda \
  --batch-size 64
```

Download the authors' small published reference table and compare:

```bash
python scripts/download_reference_data.py
python scripts/compare_french_reference.py \
  --output output/french_finance_annual.csv
```

The comparison reports Pearson correlation, mean absolute error, and maximum
absolute error for both the raw and standardized French series.

If 6 GB of VRAM is insufficient at batch size 64, use 32 or 16. The scoring
step embeds distinct contexts only; annual frequency is applied afterward.

## Fidelity and expected differences

The configuration contains the French finance stem and the five French anchor
pairs from the authors' public script. Compare the resulting annual shape to
their released `sentiment.dta`. Exact levels can differ because their archive uses the retired
`pytorch_pretrained_bert` implementation and serialized model objects, while
this project uses the current Transformers API with the same model family.

The released `sentiment.dta` contains the standardized final index. The authors'
`f.tab` shows that it is a global z-score across the 1,068 available
language-year raw scores. The configuration records that published mean and
sample standard deviation, so a French-only run can produce both the raw cosine
score and the paper-scale standardized score without recomputing all languages.

## Minimum validation gate

Do not treat the French finance reproduction as successful until all of the
following hold:

- the output contains 140 annual rows for 1870-2009;
- the raw annual series broadly reproduces the published French series;
- the positive and negative extremes are semantically sensible;
- the standardized output closely matches the published French values.

## Source-method notes

The authors' public replication archive contains extraction and scoring scripts,
serialized BERT/tokenizer objects, a small 1870 sample, and final annual data.
Their implementation expands annual match counts into repeated lines and then
counts duplicates. This v1 sums `match_count` directly, which is algebraically
equivalent but far less expensive in disk and memory.

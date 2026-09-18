# All-context French embedding map

This viewer projects **all 219,584 existing French contexts**, with no sampling,
new corpus retrieval, BERT inference, external upload, account or CDN. It reads
`embeddings.npy` using `embeddings.open_vectors`, and joins the persisted
`vector_rows(row_id,text)` mapping to `contexts`. Matrix offset `i`, coordinate
pair `i` and `row_id=i` are identical. Text is the existing cleaned five-gram.
Counts are summed across the available years, with distinct year count and first/
last year. Occurrences are not expanded into additional points.

The original sum-of-last-four-CLS BERT vectors remain unchanged. UMAP uses cosine
distance directly on all 768 dimensions, approximate nearest-neighbor search,
a sparse graph and `low_memory=True`; it does not construct a dense N×N distance
matrix. The viewer uses native WebGL `POINTS`, with every coordinate in one GPU
buffer. Metadata is requested only when selected or searched. Everything is
served from this repository, with no browser dependencies fetched externally.

## Compute remotely, copy the map, view locally

**The remote machine only computes files. It does not run a web server.**
Copy just `map.json`, `points.f32`, and `metadata.sqlite` to your computer, then
run the viewer locally. You do not need the original embeddings, corpus database,
BERT model or Hugging Face cache on your computer for this workflow.

Run these steps in order. Commands labelled **LOCAL** run on your computer;
commands labelled **REMOTE** run inside your SSH session. Replace
`YOUR_USER@YOUR_SERVER` with the SSH destination you normally use.

### 1. LOCAL — commit and push the new code

The implementation was created in a detached worktree and was not committed or
pushed. These commands create a feature branch from that worktree. Run this step
once; do not repeat `git switch -c` if you have already created the branch.

```bash
cd "/Users/bg/.codex/worktrees/825d/Does Finance Benefit Society"
git switch -c codex/embedding-map
git add README.md pyproject.toml docs/embedding-map.md \
  src/finance_sentiment/map.py src/finance_sentiment/map_static \
  tests/test_map.py tests/test_map_viewer.mjs
git commit -m "Add portable all-context embedding map"
git push -u origin codex/embedding-map
```

Only code, documentation and tests go into Git. Map files stay outside the repo.
If you have already integrated these changes into another branch, use that branch
for the push and remote checkout instead.

### 2. REMOTE — get the code and install the map dependency

Connect from your computer with `ssh YOUR_USER@YOUR_SERVER`, then run:

```bash
cd ~/projects/sentiment
git fetch origin
git switch --track origin/codex/embedding-map
map_install_started=$SECONDS
printf '[%s] Step 2: install projection dependency\n' "$(date -u +%FT%TZ)"
.venv/bin/python -I -m pip install --progress-bar on -e '.[map]' &&
  printf '[%s] Step 2 complete; elapsed %ss\n' "$(date -u +%FT%TZ)" "$((SECONDS-map_install_started))"
```

The `git switch --track` command is for the first checkout of the new branch.
On subsequent visits use `git switch codex/embedding-map` and `git pull --ff-only`.
If Git reports local changes or installation fails, stop and resolve that error
before continuing. The original completed run remains at its existing path.
UMAP is pinned to 0.5.8. No new Torch/Transformers installation is needed.

### 3. REMOTE — compute every point from the saved vectors

Run in your usual persistent CPU session/allocation. This can take hours; keep
the session alive. Do not run retrieval or embedding again.

```bash
cd ~/projects/sentiment
export HF_HOME="$HOME/projects/sentiment_data/cache/huggingface"
export FINANCE_RUN="$HOME/projects/sentiment_data/runs/fre/finance-2012-all-v1"
export FINANCE_MAP="$HOME/projects/sentiment_data/maps/fre/finance-2012-all-v1-umap-v1"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$FINANCE_MAP"
set -o pipefail
.venv/bin/python -I -u -m finance_sentiment.map build \
  --run-dir "$FINANCE_RUN" --output "$FINANCE_MAP" \
  --expected-rows 219584 --neighbors 30 --min-dist 0.1 --epochs 500 --seed 42 \
  2>&1 | tee "$FINANCE_MAP/build-$(date +%Y%m%d-%H%M%S).log"
```

Wait for `Ready: 219,584 / 219,584 points` and a successful exit. This reads the
saved embeddings and preserves the original run. Stage logs include timestamps,
elapsed time, metadata throughput/ETA, native UMAP progress and a 30-second
heartbeat. First-run Numba compilation may take time. If interrupted, rerun this
same command; a completed projection is reused, but an unfinished UMAP fit must
restart. **Do not run `serve` on the remote machine.**

### 4. LOCAL — copy the three map files

Open a new terminal on your computer (not inside SSH):

```bash
export MAP_REMOTE='YOUR_USER@YOUR_SERVER'
export LOCAL_MAP="$HOME/projects/sentiment_data/maps/fre/finance-2012-all-v1-umap-v1"
mkdir -p "$LOCAL_MAP"
rsync -avP \
  --include='map.json' --include='points.f32' --include='metadata.sqlite' \
  --exclude='*' \
  "$MAP_REMOTE:projects/sentiment_data/maps/fre/finance-2012-all-v1-umap-v1/" \
  "$LOCAL_MAP/"
ls -lh "$LOCAL_MAP/map.json" "$LOCAL_MAP/points.f32" "$LOCAL_MAP/metadata.sqlite"
```

The path after the colon is relative to your remote home directory. Use the same
SSH alias you normally use if your connection requires a jump host or custom port.
`rsync -P` shows progress and retains partial transfers; rerun this command if
interrupted. All three named files must be present before the next step. The
coordinates are only 1.68 MiB; the text/count database is additional. The local
viewer verifies the transferred files against the hashes in `map.json`.

### 5. LOCAL — install the lightweight viewer

Use the same local code worktree from step 1. Python 3.10 or newer is required.

```bash
cd "/Users/bg/.codex/worktrees/825d/Does Finance Benefit Society"
map_local_started=$SECONDS
printf '[%s] Step 5: create local viewer environment\n' "$(date -u +%FT%TZ)"
python3 -m venv .venv &&
  .venv/bin/python -I -m pip install --progress-bar on -e . &&
  printf '[%s] Step 5 complete; total elapsed %ss\n' "$(date -u +%FT%TZ)" "$((SECONDS-map_local_started))"
```

This installs only the base viewer dependency (NumPy), not UMAP, BERT or model
weights. Stop if the command reports an error. If you already have a working
`.venv` in this checkout, skip its creation and run the pip command only.

### 6. LOCAL — start the viewer and open it

```bash
cd "/Users/bg/.codex/worktrees/825d/Does Finance Benefit Society"
.venv/bin/python -I -u -m finance_sentiment.map serve \
  --output "$HOME/projects/sentiment_data/maps/fre/finance-2012-all-v1-umap-v1" \
  --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your local browser.
Leave this local terminal running. Ctrl-C stops the viewer; rerun step 6 whenever
you want to reopen it. If port 8765 is occupied, use `--port 8766` and open
`http://127.0.0.1:8766`. No SSH tunnel is needed, and the remote machine can be
disconnected after the transfer. The viewer binds to your computer's loopback
interface only. To move it to another computer, copy the same three files and
install this repository's base package there.

## Viewer controls and interpretation

- Drag to pan, scroll to zoom about the cursor, and use **Fit all points** to reset.
- Click to reveal cleaned text, exact row ID, summed `match_count`, years present,
  and first/last year. The selected point turns gold. Point size is adjustable.
- Dense overlaps show up to the nearest 20 row IDs within 10 screen pixels;
  zoom further or use the exact row lookup to select any row, including overlaps.
- **Go to exact row_id** selects and centers any context. **Find cleaned text**
  performs a literal, case-sensitive substring search (first 50 matches in row
  order); type lowercase French text including its accents. This is not a BERT
  query or a nearest-neighbor search.

2D proximity is approximate. Distances between islands, local density, and gaps
can be projection artifacts; they do not establish clusters, causal relations
or sentiment. Different UMAP parameters/seeds can change the visual arrangement.
No sentiment labels or inferred clusters are added. The coordinate axes have no
substantive units, and cosine similarity is not a sentiment score.

Typed query projection is not included: this version saves coordinates, not a
serialized UMAP transform model. For optional semantic queries, run the existing CLI on the remote machine
where the BERT cache and full vectors already live. Its exact, blockwise CPU
search embeds one query with the run's pinned BERT model, pooling and preprocessing, then ranks
neighbors in the **original 768-dimensional vector space**:

```bash
cd ~/projects/sentiment
export HF_HOME="$HOME/projects/sentiment_data/cache/huggingface"
.venv/bin/python -I -m finance_sentiment.search \
  --run-dir "$HOME/projects/sentiment_data/runs/fre/finance-2012-all-v1" \
  --text 'la finance aide notre société' --k 10 --device cpu --offline
```

This requires the existing ML dependencies and cached pinned model. `--offline`
prevents model downloads. Paste a returned `row_id` into the viewer to locate a
neighbor. It never re-embeds the corpus. No sentence-transformer is substituted.

## Artifacts, validation, and resume

The artifact saves the seed, exact projection settings, run identity hash/config,
embedding metadata/model revision, SHA-256 of the saved vector file, hash of the
ordered row/text/aggregate-count tuples, and numeric package versions.

| File | Purpose |
| --- | --- |
| `request.json` | Input/settings/environment identity; refuses incompatible cache reuse |
| `metadata.sqlite` | Compact read-only row/text/count lookup, separate from corpus DB |
| `points.f32` | Little-endian float32 `(x,y)` pairs in exact row ID order |
| `projection.json` | Completed UMAP checkpoint with identity and coordinate checksum |
| `map.json` | Final completion marker, bounds, identity, checksums and build duration |

For 219,584 points, the coordinate payload is **1,756,672 bytes (~1.68 MiB)**;
raw 768D vectors and the full text database are not sent to the browser.
The metadata database is copied to your computer; its size depends on context text.
The map can be served independently of the run after a successful build; serving
needs only the base NumPy dependency and packaged viewer files, not UMAP or BERT.

Rerun exactly the same build command after an interruption. Each invocation
revalidates the source vectors/mapping/counts and rebuilds the cheap metadata
snapshot, then reuses a verified completed projection or map. A failure during
UMAP restarts the **whole UMAP fit**; there is no per-epoch or neighbor-graph
checkpoint. No BERT work is repeated. A failure after `projection.json` was
committed resumes final publication. Partial files do not count as completion.
One builder per output is enforced by a POSIX file lock. Changed source values,
parameters or numeric package versions are refused; use a new output directory
for another projection. Do not edit the original run to bypass validation.
Corrupt completed artifacts are rejected at startup; recover from backup or
build to a new directory. Keep the source run idle while exporting.

## Resources and validation limits

The full source payload is 674,562,048 bytes (~643 MiB), memory mapped. UMAP still
needs working copies, an approximate neighbor index and a sparse graph. Plan for
several GiB of RAM; an initial **16–32 GiB CPU allocation** is a conservative
starting budget, not a measured peak or upper bound. No GPU is used, so the eight
RTX A6000s are unnecessary for this stage. Fixed seed and one worker favor
reproducibility over throughput. On 219k × 768 values, fitting can take hours;
there is no server benchmark yet. Exact timings depend on CPU, memory bandwidth
and filesystem. The initial hash/metadata pass reads all saved vectors and
context-year counts even when using a cached map. NumPy/Numba/software and platform
differences may prevent bit-identical fits; recorded versions plus checksummed
saved coordinates identify the artifact actually viewed.

Local tests use synthetic vectors and SQLite rows. They cover mapping order,
counts, all-row output, source immutability, incomplete/corrupt input rejection,
cache invalidation, restart after interruption, HTTP endpoints and viewer
coordinate/picking/zoom behavior (including row 219583). A real UMAP smoke test
runs only if the optional dependency is installed. It was run locally, and a
browser check with 219,584 synthetic points verified WebGL rendering, final-row
lookup, pan, zoom and click selection. These synthetic coordinates are not a
projection of the French corpus. Real French projection quality,
full-run RAM/runtime, remote filesystem behavior and browser GPU performance must
be validated on the server. No real remote files are accessible locally.

```bash
.venv/bin/python -I -m pip install --progress-bar on -e '.[map,dev]'
.venv/bin/python -I -m pytest tests/test_map.py -q
node --test tests/test_map_viewer.mjs
```

Primary documentation checked for the design:
[UMAP API](https://umap-learn.readthedocs.io/en/latest/api.html),
[UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html),
[UMAP memory FAQ](https://umap-learn.readthedocs.io/en/latest/faq.html), and
[WebGL drawArrays](https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/drawArrays).

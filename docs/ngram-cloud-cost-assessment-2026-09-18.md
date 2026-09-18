# Ngram cloud feasibility and planning costs

Checked 2026-09-18. USD list prices, excluding tax, credits, institutional discounts and labor. No paid cloud jobs were run. Source sizes come from `ngram-size-comparison-2026-09-18.json`: compressed **five-grams only**, not all ngram lengths. TB = 10^12 bytes; GiB = 2^30 bytes; TiB = 2^40 bytes.

## BigQuery finding

The official [Google Books Ngrams 2020 Marketplace listing](https://console.cloud.google.com/marketplace/product/bigquery-public-data/google-books-ngrams-2020) explicitly describes N=1, single words only. Its published sample queries `bigquery-public-data.google_books_ngrams_2020.eng_1`, with `term` and repeated `years` containing `year` and `term_frequency`. This is useful for word frequency, but cannot supply five-word contexts for contextual embeddings. The listing's coverage description also references older documentation; do not infer precise release coverage solely from its title.

A different, potentially useful route is a BigQuery external table pointing directly at existing Google Cloud Storage source objects. Public bucket metadata reports location US. This does not require a private raw mirror. BigQuery supports gzip CSV external sources, but the 2020 ngram layout is variable-width: a text field followed by tab-separated year,count,volume triples. A pilot could read each line as one STRING using a nonoccurring delimiter and parse it in SQL. This integration has **not** been executed or validated. Permissions, gzip/file limits, encoding, parsing and result equivalence need verification before scaling.

Proposed pilot:

1. Use an existing billing-enabled GCP project with a US BigQuery dataset.
2. Point an external table at one explicit French source shard, not a corpus-wide wildcard.
3. Parse and filter `financ`, retain original ngram text, year, frequency and volume counts, and apply the intended year range.
4. Materialize only matches into a destination table, export compressed results, and transfer those to `~/projects/sentiment_data` on the lab server.
5. Check bytes billed, elapsed time, output bytes, rejected lines and equivalence with the Python parser. Expand to a representative set of shards before projecting corpus totals; one shard is not a reliable estimate of term prevalence.

External-table dry runs can report zero as a lower bound, so a zero dry-run estimate is not evidence of a free scan. Filter selectivity does not eliminate scanning raw gzip files. Repeated searches may repeatedly incur scan costs. A CPU worker streaming the same public objects and writing only matches is a simpler alternative if SQL parsing proves awkward.

References: [external Cloud Storage sources](https://docs.cloud.google.com/bigquery/docs/external-data-cloud-storage), [external-table limitations](https://docs.cloud.google.com/bigquery/docs/external-tables), [DDL options](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language), [gzip CSV constraints](https://docs.cloud.google.com/bigquery/docs/loading-data-cloud-storage-csv).

## Recurring raw-mirror storage

Assumed GCS regional Standard $0.020/GiB-month; S3 Standard US East first-tier $0.023/GB-month, with binary storage units. GCS US multiregion is $0.026/GiB-month, 30% above the regional column. These are one-copy storage costs, without output data, requests, downloads, version history or backups.

| 2020 five-grams | Compressed TB | GCS regional/month | GCS/year | S3/month | S3/year |
|---|---:|---:|---:|---:|---:|
| French | 1.821 | $33.93 | $407 | $39.02 | $468 |
| English | 8.179 | $152.35 | $1,828 | $175.20 | $2,102 |
| All eight languages | 13.395 | $249.50 | $2,994 | $286.93 | $3,443 |

Eight languages means Chinese Simplified, English, French, German, Hebrew, Italian, Russian and Spanish. English fiction, US and GB subsets are excluded to avoid counting overlapping corpora as additional languages.

Filtered-output scenarios, **not predictions of finance data volume**:

| Retained output | GCS regional/month | S3/month |
|---|---:|---:|
| 10 GiB | $0.20 | $0.23 |
| 100 GiB | $2.00 | $2.30 |
| 1 TiB | $20.48 | $23.55 |

GCS Standard internet transfer to the US/Europe starts at $0.12/GiB for the first 10 TiB, before applicable allowances: downloading 100 GiB is approximately $12; 1 TiB approximately $122.88. Every additional download can incur another transfer bill. Requests and temporary result tables add costs; cold tiers add retention and retrieval constraints. No cloud copy means no cloud storage rent, but university quota/recharge rules still apply.

Sources: [GCS pricing](https://cloud.google.com/storage/pricing), [S3 pricing](https://aws.amazon.com/s3/pricing/), [official S3 US East price catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/us-east-1/index.json).

## One-time scan time and CPU rental

The following are sensitivity calculations, not benchmarks. Rate means aggregate **end-to-end compressed source bytes processed per second**, including the effects of network, decompression, filtering and output. The current pipeline is not demonstrated to achieve these rates; high throughput likely needs bounded parallel shard workers.

| Corpus | 5 MB/s | 20 MB/s | 100 MB/s |
|---|---:|---:|---:|
| French | 101.2 h / 4.2 days | 25.3 h | 5.1 h |
| English | 454.4 h / 18.9 days | 113.6 h / 4.7 days | 22.7 h |
| Eight languages | 744.2 h / 31.0 days | 186.0 h / 7.8 days | 37.2 h / 1.6 days |

Illustrative Iowa on-demand e2-standard-16 VM: 16 vCPUs, 64 GiB, $0.53609136/hour. Multiplying those hours by the VM price gives:

| Corpus | At 5 MB/s | At 20 MB/s | At 100 MB/s |
|---|---:|---:|---:|
| French | $54 | $14 | $3 |
| English | $244 | $61 | $12 |
| Eight languages | $399 | $100 | $20 |

These do not assert that this VM can attain 100 MB/s; benchmark first. Disk, output storage/transfer, retries and setup time are additional. Multiple machines multiply hourly price but can shorten wall time. Leaving this VM running 730 hours costs about $391/month before extras. Lab execution incurs no external VM rental, assuming no institutional recharge. Google source hosting policy and network placement must be checked when selecting a cloud route.

Source: [Compute Engine general-purpose prices](https://cloud.google.com/products/compute/pricing/general-purpose).

## BigQuery scan-cost sensitivity

US on-demand analysis is $6.25/TiB, with the first 1 TiB/month free subject to account eligibility/usage. Below ignores the free allowance. Compressed archive size is **not** billable scan size. We have no measured expansion/billing factor; 2x, 4x and 8x are illustrative sensitivity factors, not a confidence interval or upper bound.

| Corpus | Billed bytes = 2x compressed | 4x | 8x |
|---|---:|---:|---:|
| French | $21 | $41 | $83 |
| English | $93 | $186 | $372 |
| Eight languages | $152 | $305 | $609 |

Formula: `compressed_bytes / 2**40 * assumed_billed_ratio * 6.25`. Runtime needs a pilot; these dollars do not determine query wall time. External sources do not automatically provide native-table speed. Ten repeated scans at the same size cost roughly ten times as much before allowances. BigQuery active logical result storage is about $0.023/GiB-month using 730 hours, with applicable free allowances; this is distinct from GCS export storage.

Example only: all eight languages at a measured-equivalent 4x billing ratio, retaining/exporting 100 GiB for one month, then downloading it once, would be about $305 + $2–$2.60 + $12 = $319–$320, plus temporary BigQuery result storage and small request charges. Neither the 4x ratio nor the 100 GiB output is measured. A 20 MB/s CPU-streaming scenario would instead have about $100 VM rental plus the same output costs and VM disk costs.

Source: [BigQuery pricing](https://cloud.google.com/bigquery/pricing).

## Embedding time and storage

Embed distinct text contexts once; store annual counts separately. Repeated counts do not require repeated copies of the same static BERT embedding. With 768 float32 components, vectors alone use 3,072 bytes per distinct context: 1 million = 3.072 GB; 10 million = 30.72 GB; 100 million = 307.2 GB. Metadata, counts and indexes are additional. Float16 halves vector storage but requires numerical validation before replacing the method's float32 outputs.

| Distinct contexts | At 100 contexts/sec | At 500 contexts/sec | At 1,000 contexts/sec |
|---|---:|---:|---:|
| 1 million | 2.78 h | 0.56 h | 0.28 h |
| 10 million | 27.78 h | 5.56 h | 2.78 h |
| 100 million | 277.78 h | 55.56 h | 27.78 h |

These are arithmetic scenarios, not A6000 benchmarks. Tokenization, batching, sequence lengths, layer extraction and I/O affect throughput. GPU rental cost equals measured GPU-hours times the selected instance's hourly price; at a hypothetical $1/hour the hours above also give dollars, but that is not a vendor quote. Use the existing lab GPUs first, subject to allocation rules. Current code is not automatically eight-GPU parallel.

## Recommendation

Keep code in `~/projects/sentiment` and outputs in `~/projects/sentiment_data`. Do not buy a raw mirror just to filter one vocabulary. Benchmark one French shard, then several representative shards, preserving counts and provenance. Either stream on the lab machine or trial external BigQuery/cloud CPU filtering to avoid raw transfer to the lab. Persist compressed filtered records, manifests and float32 embeddings; add object storage for selected backups or distribution when their measured size is known. A public query/embedding API is a separate service with its own compute and bandwidth costs; object storage alone does not provide it.

# Google Books five-gram download sizes

Checked 2026-09-18. Decimal GB = 1,000,000,000 bytes; TB = 1,000 GB. All gzip shards, not finance-only matches or the combined 1–5-gram corpus.

The latest release on the [official bulk index](https://storage.googleapis.com/books/ngrams/books/datasetsv3.html) is 20200217 (February 2020), called 2019 in the [Viewer documentation](https://books.google.com/ngrams/info). The Viewer has newer data, but its latest series is not the same as this bulk export.

| Language/corpus | 2012 GB | 2020 GB | New/old | 2012 files | 2020 files |
| --- | ---: | ---: | ---: | ---: | ---: |
| English (general) | 253.53 | 8,179.33 | 32.26× | 723 | 19,423 |
| French | 55.35 | 1,821.48 | 32.91× | 678 | 3,071 |
| German | 16.63 | 1,422.92 | 85.59× | 674 | 2,262 |
| Spanish | 34.73 | 844.82 | 24.33× | 643 | 1,415 |
| Italian | 10.27 | 699.88 | 68.14× | 636 | 984 |
| Russian | 494.28 | 356.12 | 0.72× | 633 | 633 |
| Chinese (simplified) | 156.03 | 38.51 | 0.25× | 519 | 105 |
| Hebrew | 70.37 | 32.05 | 0.46× | 400 | 42 |
| American English | 187.20 | 4,877.57 | 26.06× | 722 | 11,145 |
| British English | 72.13 | 1,813.36 | 25.14× | 709 | 3,098 |
| English fiction | 32.18 | 659.02 | 20.48× | 669 | 1,449 |

Numbers are calculated by summing Google Cloud Storage object `size` metadata across every page for each prefix. Only .gz objects are included. For 2020 each corpus file count was also checked against the shard total encoded in its filenames. No corpus payloads were downloaded.

English variants overlap the general English corpus and should not be treated as independent languages in totals. English One Million is a 2009-only collection, absent from both compared releases. Compressed-size ratios do not measure growth in distinct books, phrases, or finance contexts.

## Metadata sources

- English (general): [2012 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-eng-all-5gram-20120701-&maxResults=1000), [2020 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2F20200217%2Feng%2F5-&maxResults=1000). Follow nextPageToken for subsequent pages.
- French: [2012 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-fre-all-5gram-20120701-&maxResults=1000), [2020 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2F20200217%2Ffre%2F5-&maxResults=1000). Follow nextPageToken for subsequent pages.
- German: [2012 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-ger-all-5gram-20120701-&maxResults=1000), [2020 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2F20200217%2Fger%2F5-&maxResults=1000). Follow nextPageToken for subsequent pages.
- Spanish: [2012 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-spa-all-5gram-20120701-&maxResults=1000), [2020 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2F20200217%2Fspa%2F5-&maxResults=1000). Follow nextPageToken for subsequent pages.
- Italian: [2012 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-ita-all-5gram-20120701-&maxResults=1000), [2020 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2F20200217%2Fita%2F5-&maxResults=1000). Follow nextPageToken for subsequent pages.
- Russian: [2012 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-rus-all-5gram-20120701-&maxResults=1000), [2020 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2F20200217%2Frus%2F5-&maxResults=1000). Follow nextPageToken for subsequent pages.
- Chinese (simplified): [2012 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-chi-sim-all-5gram-20120701-&maxResults=1000), [2020 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2F20200217%2Fchi_sim%2F5-&maxResults=1000). Follow nextPageToken for subsequent pages.
- Hebrew: [2012 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-heb-all-5gram-20120701-&maxResults=1000), [2020 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2F20200217%2Fheb%2F5-&maxResults=1000). Follow nextPageToken for subsequent pages.
- American English: [2012 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-eng-us-all-5gram-20120701-&maxResults=1000), [2020 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2F20200217%2Feng-us%2F5-&maxResults=1000). Follow nextPageToken for subsequent pages.
- British English: [2012 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-eng-gb-all-5gram-20120701-&maxResults=1000), [2020 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2F20200217%2Feng-gb%2F5-&maxResults=1000). Follow nextPageToken for subsequent pages.
- English fiction: [2012 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2Fgooglebooks-eng-fiction-all-5gram-20120701-&maxResults=1000), [2020 metadata](https://storage.googleapis.com/storage/v1/b/books/o?prefix=ngrams%2Fbooks%2F20200217%2Feng-fiction%2F5-&maxResults=1000). Follow nextPageToken for subsequent pages.

The current project pipeline still targets 2012. Supporting 2020 requires release-aware discovery and ingestion, with separate run directories; this inventory does not modify running jobs.

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List or download Google Books Ngram shards using the authors' downloader."
    )
    parser.add_argument("--language", default="fre")
    parser.add_argument("--ngram-length", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/raw/fre"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--all-shards",
        action="store_true",
        help=(
            "Download every shard. By default, reproduce the authors' French "
            "partsonly selection and retain filenames ending in '_.gz'."
        ),
    )
    args = parser.parse_args()

    try:
        from google_ngram_downloader import util
    except ImportError as error:
        raise SystemExit(
            "Install the download extra first: pip install -e '.[download]'"
        ) from error

    records = util.iter_google_store(
        ngram_len=args.ngram_length,
        lang=args.language,
        verbose=True,
        getrequest=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    selected = 0
    for filename, url, _request in records:
        if not args.all_shards and args.language == "fre" and not filename.endswith("_.gz"):
            continue
        if args.limit is not None and selected >= args.limit:
            break
        selected += 1
        destination = args.output / filename
        print(f"{filename}\t{url}")
        if not args.dry_run and not destination.exists():
            urlretrieve(url, destination)


if __name__ == "__main__":
    main()

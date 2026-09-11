from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


REFERENCE_URL = "https://dataverse.harvard.edu/api/access/datafile/10695404"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the authors' public raw and standardized annual scores."
    )
    parser.add_argument("--output", type=Path, default=Path("data/reference/f.tab"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(REFERENCE_URL, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Set

from bioxrep.data.io import read_jsonl, write_jsonl


def reference_fact_ids(path: Path) -> Set[str]:
    return {str(row["fact_id"]) for row in read_jsonl(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter equivalence classes by fact IDs present in a reference JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_ids = reference_fact_ids(args.reference)
    examples = [row for row in read_jsonl(args.input) if str(row["fact_id"]) in keep_ids]
    write_jsonl(examples, args.output)
    print(f"Wrote {len(examples)} equivalence classes to {args.output}")


if __name__ == "__main__":
    main()

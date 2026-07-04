from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Set

from bioxrep.data.io import read_jsonl, write_jsonl


def reference_fact_ids(path: Path) -> Set[str]:
    return {str(row["fact_id"]) for row in read_jsonl(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter equivalence classes by fact IDs present in a reference JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["keep", "exclude"],
        default="keep",
        help=(
            "keep: retain input classes whose fact_id IS in --reference. "
            "exclude: drop input classes whose fact_id is in --reference "
            "(use this to remove test facts from a training set)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_ids = reference_fact_ids(args.reference)
    rows = read_jsonl(args.input)
    if args.mode == "keep":
        examples = [row for row in rows if str(row["fact_id"]) in reference_ids]
    else:
        examples = [row for row in rows if str(row["fact_id"]) not in reference_ids]
    write_jsonl(examples, args.output)
    print(
        f"Wrote {len(examples)} equivalence classes to {args.output} "
        f"(mode={args.mode}, {len(rows)} input, {len(reference_ids)} reference fact_ids)"
    )


if __name__ == "__main__":
    main()

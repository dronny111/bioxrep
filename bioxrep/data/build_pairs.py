from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from bioxrep.data.io import read_jsonl, write_jsonl


def forms_by_notation(example: Dict[str, object], notation: str) -> List[Dict[str, str]]:
    return [form for form in example["forms"] if form["notation"] == notation]


def flat_forms_by_notation(
    rows: Sequence[Dict[str, object]],
    notation: str,
    split: str | None,
) -> List[Dict[str, str]]:
    forms: List[Dict[str, str]] = []
    for row in rows:
        if row["notation"] != notation:
            continue
        if split is not None and row.get("split") != split:
            continue
        forms.append({"text": str(row["text"]), "notation": str(row["notation"])})
    return forms


def build_pairs(
    examples: Iterable[Dict[str, object]],
    left_notation: str,
    right_notation: str,
    left_split: str | None = None,
    right_split: str | None = None,
    max_pairs: int | None = None,
) -> List[Dict[str, object]]:
    if left_split is not None or right_split is not None:
        raise ValueError(
            "Split filters require flattened form rows with a 'split' field. "
            "Run build_pairs on the output of make_splits, or omit --left-split/--right-split "
            "for equivalence-class JSONL input."
        )

    pairs: List[Dict[str, object]] = []
    for example in examples:
        left_forms = forms_by_notation(example, left_notation)
        right_forms = forms_by_notation(example, right_notation)
        for left in left_forms:
            for right in right_forms:
                if left["text"] == right["text"]:
                    continue
                pairs.append(
                    {
                        "fact_id": example["fact_id"],
                        "track": example["track"],
                        "left_text": left["text"],
                        "left_notation": left["notation"],
                        "right_text": right["text"],
                        "right_notation": right["notation"],
                        "attributes": example["attributes"],
                    }
                )
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs
    return pairs


def build_pairs_from_flat_rows(
    rows: Sequence[Dict[str, object]],
    left_notation: str,
    right_notation: str,
    left_split: str | None = None,
    right_split: str | None = None,
    max_pairs: int | None = None,
) -> List[Dict[str, object]]:
    rows_by_fact: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_fact[str(row["fact_id"])].append(row)

    pairs: List[Dict[str, object]] = []
    for fact_rows in rows_by_fact.values():
        left_forms = flat_forms_by_notation(fact_rows, left_notation, left_split)
        right_forms = flat_forms_by_notation(fact_rows, right_notation, right_split)
        if not left_forms or not right_forms:
            continue
        first_row = fact_rows[0]
        for left in left_forms:
            for right in right_forms:
                if left["text"] == right["text"]:
                    continue
                pairs.append(
                    {
                        "fact_id": first_row["fact_id"],
                        "track": first_row["track"],
                        "left_text": left["text"],
                        "left_notation": left["notation"],
                        "right_text": right["text"],
                        "right_notation": right["notation"],
                        "attributes": first_row["attributes"],
                    }
                )
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BioXRep positive training pairs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-notation", required=True)
    parser.add_argument("--right-notation", required=True)
    parser.add_argument("--left-split", default=None)
    parser.add_argument("--right-split", default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    if rows and "forms" in rows[0]:
        pairs = build_pairs(
            rows,
            left_notation=args.left_notation,
            right_notation=args.right_notation,
            left_split=args.left_split,
            right_split=args.right_split,
            max_pairs=args.max_pairs,
        )
    else:
        pairs = build_pairs_from_flat_rows(
            rows,
            left_notation=args.left_notation,
            right_notation=args.right_notation,
            left_split=args.left_split,
            right_split=args.right_split,
            max_pairs=args.max_pairs,
        )
    write_jsonl(pairs, args.output)
    print(f"Wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()

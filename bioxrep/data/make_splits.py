from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from bioxrep.data.io import FlatForm, flatten_equivalence_classes, read_jsonl, write_jsonl


def assign_heldout_notation_split(
    rows: List[FlatForm],
    heldout_notation: str,
    track: str | None,
) -> List[FlatForm]:
    split_rows: List[FlatForm] = []
    for row in rows:
        row_with_split = dict(row)
        is_target_track = track is None or row["track"] == track
        row_with_split["split"] = "test" if is_target_track and row["notation"] == heldout_notation else "train"
        split_rows.append(row_with_split)
    return split_rows


def assign_heldout_attribute_split(
    rows: List[FlatForm],
    attribute: str,
    value: str,
    track: str | None,
) -> List[FlatForm]:
    split_rows: List[FlatForm] = []
    for row in rows:
        row_with_split = dict(row)
        is_target_track = track is None or row["track"] == track
        attr_value = row["attributes"].get(attribute)
        row_with_split["split"] = "test" if is_target_track and str(attr_value) == value else "train"
        split_rows.append(row_with_split)
    return split_rows


def numeric_attribute_value(value: object) -> float | None:
    if value is None or isinstance(value, list):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def assign_heldout_numeric_range_split(
    rows: List[FlatForm],
    attribute: str,
    min_value: float,
    max_value: float,
    track: str | None,
) -> List[FlatForm]:
    split_rows: List[FlatForm] = []
    for row in rows:
        row_with_split = dict(row)
        is_target_track = track is None or row["track"] == track
        attr_value = numeric_attribute_value(row["attributes"].get(attribute))
        in_range = attr_value is not None and min_value <= attr_value <= max_value
        row_with_split["split"] = "test" if is_target_track and in_range else "train"
        split_rows.append(row_with_split)
    return split_rows


def split_counts(rows: List[FlatForm]) -> Dict[str, int]:
    counts = {"train": 0, "test": 0}
    for row in rows:
        counts[str(row["split"])] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create BioXRep train/test form splits.")
    parser.add_argument("--input", type=Path, default=Path("data/bioxrep_synth.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/bioxrep_synth_split.jsonl"))
    parser.add_argument(
        "--track",
        choices=["variant", "lab", "gene_alias", "clinvar_gene", "clinvar_hgvs", "clinvar_summary"],
        default=None,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--heldout-notation", default=None)
    group.add_argument("--heldout-attribute", nargs=2, metavar=("NAME", "VALUE"))
    group.add_argument("--heldout-numeric-range", nargs=3, metavar=("NAME", "MIN", "MAX"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = flatten_equivalence_classes(read_jsonl(args.input))

    if args.heldout_notation is not None:
        split_rows = assign_heldout_notation_split(rows, args.heldout_notation, args.track)
        split_name = f"notation={args.heldout_notation}"
    elif args.heldout_attribute is not None:
        attribute, value = args.heldout_attribute
        split_rows = assign_heldout_attribute_split(rows, attribute, value, args.track)
        split_name = f"attribute={attribute}:{value}"
    else:
        attribute, min_value, max_value = args.heldout_numeric_range
        min_bound = float(min_value)
        max_bound = float(max_value)
        if min_bound > max_bound:
            raise ValueError(f"Invalid numeric range: min {min_bound} is greater than max {max_bound}")
        split_rows = assign_heldout_numeric_range_split(rows, attribute, min_bound, max_bound, args.track)
        split_name = f"numeric_range={attribute}:{min_bound}:{max_bound}"

    counts = split_counts(split_rows)
    if counts["test"] == 0:
        raise ValueError(f"Split produced no test rows for {split_name}")

    write_jsonl(split_rows, args.output)
    print(f"Wrote {counts['train']} train and {counts['test']} test rows to {args.output}")


if __name__ == "__main__":
    main()

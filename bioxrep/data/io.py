from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


Example = Dict[str, Any]
FlatForm = Dict[str, Any]


def read_jsonl(path: Path) -> List[Example]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def flatten_equivalence_classes(examples: Iterable[Example]) -> List[FlatForm]:
    rows: List[FlatForm] = []
    for example in examples:
        for form_idx, form in enumerate(example["forms"]):
            rows.append(
                {
                    "form_id": f"{example['fact_id']}::{form_idx}",
                    "fact_id": example["fact_id"],
                    "track": example["track"],
                    "notation": form["notation"],
                    "text": form["text"],
                    "attributes": example["attributes"],
                }
            )
    return rows

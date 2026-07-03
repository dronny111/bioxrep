from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from bioxrep.data.io import write_jsonl
from bioxrep.data.schemas import EquivalenceClass, NotationForm


PIPE_FIELDS = {
    "alias_symbol",
    "alias_name",
    "prev_symbol",
    "prev_name",
    "refseq_accession",
    "uniprot_ids",
}


def split_pipe_values(value: str) -> List[str]:
    if not value:
        return []
    value = value.strip().strip('"')
    return [part.strip() for part in value.split("|") if part.strip()]


def mane_values(value: str) -> List[str]:
    if not value:
        return []
    pieces: List[str] = []
    for group in split_pipe_values(value):
        pieces.extend(part.strip() for part in group.split("|") if part.strip())
    return pieces


def add_unique_form(forms: List[NotationForm], seen: set[str], text: str, notation: str) -> None:
    cleaned = text.strip().strip('"')
    if not cleaned or cleaned in seen:
        return
    seen.add(cleaned)
    forms.append(NotationForm(text=cleaned, notation=notation))


def row_to_equivalence_class(row: Dict[str, str]) -> EquivalenceClass | None:
    forms: List[NotationForm] = []
    seen: set[str] = set()

    add_unique_form(forms, seen, row.get("symbol", ""), "approved_symbol")
    add_unique_form(forms, seen, row.get("name", ""), "approved_name")
    add_unique_form(forms, seen, row.get("ensembl_gene_id", ""), "ensembl_gene_id")
    add_unique_form(forms, seen, row.get("entrez_id", ""), "entrez_gene_id")

    for field in PIPE_FIELDS:
        for value in split_pipe_values(row.get(field, "")):
            add_unique_form(forms, seen, value, field)

    for value in mane_values(row.get("mane_select", "")):
        add_unique_form(forms, seen, value, "mane_select")

    if len(forms) < 2:
        return None

    symbol = row.get("symbol", "").strip()
    hgnc_id = row.get("hgnc_id", "").strip()
    fact_id = f"gene:{hgnc_id}:{symbol}"

    return EquivalenceClass(
        fact_id=fact_id,
        track="gene_alias",
        forms=forms,
        attributes={
            "hgnc_id": hgnc_id,
            "symbol": symbol,
            "name": row.get("name", "").strip(),
            "locus_group": row.get("locus_group", "").strip(),
            "locus_type": row.get("locus_type", "").strip(),
            "status": row.get("status", "").strip(),
            "location": row.get("location", "").strip(),
            "entrez_id": row.get("entrez_id", "").strip(),
            "ensembl_gene_id": row.get("ensembl_gene_id", "").strip(),
        },
    )


def build_hgnc_alias_examples(input_path: Path, max_examples: int | None = None) -> List[EquivalenceClass]:
    examples: List[EquivalenceClass] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            example = row_to_equivalence_class(row)
            if example is None:
                continue
            examples.append(example)
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BioXRep gene-alias equivalence classes from HGNC.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/hgnc_complete_set/hgnc_complete_set.txt"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/bioxrep_hgnc_aliases.jsonl"))
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_hgnc_alias_examples(args.input, max_examples=args.max_examples)
    write_jsonl((example.to_dict() for example in examples), args.output)
    print(f"Wrote {len(examples)} HGNC alias equivalence classes to {args.output}")


if __name__ == "__main__":
    main()

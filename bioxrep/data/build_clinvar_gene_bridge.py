from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
from typing import Dict, Iterable, List

from bioxrep.data.io import write_jsonl
from bioxrep.data.schemas import EquivalenceClass, NotationForm


def split_pipe_values(value: str) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.strip().strip('"').split("|") if part.strip()]


def add_unique_form(forms: List[NotationForm], seen: set[str], text: str, notation: str) -> None:
    cleaned = text.strip().strip('"')
    if not cleaned or cleaned in seen:
        return
    seen.add(cleaned)
    forms.append(NotationForm(text=cleaned, notation=notation))


def load_hgnc_by_entrez(path: Path) -> Dict[str, Dict[str, str]]:
    hgnc_by_entrez: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entrez_id = row.get("entrez_id", "").strip()
            if entrez_id:
                hgnc_by_entrez[entrez_id] = row
    return hgnc_by_entrez


def clinvar_rows(path: Path) -> Iterable[Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames and reader.fieldnames[0].startswith("#"):
            reader.fieldnames[0] = reader.fieldnames[0].lstrip("#")
        for row in reader:
            yield row


def row_to_equivalence_class(
    row: Dict[str, str],
    hgnc_by_entrez: Dict[str, Dict[str, str]],
) -> EquivalenceClass | None:
    allele_id = row.get("AlleleID", "").strip()
    gene_id = row.get("GeneID", "").strip()
    symbol = row.get("Symbol", "").strip()
    name = row.get("Name", "").strip()
    hgnc = hgnc_by_entrez.get(gene_id, {})

    if not allele_id or not gene_id or not symbol:
        return None

    forms: List[NotationForm] = []
    seen: set[str] = set()
    add_unique_form(forms, seen, allele_id, "clinvar_allele_id")
    add_unique_form(forms, seen, f"ClinVar AlleleID {allele_id}", "clinvar_allele_text")
    add_unique_form(forms, seen, gene_id, "entrez_gene_id")
    add_unique_form(forms, seen, symbol, "gene_symbol")
    add_unique_form(forms, seen, name, "gene_name")

    add_unique_form(forms, seen, hgnc.get("hgnc_id", ""), "hgnc_id")
    add_unique_form(forms, seen, hgnc.get("symbol", ""), "hgnc_symbol")
    add_unique_form(forms, seen, hgnc.get("name", ""), "hgnc_name")
    add_unique_form(forms, seen, hgnc.get("ensembl_gene_id", ""), "ensembl_gene_id")

    for value in split_pipe_values(hgnc.get("alias_symbol", "")):
        add_unique_form(forms, seen, value, "hgnc_alias_symbol")
    for value in split_pipe_values(hgnc.get("prev_symbol", "")):
        add_unique_form(forms, seen, value, "hgnc_prev_symbol")
    for value in split_pipe_values(hgnc.get("refseq_accession", "")):
        add_unique_form(forms, seen, value, "refseq_accession")
    for value in split_pipe_values(hgnc.get("uniprot_ids", "")):
        add_unique_form(forms, seen, value, "uniprot_id")

    if len(forms) < 3:
        return None

    return EquivalenceClass(
        fact_id=f"clinvar_gene:{allele_id}:{gene_id}",
        track="clinvar_gene",
        forms=forms,
        attributes={
            "allele_id": allele_id,
            "gene_id": gene_id,
            "symbol": symbol,
            "name": name,
            "genes_per_allele_id": row.get("GenesPerAlleleID", "").strip(),
            "category": row.get("Category", "").strip(),
            "source": row.get("Source", "").strip(),
            "hgnc_id": hgnc.get("hgnc_id", "").strip(),
            "hgnc_symbol": hgnc.get("symbol", "").strip(),
            "hgnc_name": hgnc.get("name", "").strip(),
            "ensembl_gene_id": hgnc.get("ensembl_gene_id", "").strip(),
        },
    )


def build_examples(
    clinvar_path: Path,
    hgnc_path: Path,
    max_examples: int | None = None,
    only_single_gene: bool = True,
) -> List[EquivalenceClass]:
    hgnc_by_entrez = load_hgnc_by_entrez(hgnc_path)
    examples: List[EquivalenceClass] = []
    seen_fact_ids: set[str] = set()

    for row in clinvar_rows(clinvar_path):
        if only_single_gene and row.get("GenesPerAlleleID", "").strip() != "1":
            continue
        example = row_to_equivalence_class(row, hgnc_by_entrez)
        if example is None or example.fact_id in seen_fact_ids:
            continue
        seen_fact_ids.add(example.fact_id)
        examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BioXRep ClinVar allele-to-gene equivalence classes.")
    parser.add_argument(
        "--clinvar-allele-gene",
        type=Path,
        default=Path("data/raw/clinvar_allele_gene/allele_gene.txt.gz"),
    )
    parser.add_argument(
        "--hgnc",
        type=Path,
        default=Path("data/raw/hgnc_complete_set/hgnc_complete_set.txt"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/bioxrep_clinvar_gene_bridge.jsonl"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--include-multi-gene", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_examples(
        clinvar_path=args.clinvar_allele_gene,
        hgnc_path=args.hgnc,
        max_examples=args.max_examples,
        only_single_gene=not args.include_multi_gene,
    )
    write_jsonl((example.to_dict() for example in examples), args.output)
    print(f"Wrote {len(examples)} ClinVar-gene bridge equivalence classes to {args.output}")


if __name__ == "__main__":
    main()

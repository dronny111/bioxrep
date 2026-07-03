from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from bioxrep.data.io import write_jsonl
from bioxrep.data.schemas import EquivalenceClass, NotationForm


HGVS_COLUMNS = [
    "Symbol",
    "GeneID",
    "VariationID",
    "AlleleID",
    "Type",
    "Assembly",
    "NucleotideExpression",
    "NucleotideChange",
    "ProteinExpression",
    "ProteinChange",
    "UsedForNaming",
    "Submitted",
    "OnRefSeqGene",
]


def clean(value: str) -> str:
    value = value.strip().strip('"')
    return "" if value == "-" else value


def split_pipe_values(value: str) -> List[str]:
    value = clean(value)
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def add_unique_form(forms: List[NotationForm], seen: set[str], text: str, notation: str) -> None:
    cleaned = clean(text)
    if not cleaned or cleaned in seen:
        return
    seen.add(cleaned)
    forms.append(NotationForm(text=cleaned, notation=notation))


PROTEIN_POSITION_RE = re.compile(r":p\.[A-Za-z*=()?_\[\]-]*?(\d+)")
NUCLEOTIDE_POSITION_RE = re.compile(r":[cgnmr]\.([*-]?\d+)")


def parse_protein_position(value: str) -> int | None:
    cleaned = clean(value)
    if not cleaned:
        return None
    match = PROTEIN_POSITION_RE.search(cleaned)
    if match is None:
        return None
    return int(match.group(1))


def parse_nucleotide_position(value: str) -> int | None:
    cleaned = clean(value)
    if not cleaned:
        return None
    match = NUCLEOTIDE_POSITION_RE.search(cleaned)
    if match is None:
        return None
    token = match.group(1)
    token = token.lstrip("*")
    try:
        return int(token)
    except ValueError:
        return None


def load_hgnc_by_entrez(path: Path) -> Dict[str, Dict[str, str]]:
    hgnc_by_entrez: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entrez_id = row.get("entrez_id", "").strip()
            if entrez_id:
                hgnc_by_entrez[entrez_id] = row
    return hgnc_by_entrez


def hgvs_rows(path: Path) -> Iterator[Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            values = line.rstrip("\n").split("\t")
            if len(values) != len(HGVS_COLUMNS):
                continue
            yield dict(zip(HGVS_COLUMNS, values))


def grouped_by_variation(rows: Iterable[Dict[str, str]]) -> Iterator[List[Dict[str, str]]]:
    current_id = ""
    group: List[Dict[str, str]] = []
    for row in rows:
        variation_id = clean(row.get("VariationID", ""))
        if not variation_id:
            continue
        if group and variation_id != current_id:
            yield group
            group = []
        current_id = variation_id
        group.append(row)
    if group:
        yield group


def group_to_equivalence_class(
    group: List[Dict[str, str]],
    hgnc_by_entrez: Dict[str, Dict[str, str]],
    require_gene: bool = True,
) -> EquivalenceClass | None:
    first = group[0]
    variation_id = clean(first.get("VariationID", ""))
    gene_id = clean(first.get("GeneID", ""))
    symbol = clean(first.get("Symbol", ""))
    allele_ids = sorted({clean(row.get("AlleleID", "")) for row in group if clean(row.get("AlleleID", ""))})

    if require_gene and (not gene_id or not symbol):
        return None

    hgnc = hgnc_by_entrez.get(gene_id, {})
    forms: List[NotationForm] = []
    seen: set[str] = set()

    add_unique_form(forms, seen, variation_id, "clinvar_variation_id")
    add_unique_form(forms, seen, f"ClinVar VariationID {variation_id}", "clinvar_variation_text")
    for allele_id in allele_ids:
        add_unique_form(forms, seen, allele_id, "clinvar_allele_id")
        add_unique_form(forms, seen, f"ClinVar AlleleID {allele_id}", "clinvar_allele_text")

    add_unique_form(forms, seen, gene_id, "entrez_gene_id")
    add_unique_form(forms, seen, symbol, "gene_symbol")
    add_unique_form(forms, seen, hgnc.get("hgnc_id", ""), "hgnc_id")
    add_unique_form(forms, seen, hgnc.get("name", ""), "hgnc_name")
    add_unique_form(forms, seen, hgnc.get("ensembl_gene_id", ""), "ensembl_gene_id")

    for value in split_pipe_values(hgnc.get("alias_symbol", "")):
        add_unique_form(forms, seen, value, "hgnc_alias_symbol")
    for value in split_pipe_values(hgnc.get("uniprot_ids", "")):
        add_unique_form(forms, seen, value, "uniprot_id")

    hgvs_types = set()
    assemblies = set()
    protein_positions = set()
    nucleotide_positions = set()
    cdna_positions = set()
    genomic_positions = set()
    for row in group:
        hgvs_type = clean(row.get("Type", ""))
        assembly = clean(row.get("Assembly", ""))
        if hgvs_type:
            hgvs_types.add(hgvs_type)
        if assembly and assembly != "na":
            assemblies.add(assembly)

        add_unique_form(forms, seen, row.get("NucleotideExpression", ""), "nucleotide_expression")
        add_unique_form(forms, seen, row.get("NucleotideChange", ""), "nucleotide_change")
        add_unique_form(forms, seen, row.get("ProteinExpression", ""), "protein_expression")
        add_unique_form(forms, seen, row.get("ProteinChange", ""), "protein_change")

        protein_position = parse_protein_position(row.get("ProteinExpression", ""))
        if protein_position is None:
            protein_position = parse_protein_position(row.get("ProteinChange", ""))
        if protein_position is not None:
            protein_positions.add(protein_position)

        for key in ("NucleotideExpression", "NucleotideChange"):
            cleaned_value = clean(row.get(key, ""))
            nucleotide_position = parse_nucleotide_position(cleaned_value)
            if nucleotide_position is None:
                continue
            nucleotide_positions.add(nucleotide_position)
            if ":c." in cleaned_value:
                cdna_positions.add(nucleotide_position)
            if ":g." in cleaned_value:
                genomic_positions.add(nucleotide_position)

    if len(forms) < 4:
        return None

    return EquivalenceClass(
        fact_id=f"clinvar_hgvs:{variation_id}",
        track="clinvar_hgvs",
        forms=forms,
        attributes={
            "variation_id": variation_id,
            "allele_ids": allele_ids,
            "gene_id": gene_id,
            "symbol": symbol,
            "hgnc_id": hgnc.get("hgnc_id", "").strip(),
            "hgnc_name": hgnc.get("name", "").strip(),
            "ensembl_gene_id": hgnc.get("ensembl_gene_id", "").strip(),
            "hgvs_types": sorted(hgvs_types),
            "assemblies": sorted(assemblies),
            "protein_position": min(protein_positions) if protein_positions else None,
            "nucleotide_position": min(nucleotide_positions) if nucleotide_positions else None,
            "cdna_position": min(cdna_positions) if cdna_positions else None,
            "genomic_position": min(genomic_positions) if genomic_positions else None,
        },
    )


def build_examples(
    hgvs_path: Path,
    hgnc_path: Path,
    max_examples: int | None = None,
    require_gene: bool = True,
) -> List[EquivalenceClass]:
    hgnc_by_entrez = load_hgnc_by_entrez(hgnc_path)
    examples: List[EquivalenceClass] = []
    for group in grouped_by_variation(hgvs_rows(hgvs_path)):
        example = group_to_equivalence_class(group, hgnc_by_entrez, require_gene=require_gene)
        if example is None:
            continue
        examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BioXRep ClinVar HGVS variant equivalence classes.")
    parser.add_argument("--hgvs", type=Path, default=Path("data/raw/clinvar_hgvs/hgvs4variation.txt.gz"))
    parser.add_argument("--hgnc", type=Path, default=Path("data/raw/hgnc_complete_set/hgnc_complete_set.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/bioxrep_clinvar_hgvs_variants.jsonl"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--include-intergenic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_examples(
        hgvs_path=args.hgvs,
        hgnc_path=args.hgnc,
        max_examples=args.max_examples,
        require_gene=not args.include_intergenic,
    )
    write_jsonl((example.to_dict() for example in examples), args.output)
    print(f"Wrote {len(examples)} ClinVar HGVS equivalence classes to {args.output}")


if __name__ == "__main__":
    main()

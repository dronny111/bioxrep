from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from bioxrep.data.io import write_jsonl
from bioxrep.data.schemas import EquivalenceClass, NotationForm


def clean(value: str) -> str:
    value = value.strip().strip('"')
    return "" if value in {"", "-", "na"} else value


def split_values(value: str, sep: str = "|") -> List[str]:
    value = clean(value)
    if not value:
        return []
    return [part.strip() for part in value.split(sep) if clean(part)]


def add_unique_form(forms: List[NotationForm], seen: set[str], text: str, notation: str) -> None:
    cleaned = clean(text)
    if not cleaned or cleaned in seen:
        return
    seen.add(cleaned)
    forms.append(NotationForm(text=cleaned, notation=notation))


def variant_summary_rows(path: Path) -> Iterator[Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames and reader.fieldnames[0].startswith("#"):
            reader.fieldnames[0] = reader.fieldnames[0].lstrip("#")
        for row in reader:
            yield row


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


def group_to_equivalence_class(group: List[Dict[str, str]]) -> EquivalenceClass | None:
    first = group[0]
    variation_id = clean(first.get("VariationID", ""))
    allele_id = clean(first.get("AlleleID", ""))
    gene_id = clean(first.get("GeneID", ""))
    gene_symbol = clean(first.get("GeneSymbol", ""))
    hgnc_id = clean(first.get("HGNC_ID", ""))

    if not variation_id:
        return None

    forms: List[NotationForm] = []
    seen: set[str] = set()
    add_unique_form(forms, seen, variation_id, "clinvar_variation_id")
    add_unique_form(forms, seen, f"ClinVar VariationID {variation_id}", "clinvar_variation_text")
    add_unique_form(forms, seen, allele_id, "clinvar_allele_id")
    add_unique_form(forms, seen, gene_id, "entrez_gene_id")
    add_unique_form(forms, seen, gene_symbol, "gene_symbol")
    add_unique_form(forms, seen, hgnc_id, "hgnc_id")

    assemblies = set()
    clinical_significance = set()
    phenotypes = set()
    review_status = set()
    variant_types = set()

    for row in group:
        assembly = clean(row.get("Assembly", ""))
        chromosome = clean(row.get("Chromosome", ""))
        start = clean(row.get("Start", ""))
        stop = clean(row.get("Stop", ""))
        reference_vcf = clean(row.get("ReferenceAlleleVCF", ""))
        alternate_vcf = clean(row.get("AlternateAlleleVCF", ""))

        if assembly:
            assemblies.add(assembly)
        if clean(row.get("Type", "")):
            variant_types.add(clean(row.get("Type", "")))
        if clean(row.get("ClinicalSignificance", "")):
            clinical_significance.add(clean(row.get("ClinicalSignificance", "")))
        if clean(row.get("ReviewStatus", "")):
            review_status.add(clean(row.get("ReviewStatus", "")))

        add_unique_form(forms, seen, row.get("Name", ""), "clinvar_name")
        add_unique_form(forms, seen, row.get("RS# (dbSNP)", ""), "dbsnp_rs")
        add_unique_form(forms, seen, row.get("Cytogenetic", ""), "cytogenetic")
        add_unique_form(forms, seen, row.get("ClinicalSignificance", ""), "clinical_significance")
        add_unique_form(forms, seen, row.get("ReviewStatus", ""), "review_status")

        for rcv in split_values(row.get("RCVaccession", "")):
            add_unique_form(forms, seen, rcv, "rcv_accession")
        for phenotype in split_values(row.get("PhenotypeList", "")):
            phenotypes.add(phenotype)
            add_unique_form(forms, seen, phenotype, "phenotype")
        for other_id in split_values(row.get("OtherIDs", "")):
            add_unique_form(forms, seen, other_id, "other_id")

        if chromosome and start:
            coord = f"{assembly}:{chromosome}:{start}-{stop or start}"
            add_unique_form(forms, seen, coord, "genomic_coordinate")
        if reference_vcf and alternate_vcf:
            vcf = f"{chromosome}:{clean(row.get('PositionVCF', ''))}:{reference_vcf}>{alternate_vcf}"
            add_unique_form(forms, seen, vcf, "vcf_allele")

    if len(forms) < 4:
        return None

    return EquivalenceClass(
        fact_id=f"clinvar_summary:{variation_id}",
        track="clinvar_summary",
        forms=forms,
        attributes={
            "variation_id": variation_id,
            "allele_id": allele_id,
            "gene_id": gene_id,
            "gene_symbol": gene_symbol,
            "hgnc_id": hgnc_id,
            "variant_types": sorted(variant_types),
            "assemblies": sorted(assemblies),
            "clinical_significance": sorted(clinical_significance),
            "phenotypes": sorted(phenotypes),
            "review_status": sorted(review_status),
        },
    )


def build_examples(path: Path, max_examples: int | None = None) -> List[EquivalenceClass]:
    examples: List[EquivalenceClass] = []
    for group in grouped_by_variation(variant_summary_rows(path)):
        example = group_to_equivalence_class(group)
        if example is None:
            continue
        examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BioXRep ClinVar summary equivalence classes.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/clinvar_variant_summary/variant_summary.txt.gz"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/bioxrep_clinvar_summary.jsonl"))
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_examples(args.input, max_examples=args.max_examples)
    write_jsonl((example.to_dict() for example in examples), args.output)
    print(f"Wrote {len(examples)} ClinVar summary equivalence classes to {args.output}")


if __name__ == "__main__":
    main()

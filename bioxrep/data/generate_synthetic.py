from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List

from bioxrep.data.schemas import EquivalenceClass, NotationForm


AMINO_ACIDS = [
    ("A", "Ala", "alanine"),
    ("R", "Arg", "arginine"),
    ("N", "Asn", "asparagine"),
    ("D", "Asp", "aspartic acid"),
    ("C", "Cys", "cysteine"),
    ("Q", "Gln", "glutamine"),
    ("E", "Glu", "glutamic acid"),
    ("G", "Gly", "glycine"),
    ("H", "His", "histidine"),
    ("I", "Ile", "isoleucine"),
    ("L", "Leu", "leucine"),
    ("K", "Lys", "lysine"),
    ("M", "Met", "methionine"),
    ("F", "Phe", "phenylalanine"),
    ("P", "Pro", "proline"),
    ("S", "Ser", "serine"),
    ("T", "Thr", "threonine"),
    ("W", "Trp", "tryptophan"),
    ("Y", "Tyr", "tyrosine"),
    ("V", "Val", "valine"),
]

GENES = [
    ("BRAF", "NM_004333.6"),
    ("EGFR", "NM_005228.5"),
    ("TP53", "NM_000546.6"),
    ("KRAS", "NM_004985.5"),
    ("PIK3CA", "NM_006218.4"),
    ("BRCA1", "NM_007294.4"),
    ("BRCA2", "NM_000059.4"),
    ("ALK", "NM_004304.5"),
]

LABS = [
    {
        "name": "glucose",
        "canonical_unit": "mg/dL",
        "alt_unit": "mmol/L",
        "factor": 0.0555,
        "low": 70.0,
        "high": 99.0,
        "values": [62.0, 85.0, 126.0, 180.0],
    },
    {
        "name": "creatinine",
        "canonical_unit": "mg/dL",
        "alt_unit": "umol/L",
        "factor": 88.4,
        "low": 0.7,
        "high": 1.3,
        "values": [0.6, 1.0, 1.8, 3.2],
    },
    {
        "name": "sodium",
        "canonical_unit": "mmol/L",
        "alt_unit": "mEq/L",
        "factor": 1.0,
        "low": 135.0,
        "high": 145.0,
        "values": [128.0, 140.0, 151.0],
    },
    {
        "name": "hemoglobin",
        "canonical_unit": "g/dL",
        "alt_unit": "g/L",
        "factor": 10.0,
        "low": 12.0,
        "high": 16.0,
        "values": [8.5, 13.7, 17.2],
    },
]


def status_for_value(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "normal"


def generate_variant_examples(rng: random.Random, n: int) -> List[EquivalenceClass]:
    examples: List[EquivalenceClass] = []
    for idx in range(n):
        gene, transcript = rng.choice(GENES)
        ref = rng.choice(AMINO_ACIDS)
        alt = rng.choice([aa for aa in AMINO_ACIDS if aa[0] != ref[0]])
        protein_position = rng.randint(5, 1200)
        cdna_position = protein_position * 3 - rng.choice([0, 1, 2])
        ref_base, alt_base = rng.choice([("A", "T"), ("C", "G"), ("G", "A"), ("T", "C")])
        fact_id = f"variant:{gene}:p.{ref[1]}{protein_position}{alt[1]}"

        forms = [
            NotationForm(f"{gene} {ref[0]}{protein_position}{alt[0]}", "protein_one_letter"),
            NotationForm(f"{gene} p.{ref[1]}{protein_position}{alt[1]}", "protein_three_letter"),
            NotationForm(f"{gene} {ref[1]}{protein_position}{alt[1]}", "protein_three_letter_short"),
            NotationForm(f"{transcript}:c.{cdna_position}{ref_base}>{alt_base}", "cdna_hgvs_like"),
            NotationForm(
                (
                    f"missense variant in {gene} changing {ref[2]} to {alt[2]} "
                    f"at protein position {protein_position}"
                ),
                "text_description",
            ),
        ]

        examples.append(
            EquivalenceClass(
                fact_id=fact_id,
                track="variant",
                forms=forms,
                attributes={
                    "gene": gene,
                    "transcript": transcript,
                    "protein_position": protein_position,
                    "reference_aa_1": ref[0],
                    "reference_aa_3": ref[1],
                    "reference_aa_name": ref[2],
                    "alternate_aa_1": alt[0],
                    "alternate_aa_3": alt[1],
                    "alternate_aa_name": alt[2],
                    "cdna_position": cdna_position,
                    "reference_base": ref_base,
                    "alternate_base": alt_base,
                    "mutation_type": "missense",
                },
            )
        )
    return examples


def generate_lab_examples() -> List[EquivalenceClass]:
    examples: List[EquivalenceClass] = []
    for lab in LABS:
        for value in lab["values"]:
            alt_value = value * lab["factor"]
            status = status_for_value(value, lab["low"], lab["high"])
            fact_id = f"lab:{lab['name']}:{value:g}:{lab['canonical_unit']}"

            forms = [
                NotationForm(f"{lab['name']} {value:g} {lab['canonical_unit']}", "canonical_unit"),
                NotationForm(f"{lab['name']} {alt_value:.3g} {lab['alt_unit']}", "alternate_unit"),
                NotationForm(
                    f"{lab['name']} reference range {lab['low']:g}-{lab['high']:g} {lab['canonical_unit']}",
                    "reference_range",
                ),
                NotationForm(f"{lab['name']} is {status}", "text_interpretation"),
            ]

            examples.append(
                EquivalenceClass(
                    fact_id=fact_id,
                    track="lab",
                    forms=forms,
                    attributes={
                        "lab_name": lab["name"],
                        "canonical_value": value,
                        "canonical_unit": lab["canonical_unit"],
                        "alternate_value": alt_value,
                        "alternate_unit": lab["alt_unit"],
                        "reference_low": lab["low"],
                        "reference_high": lab["high"],
                        "status": status,
                    },
                )
            )
    return examples


def write_jsonl(examples: Iterable[EquivalenceClass], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BioXRep synthetic equivalence classes.")
    parser.add_argument("--output", type=Path, default=Path("data/bioxrep_synth.jsonl"))
    parser.add_argument("--variant-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    examples = generate_variant_examples(rng, args.variant_count)
    examples.extend(generate_lab_examples())
    write_jsonl(examples, args.output)
    print(f"Wrote {len(examples)} BioXRep examples to {args.output}")


if __name__ == "__main__":
    main()

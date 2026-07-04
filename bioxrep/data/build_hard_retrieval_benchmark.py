from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from bioxrep.data.io import read_jsonl, write_jsonl
from bioxrep.data.build_clinvar_hgvs_variants import parse_nucleotide_position, parse_protein_position


Example = Dict[str, Any]
Candidate = Dict[str, Any]
ConfoundKey = Tuple[Any, ...]


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def normalized_attr_value(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        if not value:
            return None
        return tuple(value)
    return value


def confound_key(attributes: Dict[str, Any], fields: Sequence[str]) -> ConfoundKey | None:
    values = tuple(normalized_attr_value(attributes.get(field)) for field in fields)
    if any(value is None for value in values):
        return None
    return values


def forms_by_notation(example: Example, notation: str) -> List[Dict[str, str]]:
    return [form for form in example.get("forms", []) if form.get("notation") == notation]


def parsed_form_value(text: str, field: str) -> int | None:
    if field == "protein_position":
        return parse_protein_position(text)
    if field in {"nucleotide_position", "cdna_position", "genomic_position"}:
        return parse_nucleotide_position(text)
    return None


def form_matches_confound_key(form: Dict[str, str], confound_fields: Sequence[str], key: ConfoundKey) -> bool:
    for field, expected_value in zip(confound_fields, key):
        parsed_value = parsed_form_value(form["text"], field)
        if parsed_value is not None and parsed_value != expected_value:
            return False
    return True


def candidate_from_form(
    example: Example,
    form: Dict[str, str],
    is_positive: bool,
    decoy_kind: str = "positive",
) -> Candidate:
    return {
        "text": form["text"],
        "notation": form["notation"],
        "fact_id": example["fact_id"],
        "track": example["track"],
        "attributes": example.get("attributes", {}),
        "is_positive": is_positive,
        # "matched" decoys share the query's confound key (a hard negative);
        # "random" decoys are fill from the input pool and do NOT share it.
        "decoy_kind": decoy_kind,
    }


def build_target_index(
    examples: Sequence[Example],
    target_notation: str,
    confound_fields: Sequence[str],
) -> Dict[ConfoundKey, List[Example]]:
    indexed: Dict[ConfoundKey, List[Example]] = defaultdict(list)
    for example in examples:
        if not forms_by_notation(example, target_notation):
            continue
        key = confound_key(example.get("attributes", {}), confound_fields)
        if key is not None:
            indexed[key].append(example)
    return indexed


def build_hard_benchmark(
    examples: Sequence[Example],
    query_notation: str,
    candidate_notation: str,
    confound_fields: Sequence[str],
    min_decoys: int,
    max_queries: int | None,
    max_candidates: int,
    positives_per_query: int,
    fill_random_decoys: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    if max_candidates < positives_per_query + min_decoys:
        raise ValueError("--max-candidates must allow the requested positives and decoys")

    rng = random.Random(seed)
    target_index = build_target_index(examples, candidate_notation, confound_fields)
    candidate_examples = [example for example in examples if forms_by_notation(example, candidate_notation)]
    records: List[Dict[str, Any]] = []

    shuffled_examples = list(examples)
    rng.shuffle(shuffled_examples)

    for example in shuffled_examples:
        key = confound_key(example.get("attributes", {}), confound_fields)
        if key is None:
            continue

        query_forms = [
            form
            for form in forms_by_notation(example, query_notation)
            if form_matches_confound_key(form, confound_fields, key)
        ]
        positive_forms = forms_by_notation(example, candidate_notation)
        if not query_forms or not positive_forms:
            continue

        decoy_examples = [item for item in target_index.get(key, []) if item["fact_id"] != example["fact_id"]]
        if len(decoy_examples) < min_decoys:
            continue

        rng.shuffle(query_forms)
        rng.shuffle(positive_forms)
        rng.shuffle(decoy_examples)

        positive_candidates = [
            candidate_from_form(example, form, is_positive=True, decoy_kind="positive")
            for form in positive_forms[:positives_per_query]
        ]
        decoy_budget = max_candidates - len(positive_candidates)
        decoy_candidates: List[Candidate] = []
        for decoy in decoy_examples:
            decoy_forms = forms_by_notation(decoy, candidate_notation)
            if not decoy_forms:
                continue
            rng.shuffle(decoy_forms)
            decoy_candidates.append(
                candidate_from_form(decoy, decoy_forms[0], is_positive=False, decoy_kind="matched")
            )
            if len(decoy_candidates) >= decoy_budget:
                break

        matched_decoy_count = len(decoy_candidates)

        if fill_random_decoys and len(decoy_candidates) < decoy_budget:
            matched_fact_ids = {candidate["fact_id"] for candidate in decoy_candidates}
            random_decoys = [
                item
                for item in candidate_examples
                if item["fact_id"] != example["fact_id"] and item["fact_id"] not in matched_fact_ids
            ]
            rng.shuffle(random_decoys)
            for decoy in random_decoys:
                decoy_forms = forms_by_notation(decoy, candidate_notation)
                if not decoy_forms:
                    continue
                rng.shuffle(decoy_forms)
                decoy_candidates.append(
                    candidate_from_form(decoy, decoy_forms[0], is_positive=False, decoy_kind="random")
                )
                if len(decoy_candidates) >= decoy_budget:
                    break

        random_decoy_count = len(decoy_candidates) - matched_decoy_count

        if len(decoy_candidates) < min_decoys:
            continue

        for query_form in query_forms:
            candidates = positive_candidates + decoy_candidates
            rng.shuffle(candidates)
            records.append(
                {
                    "benchmark_id": f"{example['fact_id']}::{query_form['notation']}::{len(records)}",
                    "query": {
                        "text": query_form["text"],
                        "notation": query_form["notation"],
                        "fact_id": example["fact_id"],
                        "track": example["track"],
                        "attributes": example.get("attributes", {}),
                    },
                    "candidate_notation": candidate_notation,
                    "confound_fields": list(confound_fields),
                    "confound_values": dict(zip(confound_fields, key)),
                    "matched_decoy_count": matched_decoy_count,
                    "random_decoy_count": random_decoy_count,
                    "candidates": candidates,
                }
            )
            if max_queries is not None and len(records) >= max_queries:
                return records

    return records


def validate_examples(examples: Iterable[Example]) -> List[Example]:
    examples = list(examples)
    if not examples:
        raise ValueError("Input contains no examples")
    if "forms" not in examples[0]:
        raise ValueError("Hard benchmark construction requires equivalence-class JSONL with a forms field")
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-query hard retrieval sets with decoys matched on confounding attributes."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-notation", required=True)
    parser.add_argument("--candidate-notation", required=True)
    parser.add_argument(
        "--confound-fields",
        required=True,
        help="Comma-separated attribute fields decoys must share with the query fact.",
    )
    parser.add_argument("--min-decoys", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--positives-per-query", type=int, default=1)
    parser.add_argument(
        "--fill-random-decoys",
        action="store_true",
        help="After matched decoys are added, fill remaining candidate slots with random in-input decoys.",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = validate_examples(read_jsonl(args.input))
    confound_fields = parse_csv(args.confound_fields)
    if not confound_fields:
        raise ValueError("--confound-fields must include at least one field")

    records = build_hard_benchmark(
        examples=examples,
        query_notation=args.query_notation,
        candidate_notation=args.candidate_notation,
        confound_fields=confound_fields,
        min_decoys=args.min_decoys,
        max_queries=args.max_queries,
        max_candidates=args.max_candidates,
        positives_per_query=args.positives_per_query,
        fill_random_decoys=args.fill_random_decoys,
        seed=args.seed,
    )
    if not records:
        raise ValueError("No hard benchmark records were produced; try fewer confound fields or a larger input")

    write_jsonl(records, args.output)
    avg_candidates = sum(len(record["candidates"]) for record in records) / len(records)
    total_matched = sum(record["matched_decoy_count"] for record in records)
    total_random = sum(record["random_decoy_count"] for record in records)
    total_decoys = total_matched + total_random
    matched_fraction = total_matched / total_decoys if total_decoys else 0.0
    avg_matched = total_matched / len(records)
    all_hard = sum(1 for record in records if record["random_decoy_count"] == 0)
    print(
        f"Wrote {len(records)} hard retrieval queries to {args.output} "
        f"(avg candidates/query: {avg_candidates:.2f})"
    )
    print(
        "Decoy composition: "
        f"{matched_fraction:.1%} of decoys are confound-matched hard negatives "
        f"(avg {avg_matched:.2f} matched/query); "
        f"{all_hard}/{len(records)} queries have an all-hard pool (no random fill). "
        "Matched decoys share the query's confound key; random-fill decoys do not — "
        "report top-k with this composition in mind."
    )


if __name__ == "__main__":
    main()

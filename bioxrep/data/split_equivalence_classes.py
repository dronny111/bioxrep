from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from bioxrep.data.io import read_jsonl, write_jsonl


def has_notation(example: Dict[str, object], notation: str) -> bool:
    return any(form.get("notation") == notation for form in example.get("forms", []))


def filter_examples(
    examples: Sequence[Dict[str, object]],
    required_notations: Sequence[str],
) -> List[Dict[str, object]]:
    if not required_notations:
        return list(examples)
    return [
        example
        for example in examples
        if all(has_notation(example, notation) for notation in required_notations)
    ]


def split_examples(
    examples: Sequence[Dict[str, object]],
    train_fraction: float,
    max_examples: int | None,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    if max_examples is not None:
        shuffled = shuffled[:max_examples]
    split_at = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_fraction)))
    return shuffled[:split_at], shuffled[split_at:]


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fact-disjoint equivalence-class train/test splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--required-notations", default="")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = read_jsonl(args.input)
    required_notations = parse_csv(args.required_notations)
    examples = filter_examples(examples, required_notations)
    if len(examples) < 2:
        raise ValueError("Need at least two examples after filtering")
    train_examples, test_examples = split_examples(
        examples=examples,
        train_fraction=args.train_fraction,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    write_jsonl(train_examples, args.train_output)
    write_jsonl(test_examples, args.test_output)
    print(
        f"Wrote {len(train_examples)} train and {len(test_examples)} test equivalence classes "
        f"from {len(examples)} filtered examples"
    )


if __name__ == "__main__":
    main()

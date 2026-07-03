from __future__ import annotations

import argparse
from pathlib import Path

from bioxrep.data.io import flatten_equivalence_classes, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten BioXRep equivalence classes into form rows.")
    parser.add_argument("--input", type=Path, default=Path("data/bioxrep_synth.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/bioxrep_synth_forms.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = read_jsonl(args.input)
    rows = flatten_equivalence_classes(examples)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} flattened forms to {args.output}")


if __name__ == "__main__":
    main()

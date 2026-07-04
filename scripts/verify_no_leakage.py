"""Assert that a BioXRep train/test construction has no fact or notation leakage.

This is a guardrail for the two flagship benchmarks. It fails loudly (non-zero
exit) if any of the following hold:

* A ``fact_id`` appears in both the train and the test/held-out file
  (breaks the fact-disjoint claim).
* ``--heldout-notation`` is given and that notation appears in *any* training
  form (breaks the notation-disjoint claim, e.g. HGNC ``alias_symbol``).
* ``--heldout-notation`` is given and it is *absent* from the test file
  (the held-out notation must actually be what is tested).

Both files may be either equivalence-class JSONL (rows with a ``forms`` list) or
already-flattened form JSONL (rows with ``notation``/``fact_id``). Mixed inputs
are handled.

Examples
--------
HGVS fact-disjoint split (no notation held out)::

    python3 scripts/verify_no_leakage.py \\
        --train data/bioxrep_clinvar_hgvs_variants_numeric_scaled_train_40k.jsonl \\
        --test  data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl

HGNC alias-symbol fact- AND notation-disjoint construction::

    python3 scripts/verify_no_leakage.py \\
        --train data/bioxrep_hgnc_aliases_train_classes.jsonl \\
        --test  data/bioxrep_hgnc_alias_symbol_heldout.jsonl \\
        --test-split test \\
        --heldout-notation alias_symbol
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from bioxrep.data.io import read_jsonl


def fact_ids_and_notations(
    path: Path,
    split: str | None = None,
) -> Tuple[Set[str], Set[str]]:
    """Return (fact_ids, notations) present in ``path``.

    Works on both equivalence-class rows (with ``forms``) and flattened form
    rows. When ``split`` is given, only rows/forms in that split are counted
    (flattened rows carry ``split``; class rows are not split-filtered).
    """
    fact_ids: Set[str] = set()
    notations: Set[str] = set()
    for row in read_jsonl(path):
        forms = row.get("forms")
        if isinstance(forms, list):  # equivalence-class row
            fact_ids.add(str(row["fact_id"]))
            for form in forms:
                if isinstance(form, dict) and form.get("notation"):
                    notations.add(str(form["notation"]))
        else:  # flattened form row
            if split is not None and row.get("split") != split:
                continue
            if "fact_id" in row:
                fact_ids.add(str(row["fact_id"]))
            if row.get("notation"):
                notations.add(str(row["notation"]))
    return fact_ids, notations


def training_notation_forms(path: Path) -> Set[str]:
    """All notations that appear in the training file (any split)."""
    _, notations = fact_ids_and_notations(path, split=None)
    return notations


def check(
    train_path: Path,
    test_path: Path,
    test_split: str | None,
    heldout_notation: str | None,
) -> List[str]:
    failures: List[str] = []

    train_facts, train_notations = fact_ids_and_notations(train_path, split=None)
    test_facts, test_notations = fact_ids_and_notations(test_path, split=test_split)

    overlap = train_facts & test_facts
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        failures.append(
            f"FACT LEAKAGE: {len(overlap)} fact_id(s) appear in both train and test "
            f"(e.g. {sample}). Splits must be fact-disjoint."
        )

    if heldout_notation is not None:
        if heldout_notation in train_notations:
            failures.append(
                f"NOTATION LEAKAGE: held-out notation '{heldout_notation}' appears in the "
                f"training file — it must be entirely absent for a notation-disjoint test."
            )
        if heldout_notation not in test_notations:
            failures.append(
                f"NOTATION MISSING: held-out notation '{heldout_notation}' is not present in "
                f"the test file"
                + (f" (split={test_split})" if test_split else "")
                + " — the held-out notation must be what is actually tested."
            )

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assert no fact/notation leakage between train and test.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--test-split", default=None, help="If test is flattened, restrict test rows to this split.")
    parser.add_argument("--heldout-notation", default=None, help="Notation that must be absent from train and present in test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = check(args.train, args.test, args.test_split, args.heldout_notation)

    train_facts, train_notations = fact_ids_and_notations(args.train, split=None)
    test_facts, test_notations = fact_ids_and_notations(args.test, split=args.test_split)
    print(
        f"train: {len(train_facts)} facts, notations={sorted(train_notations)}\n"
        f"test:  {len(test_facts)} facts, notations={sorted(test_notations)}"
    )

    if failures:
        print("\nLEAKAGE CHECK FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nLEAKAGE CHECK PASSED: train/test are fact-disjoint"
          + (f" and '{args.heldout_notation}' is notation-disjoint." if args.heldout_notation else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())

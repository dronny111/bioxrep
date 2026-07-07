#!/usr/bin/env bash
# Multi-seed sweep for the MAIN HGNC results table (held-out alias_symbol row).
# Only the char-CNN student is seed-dependent; lexical / SapBERT / BioSyn are
# deterministic. Reproduces the frozen seed-13 config exactly and sweeps 5 seeds
# so the headline 0.051-vs-0.077 comparison carries across-seed mean +/- std.
set -euo pipefail
cd /Users/ronnypolle/Desktop/generic-numeric-representation-by-attention-distillation-main
export PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE

TRAINCLS=data/bioxrep_hgnc_aliases_train_classes.jsonl
VALID=data/bioxrep_hgnc_aliases_valid_pairs.jsonl
HELDOUT=data/bioxrep_hgnc_alias_symbol_heldout.jsonl
ROOT=outputs/multiseed_hgnc
mkdir -p $ROOT/eval

for seed in 13 17 23 42 101; do
  out=$ROOT/student_seed${seed}
  mkdir -p "$out"
  echo "=== TRAIN seed=$seed ==="
  python3 -m bioxrep.train.train_contrastive_student \
    --class-input $TRAINCLS \
    --class-notations approved_symbol,prev_symbol,alias_name,prev_name,approved_name \
    --forms-per-class 4 --valid-input $VALID \
    --output-dir "$out" \
    --encoder cnn --hidden-dim 64 --projection-dim 128 \
    --epochs 3 --batch-size 128 --max-length 64 --text-transform none --seed $seed 2>&1 | tail -2
  echo "=== RETRIEVAL seed=$seed (held-out alias_symbol) ==="
  python3 -m bioxrep.baselines.student_retrieval \
    --checkpoint "$out/char_cnn_student.pt" \
    --input $HELDOUT --track gene_alias \
    --query-notation alias_symbol --query-split test \
    --candidate-notation approved_symbol --candidate-split train \
    --max-queries 2000 --bootstrap \
    --output $ROOT/eval/student_alias_symbol_seed${seed}.json 2>&1 | tail -1
done
echo "=== HGNC MULTISEED DONE ==="

#!/usr/bin/env bash
# Multi-seed byte-level TRANSFORMER student for the MAIN HGNC results table
# (held-out alias_symbol row). Mirrors scripts/run_multiseed_hgnc_main.sh exactly
# (same data, hidden 64 / proj 128, 3 epochs, bs 128, max-length 64, 5 seeds,
# same held-out alias_symbol retrieval eval) and changes ONLY the encoder:
# --encoder transformer instead of --encoder cnn. This isolates the single axis
# convolution -> self-attention against the char-CNN student; SapBERT already
# covers the pretrained-subword-transformer cell.
set -euo pipefail
cd /Users/ronnypolle/Desktop/generic-numeric-representation-by-attention-distillation-main
export PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE

TRAINCLS=data/bioxrep_hgnc_aliases_train_classes.jsonl
VALID=data/bioxrep_hgnc_aliases_valid_pairs.jsonl
HELDOUT=data/bioxrep_hgnc_alias_symbol_heldout.jsonl
ROOT=outputs/multiseed_hgnc_transformer
mkdir -p $ROOT/eval

for seed in 13 17 23 42 101; do
  out=$ROOT/student_seed${seed}
  mkdir -p "$out"
  echo "=== TRAIN transformer seed=$seed ==="
  python3 -m bioxrep.train.train_contrastive_student \
    --class-input $TRAINCLS \
    --class-notations approved_symbol,prev_symbol,alias_name,prev_name,approved_name \
    --forms-per-class 4 --valid-input $VALID \
    --output-dir "$out" \
    --encoder transformer --hidden-dim 64 --projection-dim 128 \
    --transformer-layers 2 --transformer-heads 4 \
    --epochs 3 --batch-size 128 --max-length 64 --text-transform none --seed $seed 2>&1 | tail -2
  echo "=== RETRIEVAL transformer seed=$seed (held-out alias_symbol) ==="
  python3 -m bioxrep.baselines.student_retrieval \
    --checkpoint "$out/char_transformer_student.pt" \
    --input $HELDOUT --track gene_alias \
    --query-notation alias_symbol --query-split test \
    --candidate-notation approved_symbol --candidate-split train \
    --max-queries 2000 --bootstrap \
    --output $ROOT/eval/student_alias_symbol_seed${seed}.json 2>&1 | tail -1
done
echo "=== HGNC TRANSFORMER MULTISEED DONE ==="

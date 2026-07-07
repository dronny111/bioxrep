#!/usr/bin/env bash
# Track B (input-feature ablation, corrected split): compare how the numeric
# POSITION is injected as an INPUT feature, holding everything else fixed.
#   none       -> text-only byte encoder (each digit is already its own byte
#                 token, so this is also the digit-tokenization baseline)
#   explicit   -> raw normalized scalar + present-mask, projected
#   sinusoidal -> multi-frequency Fourier features of the value
#   xval       -> xVal-style: one learned embedding per field scaled by the value
# All arms: numeric-loss-weight 0 (no auxiliary loss), seed 13, evaluated on the
# full 2k strict position-confounded hard benchmark. Regenerates the explicit/
# sinusoidal rows on the deduped fact-disjoint split (the old table predates the
# 702-duplicate leakage fix) so every row is apples-to-apples.
set -euo pipefail
cd /Users/ronnypolle/Desktop/generic-numeric-representation-by-attention-distillation-main
export PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 KMP_DUPLICATE_LIB_OK=TRUE

TRAIN=data/bioxrep_clinvar_hgvs_scaled_train_pairs_50k.jsonl
VALID=data/bioxrep_clinvar_hgvs_scaled_test_pairs_20k.jsonl
HARD=data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl
ROOT=outputs/trackb_featmode
mkdir -p $ROOT/eval

run_arm () {
  local mode=$1
  local out=$ROOT/feat_${mode}
  mkdir -p "$out"
  echo "=== TRAIN numeric-feature-mode=$mode ==="
  python3 -m bioxrep.train.train_contrastive_student \
    --input $TRAIN --valid-input $VALID --max-valid-pairs 5000 \
    --hard-valid-input $HARD --max-hard-valid-queries 500 \
    --output-dir "$out" \
    --encoder cnn --hidden-dim 64 --projection-dim 128 --epochs 3 \
    --batch-size 128 --max-length 160 --seed 13 \
    --attribute-fields variation_id --attribute-loss-weight 0.2 \
    --numeric-fields protein_position,cdna_position \
    --numeric-feature-mode $mode \
    --numeric-loss-weight 0.0 2>&1 | tail -3
  echo "=== HARD retrieval (mode=$mode) ==="
  python3 -m bioxrep.eval.hard_student_retrieval \
    --checkpoint "$out/char_cnn_student.pt" --input $HARD \
    --bootstrap --output $ROOT/eval/hard_feat_${mode}.json 2>&1 | tail -2
}

run_arm none
run_arm explicit
run_arm sinusoidal
run_arm xval
echo "=== TRACK B FEATURE-MODE DONE ==="

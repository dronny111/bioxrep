#!/usr/bin/env bash
# Track B: numeric-consistency-loss ablation (gap #3).
# Isolates the auxiliary numeric loss (per-field linear head regressing each
# form's embedding to its normalized protein_position / cdna_position, MSE +
# left/right agreement). Input is TEXT-ONLY in every arm (--numeric-feature-mode
# none) so the only thing that varies is --numeric-loss-weight. This asks: does
# forcing the embedding to be numeric-decodable help or hurt cross-notation
# retrieval, especially hard top-1 under position-confounded decoys?
set -euo pipefail
cd /Users/ronnypolle/Desktop/generic-numeric-representation-by-attention-distillation-main
export PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 KMP_DUPLICATE_LIB_OK=TRUE

TRAIN=data/bioxrep_clinvar_hgvs_scaled_train_pairs_50k.jsonl
VALID=data/bioxrep_clinvar_hgvs_scaled_test_pairs_20k.jsonl
HARD=data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl
ROOT=outputs/trackb
mkdir -p $ROOT/eval

run_arm () {
  local tag=$1 w=$2
  local out=$ROOT/numloss_${tag}
  mkdir -p "$out"
  echo "=== TRAIN numeric-loss-weight=$w (text-only input) ==="
  python3 -m bioxrep.train.train_contrastive_student \
    --input $TRAIN --valid-input $VALID --max-valid-pairs 5000 \
    --hard-valid-input $HARD --max-hard-valid-queries 500 \
    --output-dir "$out" \
    --encoder cnn --hidden-dim 64 --projection-dim 128 --epochs 3 \
    --batch-size 128 --max-length 160 --seed 13 \
    --attribute-fields variation_id --attribute-loss-weight 0.2 \
    --numeric-fields protein_position,cdna_position \
    --numeric-feature-mode none \
    --numeric-loss-weight $w 2>&1 | tail -3
  echo "=== HARD retrieval (weight=$w) ==="
  python3 -m bioxrep.eval.hard_student_retrieval \
    --checkpoint "$out/char_cnn_student.pt" --input $HARD \
    --bootstrap --output $ROOT/eval/hard_numloss_${tag}.json 2>&1 | tail -2
}

run_arm w00 0.0
run_arm w01 0.1
run_arm w05 0.5
echo "=== TRACK B DONE ==="

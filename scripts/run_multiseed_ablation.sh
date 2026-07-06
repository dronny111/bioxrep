#!/usr/bin/env bash
# Multi-seed attention-distillation ablation.
# Teacher is a FIXED input (outputs/arm_teacher, seed 7) shared by all arms and
# all seeds -- we vary ONLY the student seed to measure student-side variance.
# Arms (identical student config, hidden=64/proj=128/epochs=3/bs=128/maxlen=64):
#   baseline    : --attention-distillation-weight 0.0
#   byterule    : --attention-distillation-weight 0.1 (deterministic byte/digit teacher)
#   neural      : --attention-distillation-weight 0.1 --attention-teacher-checkpoint <fixed>
# Seed 13 is already recorded elsewhere; this sweep runs the ADDITIONAL seeds and
# also re-runs 13 so every arm has a self-consistent set under outputs/multiseed/.
set -euo pipefail
cd /Users/ronnypolle/Desktop/generic-numeric-representation-by-attention-distillation-main
export PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE

CLASS_INPUT=data/bioxrep_hgnc_aliases_train_classes.jsonl
VALID_INPUT=data/bioxrep_hgnc_aliases_valid_pairs.jsonl
NOTATIONS=approved_symbol,prev_symbol,alias_name,prev_name,approved_name
HELDOUT=data/bioxrep_hgnc_alias_symbol_heldout.jsonl
TEACHER=outputs/arm_teacher/char_cnn_student.pt

SEEDS=(13 17 23 42 101)
ROOT=outputs/multiseed
mkdir -p $ROOT/eval

train_arm () {
  local arm=$1 seed=$2; shift 2
  local out=$ROOT/${arm}_seed${seed}
  mkdir -p "$out"
  echo "=== TRAIN arm=$arm seed=$seed ==="
  python3 -m bioxrep.train.train_contrastive_student \
    --class-input $CLASS_INPUT --class-notations $NOTATIONS --forms-per-class 4 \
    --valid-input $VALID_INPUT --output-dir "$out" \
    --encoder cnn --hidden-dim 64 --projection-dim 128 --epochs 3 \
    --batch-size 128 --max-length 64 --seed $seed "$@" 2>&1 | tail -2
  echo "=== RETRIEVAL arm=$arm seed=$seed ==="
  python3 -m bioxrep.baselines.student_retrieval \
    --checkpoint "$out/char_cnn_student.pt" --input $HELDOUT --track gene_alias \
    --query-notation alias_symbol --query-split test \
    --candidate-notation approved_symbol --candidate-split train \
    --max-queries 2000 --bootstrap \
    --output $ROOT/eval/retrieval_${arm}_seed${seed}.json 2>&1 | tail -1
  echo "=== INVARIANCE arm=$arm seed=$seed ==="
  python3 -m bioxrep.eval.invariance_ratio \
    --checkpoint "$out/char_cnn_student.pt" --input $HELDOUT --track gene_alias \
    --max-facts 4000 --seed 13 \
    --output $ROOT/eval/invariance_${arm}_seed${seed}.json 2>&1 | tail -1
}

for s in "${SEEDS[@]}"; do
  train_arm baseline $s --attention-distillation-weight 0.0
  train_arm byterule $s --attention-distillation-weight 0.1
  train_arm neural   $s --attention-distillation-weight 0.1 \
    --attention-teacher-checkpoint $TEACHER --attention-teacher-temperature 1.0
done
echo "=== ALL DONE ==="

#!/usr/bin/env bash
# Neural-teacher attention-distillation ablation (arm C3).
# Teacher = higher-capacity char-CNN trained longer on the SAME fact-disjoint
# train split; its learned cross-form attention replaces the byte/digit rule.
# Student config is IDENTICAL to the prior A/B arms (hidden=64, proj=128,
# epochs=3, seed=13) except --attention-teacher-checkpoint.
set -euo pipefail
cd /Users/ronnypolle/Desktop/generic-numeric-representation-by-attention-distillation-main
export PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p outputs/arm_teacher outputs/arm_neural_distill outputs/ab_eval

CLASS_INPUT=data/bioxrep_hgnc_aliases_train_classes.jsonl
VALID_INPUT=data/bioxrep_hgnc_aliases_valid_pairs.jsonl
NOTATIONS=approved_symbol,prev_symbol,alias_name,prev_name,approved_name
HELDOUT=data/bioxrep_hgnc_alias_symbol_heldout.jsonl

echo "=== [1/4] Train NEURAL TEACHER (hidden=128, proj=128, epochs=6, seed=7) ==="
python3 -m bioxrep.train.train_contrastive_student \
  --class-input $CLASS_INPUT --class-notations $NOTATIONS --forms-per-class 4 \
  --valid-input $VALID_INPUT --output-dir outputs/arm_teacher \
  --encoder cnn --hidden-dim 128 --projection-dim 128 --epochs 6 \
  --batch-size 128 --max-length 64 --seed 7 2>&1 | tail -6

echo "=== [2/4] Train STUDENT with NEURAL teacher (weight=0.1, temp=1.0, seed=13) ==="
python3 -m bioxrep.train.train_contrastive_student \
  --class-input $CLASS_INPUT --class-notations $NOTATIONS --forms-per-class 4 \
  --valid-input $VALID_INPUT --output-dir outputs/arm_neural_distill \
  --encoder cnn --hidden-dim 64 --projection-dim 128 --epochs 3 \
  --batch-size 128 --max-length 64 --seed 13 \
  --attention-distillation-weight 0.1 \
  --attention-teacher-checkpoint outputs/arm_teacher/char_cnn_student.pt \
  --attention-teacher-temperature 1.0 2>&1 | tail -8

echo "=== [3/4] Held-out retrieval (neural distill) ==="
python3 -m bioxrep.baselines.student_retrieval \
  --checkpoint outputs/arm_neural_distill/char_cnn_student.pt \
  --input $HELDOUT --track gene_alias \
  --query-notation alias_symbol --query-split test \
  --candidate-notation approved_symbol --candidate-split train \
  --max-queries 2000 --bootstrap \
  --output outputs/ab_eval/retrieval_neural_distill.json 2>&1 | tail -4

echo "=== [4/4] Invariance ratio (neural distill) ==="
python3 -m bioxrep.eval.invariance_ratio \
  --checkpoint outputs/arm_neural_distill/char_cnn_student.pt \
  --input $HELDOUT --track gene_alias \
  --max-facts 4000 --seed 13 \
  --output outputs/ab_eval/invariance_neural_distill.json 2>&1 | tail -6

echo "=== DONE ==="

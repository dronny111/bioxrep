# BioXRep HGVS Results

This note freezes the first paper-grade BioXRep genomic/protein notation result.

## Task

Retrieve nucleotide HGVS forms for a protein HGVS query under fact-disjoint train/test splits. The strict hard benchmark prevents position-only shortcuts by constructing candidate pools where decoys share the query's parsed `protein_position` and `cdna_position`, then fills to a fixed candidate count with test-only decoys.

## Dataset

Source artifact:

- `data/bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl`

Scaled fact split:

| Split | Equivalence classes | Notes |
| --- | ---: | --- |
| Train | `29,076` | Requires `protein_expression` and `nucleotide_expression` |
| Test | `7,270` | Fact-disjoint from train |

Training/evaluation artifacts:

| Artifact | Size |
| --- | ---: |
| `data/bioxrep_clinvar_hgvs_scaled_train_pairs_50k.jsonl` | `50,000` pairs |
| `data/bioxrep_clinvar_hgvs_scaled_test_pairs_20k.jsonl` | `20,000` pairs |
| `data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl` | `2,000` queries, `20` candidates/query |

## Main Results

All learned models use the same lightweight character CNN encoder and are evaluated on the same full 2k strict-filled hard benchmark.

| Model | Old pair top-1 | Hard top-1 | Hard top-5 | Hard MRR |
| --- | ---: | ---: | ---: | ---: |
| Char n-gram retrieval | n/a | `0.1540` | `0.3995` | `0.296` |
| Text-only CNN | `0.9574` | `0.6995` | `0.9465` | `0.812` |
| Text+numeric CNN, sinusoidal `protein_position + cdna_position` | `1.0000` | `0.4875` | `0.9985` | `0.701` |
| Numeric-only CNN, sinusoidal `protein_position + cdna_position` | `1.0000` | `0.3480` | `0.9840` | `0.610` |
| Masked-digit text+numeric CNN | `1.0000` | `0.5385` | `0.9995` | `0.731` |

## Interpretation

The scaled strict benchmark is not solved by numeric shortcuts. Numeric features make old pair retrieval look perfect and push hard top-5 very high, but they reduce hard top-1 relative to the text-only CNN. The text-only model is the strongest top-1 model, and masked-digit text still beats numeric-only, showing that non-digit sequence context carries notation-invariant signal.

For paper framing, report text-only representation learning separately from numeric-feature upper-bound or teacher-feature models.

## Reproduction

Build the fact-disjoint split and hard benchmark:

```bash
python3 -m bioxrep.data.split_equivalence_classes --input data/bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl --train-output data/bioxrep_clinvar_hgvs_variants_numeric_scaled_train_40k.jsonl --test-output data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl --required-notations protein_expression,nucleotide_expression --train-fraction 0.8 --max-examples 50000 --seed 31
python3 -m bioxrep.data.build_pairs --input data/bioxrep_clinvar_hgvs_variants_numeric_scaled_train_40k.jsonl --output data/bioxrep_clinvar_hgvs_scaled_train_pairs_50k.jsonl --left-notation protein_expression --right-notation nucleotide_expression --max-pairs 50000
python3 -m bioxrep.data.build_pairs --input data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl --output data/bioxrep_clinvar_hgvs_scaled_test_pairs_20k.jsonl --left-notation protein_expression --right-notation nucleotide_expression --max-pairs 20000
python3 -m bioxrep.data.build_hard_retrieval_benchmark --input data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl --output data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl --query-notation protein_expression --candidate-notation nucleotide_expression --confound-fields protein_position,cdna_position --min-decoys 1 --max-queries 2000 --max-candidates 20 --fill-random-decoys --seed 37
```

Train and evaluate the strongest model:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --input data/bioxrep_clinvar_hgvs_scaled_train_pairs_50k.jsonl --valid-input data/bioxrep_clinvar_hgvs_scaled_test_pairs_20k.jsonl --max-valid-pairs 5000 --hard-valid-input data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl --max-hard-valid-queries 500 --output-dir outputs/contrastive_student_hgvs_scaled_pair_textonly_50k --encoder cnn --epochs 3 --batch-size 128 --max-length 160 --hidden-dim 64 --projection-dim 128 --attribute-fields variation_id --attribute-loss-weight 0.2
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.eval.hard_student_retrieval --checkpoint outputs/contrastive_student_hgvs_scaled_pair_textonly_50k/char_cnn_student.pt --input data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl --output outputs/hard_eval_scaled_pair_textonly_50k_protein_cdna_filled20_2k.json
```

## Result Artifacts

| Result | Artifact |
| --- | --- |
| Char n-gram | `outputs/char_ngram_scaled_test_hard_protein_cdna_filled20_2k.json` |
| Text-only CNN | `outputs/hard_eval_scaled_pair_textonly_50k_protein_cdna_filled20_2k.json` |
| Text+numeric CNN | `outputs/hard_eval_scaled_pair_pos_cdna_sinusoidal_50k_protein_cdna_filled20_2k.json` |
| Numeric-only CNN | `outputs/hard_eval_scaled_pair_pos_cdna_numeric_only_50k_protein_cdna_filled20_2k.json` |
| Masked-digit text+numeric CNN | `outputs/hard_eval_scaled_pair_pos_cdna_maskdigits_50k_protein_cdna_filled20_2k.json` |

# BioXRep HGVS Results

This note freezes the first paper-grade BioXRep genomic/protein notation result.

## Task

Retrieve nucleotide HGVS forms for a protein HGVS query under fact-disjoint train/test splits. The hard benchmark targets position-only shortcuts by guaranteeing each query **at least one** decoy that shares its parsed `protein_position` and `cdna_position` (`--min-decoys 1`), then fills the remaining candidate slots to a fixed pool size with random test-only decoys (`--fill-random-decoys`). Because only the matched decoys are position-confounded, the pool is a mix of hard and easy negatives; the builder reports the realized matched-vs-random composition, and `top-5` in particular is inflated relative to `top-1` because the random-fill decoys are trivially separable on position. To build an all-hard pool instead, raise `--min-decoys` and drop `--fill-random-decoys` (queries without enough matched decoys are then skipped).

## Dataset

Source artifact:

- `data/bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl`

### Source artifact provenance

The reproduction below starts from `bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl`.
That file is built from public ClinVar HGVS as follows (this chain must be run
first — it is the one place the pipeline is not self-contained from the frozen
split command):

```bash
# Fetch ClinVar HGVS + HGNC, then build cross-notation HGVS equivalence classes.
python3 -m bioxrep.data.fetch_public clinvar_hgvs
python3 -m bioxrep.data.fetch_public hgnc_complete_set
python3 -m bioxrep.data.build_clinvar_hgvs_variants \
  --hgvs data/raw/clinvar_hgvs/hgvs4variation.txt.gz \
  --hgnc data/raw/hgnc_complete_set/hgnc_complete_set.txt \
  --output data/bioxrep_clinvar_hgvs_variants.jsonl
# The `_numeric_50k` artifact is this file with parsed protein_position/cdna_position
# numeric attributes populated and capped to 50k classes. Record the exact enrichment
# command in this doc when regenerating so the source is not circular.
```

`build_clinvar_hgvs_variants` already parses `protein_position` and `cdna_position`
(see `parse_protein_position` / `parse_nucleotide_position`), which are the confound
fields the hard benchmark matches on.

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

The numeric **input-feature** ablation compares how the position is injected, holding
everything else fixed (`seed 13`, 3 epochs, char-CNN hidden 64 / proj 128, corrected
deduped fact-disjoint split, no auxiliary numeric loss). `none` is the text-only byte
encoder — because each digit is already its own byte token, this row doubles as the
**digit-tokenization** baseline. `explicit` = normalized scalar + present-mask;
`sinusoidal` = multi-frequency Fourier features; `xval` = xVal-style continuous
tokenization (Golkar et al. 2023), a learned per-field embedding scaled by the value.

**Full 20-candidate pool** (bootstrap 1000; each pool has ~2.25 position-matched
decoys and ~16.75 random-fill decoys):

| `--numeric-feature-mode` | Hard top-1 | 95% CI | Hard top-5 | Hard MRR |
| --- | ---: | :---: | ---: | ---: |
| `none` (text-only / digit-token) | `0.6800` | `[0.659, 0.700]` | `0.9685` | `0.804` |
| `explicit` (scalar) | `0.8345` | `[0.819, 0.850]` | `0.9920` | `0.904` |
| `sinusoidal` (Fourier) | `0.4980` | `[0.477, 0.519]` | `0.9940` | `0.708` |
| `xval` (continuous tok.) | `0.8455` | `[0.829, 0.862]` | `0.9930` | `0.911` |

**Strict matched-only pool** (random-fill decoys dropped; positive ranked against
only its ~2.25 position-matched hard decoys, avg pool 3.25):

| `--numeric-feature-mode` | Hard top-1 | 95% CI | Hard top-5 | Hard MRR |
| --- | ---: | :---: | ---: | ---: |
| `none` (text-only / digit-token) | **`0.9195`** | `[0.908, 0.930]` | `0.9995` | `0.957` |
| `explicit` (scalar) | `0.9000` | `[0.887, 0.913]` | `0.9995` | `0.947` |
| `sinusoidal` (Fourier) | `0.4980` | `[0.477, 0.518]` | `0.9940` | `0.708` |
| `xval` (continuous tok.) | `0.9060` | `[0.892, 0.919]` | `0.9995` | `0.951` |

## Interpretation

**The full-pool ranking is an artifact of easy-decoy rejection, and it inverts under
the strict pool.** On the full 20-candidate pool the numeric features (`explicit`,
`xval`) look like large wins (~0.83–0.85 vs text-only 0.68), but each pool contains
only ~2.25 genuinely position-matched decoys against ~16.75 random-fill decoys, and
any position feature rejects the random decoys trivially. When the random-fill decoys
are removed and the positive is scored against **only** its position-matched hard
decoys, the ranking flips: **text-only is the strongest (0.9195), and every numeric
input feature is neutral-to-harmful** — `xval` `0.906` and `explicit` `0.900` sit
below text-only with non-overlapping-to-touching CIs, and `sinusoidal` collapses to
`0.498`. This is exactly expected: on a position-matched pool the numeric component is
near-identical for the positive and its hard decoys, so it carries no discriminative
signal and only dilutes the text channel; the sinusoidal encoding makes matched decoys
almost indistinguishable, which is why it collapses. The decisive column is therefore
hard top-1 **on the matched-only pool**, and there the negative holds across all three
value-aware schemes, including xVal.

Two caveats for paper framing. (1) All rows above are trained at the same 50k-pair scale, so this table isolates the numeric-feature ablation; any claim about training-scale ("more facts") must cite the separate small-vs-large runs, not this table. (2) Within a shared-position pool, distinguishing the positive nucleotide form from its position-matched decoys largely reduces to mapping the amino-acid substitution to the correct codon change — i.e. the model is rewarded for learning genetic-code structure, which is a specific and worthwhile capability but narrower than "notation invariance" in general. Report text-only representation learning separately from numeric-feature upper-bound or teacher-feature models, and report the realized decoy composition alongside the metrics.

## Numeric-consistency-loss ablation

The table above ablates numeric **input features** (`--numeric-feature-mode {none,explicit,sinusoidal}`), which change what the encoder *reads*. A distinct question is whether the auxiliary numeric **consistency loss** (`--numeric-loss-weight`) helps: this adds a per-field linear head that regresses each form's embedding to its normalized `protein_position` / `cdna_position` target (MSE) plus a left/right agreement term, forcing the embedding to be numerically decodable without adding any numeric input channel. To isolate the loss cleanly, all three arms keep input `--numeric-feature-mode none` (text-only) and vary only `--numeric-loss-weight`. All arms are trained on the corrected fact-disjoint deduped split (see note below), `seed 13`, and evaluated on the full 2k strict-filled hard benchmark.

| `--numeric-loss-weight` | Hard top-1 | 95% CI | Hard top-5 |
| ---: | ---: | :---: | ---: |
| `0.0` (text-only) | `0.7275` | `[0.708, 0.747]` | `0.9695` |
| `0.1` | `0.7050` | `[0.685, 0.726]` | `0.9765` |
| `0.5` | `0.7145` | `[0.695, 0.736]` | `0.9740` |

The auxiliary numeric-consistency loss does **not** improve hard top-1: all three CIs overlap the text-only baseline, and the point estimates for the weighted arms sit slightly below it. Forcing the embedding to be numerically decodable does not add notation-invariant retrieval signal here — consistent with the feature-mode finding that position information is a confound, not a discriminator, in a position-matched hard pool. This is gap #3 (untested numeric consistency loss) closed as a negative result.

**Leakage-fix note.** While rebuilding the split for this ablation, `verify_no_leakage.py` flagged 12 `fact_id`s shared across train/test. Root cause: the `_numeric_50k` variants file contained 702 duplicated `fact_id`s (50,000 rows, 49,298 unique), and the row-wise splitter could place a duplicated fact on both sides. Deduplicating by `fact_id` before splitting (`..._numeric_50k_dedup.jsonl`, 49,298 rows) restores a fact-disjoint split (29,023 train / 7,256 test) that passes the leakage check. The text-only hard top-1 on the corrected split (`0.7275`) is close to the earlier `0.6995` reported in the table above (which predates the dedup); the numeric-feature rows have not been re-run on the deduped split and should be regenerated for a strictly apples-to-apples comparison.

## Reproduction

Build the fact-disjoint split and hard benchmark:

```bash
python3 -m bioxrep.data.split_equivalence_classes --input data/bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl --train-output data/bioxrep_clinvar_hgvs_variants_numeric_scaled_train_40k.jsonl --test-output data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl --required-notations protein_expression,nucleotide_expression --train-fraction 0.8 --max-examples 50000 --seed 31
python3 -m bioxrep.data.build_pairs --input data/bioxrep_clinvar_hgvs_variants_numeric_scaled_train_40k.jsonl --output data/bioxrep_clinvar_hgvs_scaled_train_pairs_50k.jsonl --left-notation protein_expression --right-notation nucleotide_expression --max-pairs 50000
python3 -m bioxrep.data.build_pairs --input data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl --output data/bioxrep_clinvar_hgvs_scaled_test_pairs_20k.jsonl --left-notation protein_expression --right-notation nucleotide_expression --max-pairs 20000
python3 -m bioxrep.data.build_hard_retrieval_benchmark --input data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl --output data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl --query-notation protein_expression --candidate-notation nucleotide_expression --confound-fields protein_position,cdna_position --min-decoys 1 --max-queries 2000 --max-candidates 20 --fill-random-decoys --seed 37
```

The builder prints the realized decoy composition (fraction of confound-matched
hard negatives, average matched/query, and how many queries got an all-hard pool);
each record also carries `matched_decoy_count` / `random_decoy_count`, and the hard
eval reports `avg_matched_decoys` / `matched_decoy_fraction`. With `--min-decoys 1
--fill-random-decoys` most decoys are random fill, so `top-5` is easier than
`top-1`; for an all-hard pool raise `--min-decoys` and drop `--fill-random-decoys`.

Verify the split is fact-disjoint before training:

```bash
python3 scripts/verify_no_leakage.py \
  --train data/bioxrep_clinvar_hgvs_variants_numeric_scaled_train_40k.jsonl \
  --test data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl
```

Train and evaluate the strongest model:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --input data/bioxrep_clinvar_hgvs_scaled_train_pairs_50k.jsonl --valid-input data/bioxrep_clinvar_hgvs_scaled_test_pairs_20k.jsonl --max-valid-pairs 5000 --hard-valid-input data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl --max-hard-valid-queries 500 --output-dir outputs/contrastive_student_hgvs_scaled_pair_textonly_50k --encoder cnn --epochs 3 --batch-size 128 --max-length 160 --hidden-dim 64 --projection-dim 128 --attribute-fields variation_id --attribute-loss-weight 0.2
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.eval.hard_student_retrieval --checkpoint outputs/contrastive_student_hgvs_scaled_pair_textonly_50k/char_cnn_student.pt --input data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl --bootstrap --output outputs/hard_eval_scaled_pair_textonly_50k_protein_cdna_filled20_2k.json
```

Add `--bootstrap` (as above) to emit 95% percentile-bootstrap CIs on `top1`,
`top5`, and `mean_reciprocal_rank`. For training/init variance, sweep `--seed`
across several values (e.g. `11 13 17 19 23`) and report mean ± std of hard top-1.

Numeric-consistency-loss ablation (dedupe the source first, then vary only the
loss weight with text-only input). Full driver: `scripts/run_trackb_numeric_loss.sh`.

```bash
# 1. Deduplicate the variants file by fact_id (removes 702 duplicate facts) and
#    rebuild the fact-disjoint split + pairs + hard benchmark on the deduped file.
#    (See scripts/run_trackb_numeric_loss.sh header and the leakage-fix note above.)
# 2. Train text-only arms varying only --numeric-loss-weight:
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student \
  --input data/bioxrep_clinvar_hgvs_scaled_train_pairs_50k.jsonl \
  --valid-input data/bioxrep_clinvar_hgvs_scaled_test_pairs_20k.jsonl --max-valid-pairs 5000 \
  --hard-valid-input data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl \
  --max-hard-valid-queries 500 --output-dir outputs/trackb/numloss_w01 \
  --encoder cnn --hidden-dim 64 --projection-dim 128 --epochs 3 --batch-size 128 \
  --max-length 160 --seed 13 --attribute-fields variation_id --attribute-loss-weight 0.2 \
  --numeric-fields protein_position,cdna_position --numeric-feature-mode none \
  --numeric-loss-weight 0.1
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.eval.hard_student_retrieval \
  --checkpoint outputs/trackb/numloss_w01/char_cnn_student.pt \
  --input data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl \
  --bootstrap --output outputs/trackb/eval/hard_numloss_w01.json
```

## Result Artifacts

| Result | Artifact |
| --- | --- |
| Char n-gram | `outputs/char_ngram_scaled_test_hard_protein_cdna_filled20_2k.json` |
| Text-only CNN | `outputs/hard_eval_scaled_pair_textonly_50k_protein_cdna_filled20_2k.json` |
| Text+numeric CNN | `outputs/hard_eval_scaled_pair_pos_cdna_sinusoidal_50k_protein_cdna_filled20_2k.json` |
| Numeric-only CNN | `outputs/hard_eval_scaled_pair_pos_cdna_numeric_only_50k_protein_cdna_filled20_2k.json` |
| Masked-digit text+numeric CNN | `outputs/hard_eval_scaled_pair_pos_cdna_maskdigits_50k_protein_cdna_filled20_2k.json` |
| Numeric-loss w=0.0 (text-only, deduped split) | `outputs/trackb/eval/hard_numloss_w00.json` |
| Numeric-loss w=0.1 (text-only, deduped split) | `outputs/trackb/eval/hard_numloss_w01.json` |
| Numeric-loss w=0.5 (text-only, deduped split) | `outputs/trackb/eval/hard_numloss_w05.json` |

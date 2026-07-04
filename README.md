# BioXRep: Cross-Notation Distillation for Invariant Biological Representation Learning

**BioXRep** learns biomedical representations that recover the underlying
biological *fact* rather than the surface *notation* used to express it. The
same gene, variant, or measurement appears as a symbol, an alias, an HGVS
string, an opaque identifier, a lab value with units, or free text. BioXRep asks
whether alignment and distillation signals across these parallel notations can
teach a model to collapse them onto a shared, notation-invariant representation

---

## Project status (what is and isn't delivered)

This repository is currently a **benchmark-and-baselines** contribution, not yet a
validated method result. Read the claims with that scope:

- **Delivered:** fact-disjoint (and notation-disjoint) retrieval benchmarks for
  HGNC gene aliases and ClinVar HGVS variants; a comparable baseline suite
  (char n-gram lexical floor, SapBERT dense, BioSyn-style hybrid, canonical-teacher
  oracle ceiling); and a lightweight contrastive char-CNN student with a
  supervised-contrastive + optional attribute/numeric-consistency objective.
- **Not yet delivered:** the *teacher attention/embedding distillation* named in the
  title. The current student trains with contrastive and auxiliary losses only — no
  teacher is distilled. "Distillation" is the roadmap direction, not an implemented
  component.
- **Headline method result is negative.** On the protagonist fact- and
  notation-disjoint task (`alias_symbol`), the trained student loses to SapBERT and
  to the lexical floor. The repository therefore does **not** yet contain positive
  evidence for its core hypothesis that cross-notation training beats a general
  token encoder; it establishes the benchmarks and shows where the intuitive small
  model fails. See [`docs/bioxrep_sapbert_baseline.md`](docs/bioxrep_sapbert_baseline.md).

---

## Research thesis

Biological information is expressed through many equivalent notation systems:
DNA/RNA/protein sequences, HGVS variant notation, gene aliases and prior symbols,
database identifiers, clinical laboratory values, units, and free-text
interpretations. BioXRep studies whether **cross-notation distillation** — using a
canonical teacher or structured field as the invariant target — can produce
encoders that represent the underlying object rather than its spelling.

The project is deliberately built around **fact-disjoint evaluation**: test facts
(and, in the hardest splits, entire *notations*) are held out of training, so a
model cannot succeed by memorizing surface forms it has already seen. This
construction is what separates genuine notation invariance from same-form recall,
and it is where several intuitively-strong models are shown to fail.

---

## Results at a glance

### 1. HGNC gene-alias normalization (fact- and notation-disjoint)

Held-out `alias_symbol` queries retrieved against the approved-symbol pool
(~45k candidates). The `alias_symbol` notation is held out for validation.

| Method | MRR |
| --- | ---: | --- |
| Canonical teacher (exact structured match) | **1.000**|
| SapBERT dense | **0.134** |
| Character n-gram (TF–IDF cosine) | **0.077** |
| Contrastive char-CNN student | **0.051** |

**The lightweight distilled student.** A byte-level char-CNN
(~100k params) trained with supervised contrastive loss over 22,364 HGNC
equivalence classes reaches **0.985 in-domain validation top-1**  and **0.051 MRR** on the held-out `alias_symbol` notation,
*below* both the SapBERT dense retriever (0.134) and the char n-gram. 

Off-the-shelf methods trade places by notation: SapBERT dense wins most on the
low-surface-overlap `alias_name` notation (0.316 vs. char n-gram 0.042), while
the BioSyn-style sparse+dense hybrid is strongest on the high-overlap
`prev_symbol` notation (0.250).

### 2. HGVS variant-notation alignment (protein ↔ nucleotide, position-confounded hard pool)

Aligning protein HGVS to nucleotide HGVS on a fact-disjoint split, scored against
hard candidate pools seeded with position-confounded decoys — each query is
guaranteed at least one decoy that shares its parsed `protein_position` and
`cdna_position`, with the remaining slots filled by random test-only decoys to a
fixed pool size (2,000 queries, 20 candidates each). The builder now reports the
realized matched-vs-random decoy composition; top-5 in particular is easier than
top-1 because the random-fill decoys are trivially separable on position, so read
the two columns with that composition in mind.

| Model | Hard top-1 | Hard top-5 | Hard MRR |
| --- | ---: | ---: | ---: |
| **Text-only char-CNN** | **0.6995** | **0.9465** | **0.812** |
| Sinusoidal text+numeric CNN | 0.4875 | 0.9985 | 0.701 |
| Masked-digit text+numeric CNN | 0.5385 | 0.9995 | 0.731 |
| Numeric-only sinusoidal CNN | 0.3480 | 0.9840 | 0.610 |
| Character n-gram baseline | 0.154 | 0.3995 | 0.296 |

**Numeric position features help easy validation but hurt the hard top-1.** On
easy pair validation, adding sinusoidal `protein_position + cdna_position`
features makes retrieval look near-perfect (top-1 ≈ 1.000). But against the
position-confounded hard decoys — which share those exact fields — the numeric
component is identical for the positive and the hard negatives, so it dilutes the
discriminative text signal and the plain text-only CNN takes hard top-1. (All
rows here are trained on the same 50k-pair scale, so this table isolates the
feature ablation, not a training-scale effect; the separate scaling comparison
lives in [`docs/bioxrep_hgvs_results.md`](docs/bioxrep_hgvs_results.md).) Note the
task within a shared-position pool reduces to mapping the amino-acid change to the
correct codon change, so the text-only win is as much learned genetic-code
structure as generic notation invariance. Full ablation logs and interpretation
in [`docs/bioxrep_hgvs_results.md`](docs/bioxrep_hgvs_results.md).

The recurring lesson across both benchmarks: **evaluation design decides the
conclusion.** Fact-disjoint splits and position-confounded hard pools expose gaps
that in-distribution validation hides.

---

## Repository layout

```
bioxrep/
  data/         synthetic generator, form/split builders, public-source fetchers
  baselines/    char n-gram, SapBERT dense, BioSyn hybrid, canonical teacher,
                student flat-pool retrieval, hard-set retrieval
  train/        contrastive student trainer (mean-pool / char-CNN encoders)
  eval/         hard-candidate-set student evaluation
outputs/        result JSONs, trained checkpoints, comparison table + figure
data/           
```

---

## Installation

```bash
conda create -n bioxrep-sapbert python=3.11 -y && conda activate bioxrep-sapbert
pip install torch numpy scikit-learn transformers
```

SapBERT/BioSyn baselines download `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
from the Hugging Face hub on first use; set `HF_HOME=.hf_cache` to cache locally.

---

## Quickstart

Generate the deterministic synthetic benchmark and run the first lexical baseline:

```bash
python3 -m bioxrep.data.generate_synthetic --output data/bioxrep_synth.jsonl --variant-count 40 --seed 13
python3 -m bioxrep.data.prepare_forms --input data/bioxrep_synth.jsonl --output data/bioxrep_synth_forms.jsonl
python3 -m bioxrep.baselines.char_ngram_retrieval --input data/bioxrep_synth.jsonl
```

On all synthetic forms the char n-gram retriever reaches top-1 `0.480`, MRR
`0.529`; on a held-out cDNA-like variant notation it drops to MRR `0.077`, the
first illustration of the notation-transfer gap.

---

## Reproducing the benchmarks

### HGNC gene-alias normalization

```bash
# 1. Fetch HGNC and build real gene-alias equivalence classes
python3 -m bioxrep.data.fetch_public hgnc_complete_set
python3 -m bioxrep.data.build_hgnc_aliases --input data/raw/hgnc_complete_set/hgnc_complete_set.txt --output data/bioxrep_hgnc_aliases.jsonl
python3 -m bioxrep.data.prepare_forms --input data/bioxrep_hgnc_aliases.jsonl --output data/bioxrep_hgnc_aliases_forms.jsonl
python3 -m bioxrep.data.make_splits --input data/bioxrep_hgnc_aliases.jsonl --output data/bioxrep_hgnc_alias_symbol_heldout.jsonl --track gene_alias --heldout-notation alias_symbol

# 2. Off-the-shelf comparison (char n-gram, SapBERT dense, BioSyn hybrid, canonical teacher) + table + figure
HF_HOME=.hf_cache OMP_NUM_THREADS=8 PYTHONPATH=. python3 scripts/run_hgnc_alias_comparison.py

# 3. Build the fact-disjoint training classes (exclude alias_symbol test genes) and verify no leakage
python3 -m bioxrep.data.filter_equivalence_classes \
  --input data/bioxrep_hgnc_aliases.jsonl --reference data/bioxrep_hgnc_alias_symbol_heldout.jsonl \
  --output data/bioxrep_hgnc_aliases_train_classes.jsonl --mode exclude
python3 scripts/verify_no_leakage.py \
  --train data/bioxrep_hgnc_aliases_train_classes.jsonl \
  --test data/bioxrep_hgnc_alias_symbol_heldout.jsonl --test-split test --heldout-notation alias_symbol

# 4. Train the char-CNN student and score it on the same flat pool (negative result)
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONPATH=. python3 -m bioxrep.train.train_contrastive_student \
  --class-input data/bioxrep_hgnc_aliases_train_classes.jsonl \
  --class-notations approved_symbol,prev_symbol,alias_name,prev_name,approved_name \
  --forms-per-class 4 --valid-input data/bioxrep_hgnc_aliases_valid_pairs.jsonl \
  --output-dir outputs/contrastive_student_hgnc_alias \
  --encoder cnn --hidden-dim 64 --projection-dim 128 --epochs 3 --batch-size 128 --max-length 64 --seed 13
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONPATH=. python3 -m bioxrep.baselines.student_retrieval \
  --checkpoint outputs/contrastive_student_hgnc_alias/char_cnn_student.pt \
  --input data/bioxrep_hgnc_alias_symbol_heldout.jsonl --track gene_alias \
  --query-notation alias_symbol --query-split test --candidate-notation approved_symbol --candidate-split train \
  --max-queries 2000 --bootstrap --output outputs/student_hgnc_alias_symbol_heldout.json
```

Pass `--bootstrap` to any retrieval command to add 95% bootstrap CIs, and sweep
`--seed` on the student for mean ± std. Run `scripts/verify_no_leakage.py`
whenever you rebuild a split — it fails loudly on fact or held-out-notation leakage.

Artifact sizes: `44,997` gene-alias classes → `387,123` flattened forms;
the `alias_symbol` holdout writes `342,405` train and `44,718` test rows.

### HGVS protein↔nucleotide alignment

The full HGVS pipeline — building ClinVar HGVS classes, fact-disjoint
position holdouts, strict position-matched hard candidate pools, and the scaled
text-only / numeric-feature ablation matrix. Headline
reproduction:

```bash
# Scaled fact-disjoint split -> pairs -> strict hard pool -> text-only CNN.
# First build data/bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl from ClinVar HGVS,
# as shown in docs/bioxrep_hgvs_results.md.
python3 -m bioxrep.data.split_equivalence_classes --input data/bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl \
  --train-output data/bioxrep_clinvar_hgvs_variants_numeric_scaled_train_40k.jsonl \
  --test-output data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl \
  --required-notations protein_expression,nucleotide_expression --train-fraction 0.8 --max-examples 50000 --seed 31
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student \
  --input data/bioxrep_clinvar_hgvs_scaled_train_pairs_50k.jsonl \
  --valid-input data/bioxrep_clinvar_hgvs_scaled_test_pairs_20k.jsonl \
  --hard-valid-input data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl \
  --output-dir outputs/contrastive_student_hgvs_scaled_pair_textonly_50k \
  --encoder cnn --epochs 3 --batch-size 128 --max-length 160 --hidden-dim 64 --projection-dim 128 \
  --attribute-fields variation_id --attribute-loss-weight 0.2
```

`KMP_DUPLICATE_LIB_OK=TRUE` is a local macOS workaround for duplicate OpenMP
runtime initialization in this PyTorch environment.

### Additional notation bridges

The repository also builds ClinVar AlleleID↔gene bridges and ClinVar
variant-summary clinical-annotation classes (opaque-identifier and
clinical-significance notations), plus a credentialed MIMIC-IV lab-value track
for unit/reference-range invariance.

---

## Related work

BioXRep sits at the intersection of four literatures.

- **Numeracy in language models.** *Do NLP Models Know Numbers? Probing Numeracy
  in Embeddings* (Wallace et al., EMNLP 2019) shows standard embeddings capture
  numeric structure only partially; *xVal: A Continuous Numerical Tokenization for
  Scientific Language Models* (Golkar et al., 2023) motivates magnitude-aware
  numeric encoding — directly relevant to BioXRep's position and lab-value
  handling.
- **Multi-view / contrastive representation learning.** SimCLR (Chen et al., 2020)
  and Supervised Contrastive Learning (Khosla et al., 2020) underpin the
  equivalence-class contrastive objective; CLIP (Radford et al., 2021) motivates
  treating distinct notations as aligned views of one object.
- **Biomedical entity normalization.** BioSyn (Sung et al., ACL 2020) and SapBERT
  (Liu et al., NAACL 2021) are the strongest off-the-shelf baselines for
  alias-heavy retrieval and are benchmarked directly here.
- **Variant and clinical normalization.** tmVar 1–3 (Wei et al., 2013/2018/2022)
  and the GA4GH Variation Representation Specification (Wagner et al., Cell
  Genomics 2021) frame variant equivalence as a canonicalization problem; clinical
  foundation models — Med-BERT, BEHRT, CLMBR, G-BERT — motivate the EHR lab-value
  track.

---
## Reproducibility notes

- `data/`, `data/raw/`, and `.hf_cache/` are git-ignored and fully regenerable
  from the commands above; `data/raw/fetch_manifest.json` records fetched source
  keys, URLs, paths, timestamps, and byte counts.
- MIMIC-IV requires credentialed PhysioNet access (v3.1 training + data use
  agreement); credentials are read from `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD`
  or a git-ignored `.env` and are never stored by the code.
- Frozen, paper-grade result tables live under `docs/`; `outputs/` holds the
  regenerable result JSONs, checkpoints, comparison table, and figure. `outputs/`
  is git-ignored, so the frozen tables are not currently backed by committed
  artifacts — for an auditable paper release, commit the small result JSONs (the
  metric files, not checkpoints) so each reported number is traceable to a run.
- `scripts/verify_no_leakage.py` asserts fact- and notation-disjointness of any
  train/test construction; run it after rebuilding splits. All retrieval commands
  accept `--bootstrap` for 95% percentile-bootstrap CIs, and the learned student
  accepts `--seed` for multi-seed mean ± std reporting.

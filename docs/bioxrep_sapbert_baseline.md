# BioXRep SapBERT / BioSyn Baseline

This note freezes a learned biomedical-entity baseline for BioXRep and compares it against the existing lexical baseline, a trained BioXRep char-CNN student, and a structured upper bound on alias-heavy HGNC gene-symbol normalization.

## Motivation

The char n-gram baseline scores surface similarity. It is strong when a query notation shares substrings with the approved symbol (e.g. `prev_symbol`) and weak when it does not (e.g. `alias_name`, which is free-text description). SapBERT is a biomedical entity encoder trained by self-alignment over UMLS synonyms, so it is the natural learned reference point for "does a general biomedical embedding already solve notation-invariant gene normalization?" BioSyn adds a sparse+dense hybrid, which is the standard way that surface and semantic signal are combined for biomedical entity linking. Both are reported here against the same pool as the lexical baseline and the canonical-teacher upper bound.

## Task

Retrieve the `approved_symbol` for an alias-form query, over HGNC gene-alias equivalence classes, under fact-disjoint splits. Relevance is defined by the shared `fact_id` (the gene). Three query notations are evaluated against a single shared candidate pool of approved symbols:

- `alias_symbol` (held-out test split) — the fact-disjoint generalization task.
- `prev_symbol` — former official symbols; high surface overlap with approved symbols.
- `alias_name` — free-text descriptive names; low surface overlap.

## Dataset

Source artifact:

- `data/bioxrep_hgnc_alias_symbol_heldout.jsonl`

| Field | Value |
| --- | ---: |
| Candidate pool (`approved_symbol`, train split) | `44,997` |
| Queries per task (capped) | `2,000` |
| Held-out `alias_symbol` test rows available | `44,718` |

All four methods are scored on the identical `(queries, candidate pool)` for each task, so the numbers are directly comparable within a task.

## Methods

| Method | Score |
| --- | --- |
| Char n-gram (lexical) | Cosine of character 3-gram TF-IDF vectors (repo baseline formula). |
| SapBERT dense | Cosine of L2-normalized `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` CLS embeddings. |
| BioSyn hybrid | `(1 - λ)·sparse_cosine + λ·dense_cosine`, `λ = 0.7`. |
| BioXRep char-CNN student | Cosine of L2-normalized byte-level char-CNN embeddings, trained by supervised contrastive loss over HGNC equivalence classes. |
| Canonical teacher | Exact match on the structured `symbol` attribute (oracle canonicalizer). |

SapBERT is CLS-pooled, `max_length = 32`, batch size `128`, run CPU-only at ~77 texts/s. The candidate pool is encoded once and cached; each task then encodes only its queries.

The BioXRep student is a lightweight byte-level char-CNN (hidden 64, projection 128, ~100k parameters) trained for 3 epochs with a supervised contrastive loss (temperature 0.07) that pulls same-gene forms together. Training uses class-aware batching over `22,364` HGNC equivalence classes with the `alias_symbol` test gene `fact_id`s excluded. The `alias_symbol` notation itself is **entirely absent** from the training forms — the student only ever sees `approved_symbol`, `prev_symbol`, `alias_name`, `prev_name`, and `approved_name` — so `alias_symbol` retrieval is a strict held-out-notation generalization test. It is scored on the same flat candidate pool as the other methods via `bioxrep.baselines.student_retrieval`.

## Main Results

MRR / top-1 / top-5 over `2,000` queries against the `44,997`-symbol pool.

| Task | Method | top-1 | top-5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| `alias_symbol` heldout | Char n-gram | `0.047` | `0.108` | `0.077` |
| `alias_symbol` heldout | SapBERT dense | `0.080` | `0.190` | `0.134` |
| `alias_symbol` heldout | BioSyn hybrid (λ=0.7) | `0.077` | `0.182` | `0.129` |
| `alias_symbol` heldout | **BioXRep char-CNN student** | `0.035` | `0.064` | `0.051` |
| `alias_symbol` heldout | Canonical teacher | `1.000` | `1.000` | `1.000` |
| `prev_symbol` | Char n-gram | `0.146` | `0.276` | `0.205` |
| `prev_symbol` | SapBERT dense | `0.140` | `0.297` | `0.219` |
| `prev_symbol` | BioSyn hybrid (λ=0.7) | `0.175` | `0.325` | `0.250` |
| `prev_symbol` | BioXRep char-CNN student ⁽ᵗ⁾ | `0.118` | `0.203` | `0.158` |
| `prev_symbol` | Canonical teacher | `1.000` | `1.000` | `1.000` |
| `alias_name` | Char n-gram | `0.026` | `0.056` | `0.042` |
| `alias_name` | SapBERT dense | `0.234` | `0.404` | `0.316` |
| `alias_name` | BioSyn hybrid (λ=0.7) | `0.225` | `0.398` | `0.308` |
| `alias_name` | BioXRep char-CNN student ⁽ᵗ⁾ | `0.026` | `0.053` | `0.042` |
| `alias_name` | Canonical teacher | `1.000` | `1.000` | `1.000` |

⁽ᵗ⁾ **train-seen** — the student saw this query notation during training, so these two rows are in-distribution measurements, not held-out generalization. Only `alias_symbol` is a fact- and notation-disjoint test for the student.

![BioXRep student vs char n-gram, SapBERT dense, and BioSyn hybrid MRR across three HGNC alias notations, with canonical-teacher oracle off-scale at 1.00](../outputs/hgnc_alias_comparison_figure.png)

## Statistical reporting

The numbers above are single-run point estimates over `2,000` queries. For
paper-grade reporting, add `--bootstrap` to any of the retrieval commands
(`char_ngram_retrieval`, `sapbert_retrieval`, `student_retrieval`) to emit
percentile bootstrap 95% CIs (`{top1,top5,mean_reciprocal_rank}_ci_{low,high}`),
and re-run the learned student across several `--seed` values to report mean ± std,
e.g.:

```bash
for seed in 11 13 17 19 23; do
  HF_HUB_OFFLINE=1 PYTHONPATH=. python3 -m bioxrep.train.train_contrastive_student \
    --class-input data/bioxrep_hgnc_aliases_train_classes.jsonl \
    --class-notations approved_symbol,prev_symbol,alias_name,prev_name,approved_name \
    --forms-per-class 4 --valid-input data/bioxrep_hgnc_aliases_valid_pairs.jsonl \
    --output-dir outputs/contrastive_student_hgnc_alias_seed${seed} \
    --encoder cnn --hidden-dim 64 --projection-dim 128 --epochs 3 --batch-size 128 --max-length 64 --seed ${seed}
  HF_HUB_OFFLINE=1 PYTHONPATH=. python3 -m bioxrep.baselines.student_retrieval \
    --checkpoint outputs/contrastive_student_hgnc_alias_seed${seed}/char_cnn_student.pt \
    --input data/bioxrep_hgnc_alias_symbol_heldout.jsonl --track gene_alias \
    --query-notation alias_symbol --query-split test --candidate-notation approved_symbol --candidate-split train \
    --max-queries 2000 --bootstrap --output outputs/student_hgnc_alias_symbol_heldout_seed${seed}.json
done
```

Because the candidate pool encodes deterministically, bootstrap CIs capture
query-sampling variance; the seed sweep captures training/init variance.

## Interpretation

The gap between query notations is the story. On `alias_name`, where surface overlap with the approved symbol is minimal, SapBERT dense lifts MRR from `0.042` (lexical) to `0.316` — roughly a 7.5x improvement — because the semantic encoder recognizes descriptive names that share no characters with the symbol. On `prev_symbol`, where surface overlap is high, the lexical baseline is already competitive (`0.205` MRR) and the dense-only gain is small; here the BioSyn hybrid is best (`0.250` MRR), confirming that sparse and dense signal are complementary when both are informative.

On the fact-disjoint `alias_symbol` held-out task all learned and lexical methods remain low (MRR ≤ `0.134`): short alias symbols carry little semantic context and often no shared substring, so neither a general biomedical embedding nor character overlap resolves them reliably. This is the task where a BioXRep-trained notation-invariant encoder has the most headroom to demonstrate value.

The canonical teacher is a structured oracle: it matches on the HGNC `symbol` attribute and therefore scores `1.000` everywhere by construction. It is an upper bound on what perfect canonicalization achieves, not a learned competitor, and is reported to bracket the achievable range.

### BioXRep char-CNN student: a negative result on the protagonist task

The trained student does **not** beat SapBERT dense on the fact-disjoint `alias_symbol` task — the task it was added to contest. It scores `0.051` MRR, below both SapBERT dense (`0.134`) and the char n-gram lexical floor (`0.077`). This is a clean negative result and is reported as such.

Two factors explain it. First, `alias_symbol` is a held-out **notation**, not just held-out facts: the student never sees a single `alias_symbol` form in training, so it must transfer from other short-symbol notations (`approved_symbol`, `prev_symbol`) whose surface statistics only partly overlap. Second, a ~100k-parameter byte-level char-CNN encodes little beyond character composition; on short alias symbols that share few characters with the approved symbol, it has neither the lexical precision of TF-IDF n-grams nor the biomedical semantics SapBERT acquired from UMLS self-alignment. The in-domain validation top-1 of `0.985` on symbol↔name pairs confirms the model trained correctly — but note this is an **in-batch** retrieval number (each left form ranked against the ~128 right forms in its batch), not retrieval against the 45k-symbol pool, so it is not directly comparable to the `0.051` pool MRR. It simply confirms the model learned a same-class similarity that does not transfer to unseen short-symbol notations against a 45k-way pool.

The two `train-seen` rows are consistent with this reading: even where the query notation was in the training distribution (`prev_symbol` MRR `0.158`, `alias_name` MRR `0.042`), the small char encoder still trails the off-the-shelf baselines. It carries some lexical signal on `prev_symbol` (which overlaps approved symbols) but collapses to the lexical floor on the free-text `alias_name`, where it has no semantic capacity.

For paper framing: report SapBERT dense and the BioSyn hybrid as off-the-shelf learned baselines, the char n-gram as the lexical floor, and the canonical teacher as the structured ceiling. The char-CNN student result establishes that a lightweight surface-only encoder is **not** sufficient to beat a general biomedical embedding on fact-disjoint alias normalization; closing the `alias_symbol` gap needs either a semantically pretrained backbone, distillation from the SapBERT/teacher signal, or exposure to the target notation — directions the current student deliberately does not use.

## Reproduction

Dense SapBERT retrieval on the held-out alias-symbol task:

```bash
HF_HOME=.hf_cache OMP_NUM_THREADS=8 python3 -m bioxrep.baselines.sapbert_retrieval --input data/bioxrep_hgnc_alias_symbol_heldout.jsonl --track gene_alias --query-split test --candidate-split train --candidate-notation approved_symbol --max-queries 2000 --output outputs/sapbert_dense_hgnc_alias_symbol_heldout.json
```

BioSyn-style dense+sparse hybrid:

```bash
HF_HOME=.hf_cache OMP_NUM_THREADS=8 python3 -m bioxrep.baselines.sapbert_retrieval --input data/bioxrep_hgnc_alias_symbol_heldout.jsonl --track gene_alias --query-split test --candidate-split train --candidate-notation approved_symbol --max-queries 2000 --scoring hybrid --hybrid-lambda 0.7 --output outputs/biosyn_hybrid_hgnc_alias_symbol_heldout.json
```

Regenerate the full four-method comparison (all three tasks, shared pool, table + figure):

```bash
HF_HOME=.hf_cache OMP_NUM_THREADS=8 PYTHONPATH=. python3 scripts/run_hgnc_alias_comparison.py
```

The `prev_symbol` and `alias_name` runs use `--query-notation prev_symbol` / `--query-notation alias_name` with `--query-split train` against the same candidate pool.

### BioXRep char-CNN student

Build the fact-disjoint training class file (excludes the `alias_symbol` test gene `fact_id`s), then train the class-aware char-CNN student:

```bash
# 1) filter equivalence classes to EXCLUDE the alias_symbol test facts.
#    The reference file is the held-out test split; --mode exclude drops any class
#    whose fact_id appears there, leaving a fact-disjoint training pool.
python3 -m bioxrep.data.filter_equivalence_classes \
  --input data/bioxrep_hgnc_aliases.jsonl \
  --reference data/bioxrep_hgnc_alias_symbol_heldout.jsonl \
  --output data/bioxrep_hgnc_aliases_train_classes.jsonl \
  --mode exclude
#    (also build a small left/right valid-pairs file for pair-loss monitoring)

# 1b) VERIFY no fact or notation leakage before training (fails loudly if either leaks).
#     alias_symbol must be absent from the training classes and present in the test split.
python3 scripts/verify_no_leakage.py \
  --train data/bioxrep_hgnc_aliases_train_classes.jsonl \
  --test data/bioxrep_hgnc_alias_symbol_heldout.jsonl \
  --test-split test --heldout-notation alias_symbol

# 2) train
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONPATH=. python3 -m bioxrep.train.train_contrastive_student \
  --class-input data/bioxrep_hgnc_aliases_train_classes.jsonl \
  --class-notations approved_symbol,prev_symbol,alias_name,prev_name,approved_name \
  --forms-per-class 4 \
  --valid-input data/bioxrep_hgnc_aliases_valid_pairs.jsonl \
  --output-dir outputs/contrastive_student_hgnc_alias \
  --encoder cnn --hidden-dim 64 --projection-dim 128 \
  --epochs 3 --batch-size 128 --max-length 64 --text-transform none --seed 13
```

Score the trained student on the shared flat pool (mirrors the SapBERT adapter's CLI and output schema):

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONPATH=. python3 -m bioxrep.baselines.student_retrieval \
  --checkpoint outputs/contrastive_student_hgnc_alias/char_cnn_student.pt \
  --input data/bioxrep_hgnc_alias_symbol_heldout.jsonl \
  --track gene_alias --query-notation alias_symbol --query-split test \
  --candidate-notation approved_symbol --candidate-split train \
  --max-queries 2000 --output outputs/student_hgnc_alias_symbol_heldout.json
```

The `prev_symbol` / `alias_name` student runs swap `--query-notation` and use `--query-split train` (these are **train-seen** measurements — see the results-table footnote).

## Result Artifacts

| Result | Artifact |
| --- | --- |
| Comparison summary (all runs) | `outputs/hgnc_alias_comparison_summary.json` |
| Comparison table | `outputs/hgnc_alias_comparison_table.csv`, `outputs/hgnc_alias_comparison_table.md` |
| Comparison figure | `outputs/hgnc_alias_comparison_figure.png` |
| SapBERT dense (per task) | `outputs/sapbert_dense_hgnc_{alias_symbol_heldout,prev_symbol,alias_name}.json` |
| BioSyn hybrid (per task) | `outputs/biosyn_hybrid_hgnc_{alias_symbol_heldout,prev_symbol,alias_name}.json` |
| Char n-gram (per task) | `outputs/char_ngram_hgnc_{alias_symbol_heldout,prev_symbol,alias_name}.json` |
| Canonical teacher (per task) | `outputs/canonical_teacher_hgnc_{alias_symbol_heldout,prev_symbol,alias_name}.json` |
| BioXRep char-CNN student (per task) | `outputs/student_hgnc_{alias_symbol_heldout,prev_symbol,alias_name}.json` |
| Student checkpoint + train metrics | `outputs/contrastive_student_hgnc_alias/char_cnn_student.pt`, `outputs/contrastive_student_hgnc_alias/metrics.json` |
| Student flat-pool adapter | `bioxrep/baselines/student_retrieval.py` |

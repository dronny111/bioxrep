# BioXRep: Cross-Notation Distillation for Invariant Biological Representation Learning

This repository began as a 2021 research prototype on universal numerical representation learning by attention distillation from transliteration teachers. The project is now being reframed as **BioXRep**, a biomedical representation learning project focused on cross-notation invariance.

## Research Thesis

Biological information is often expressed through multiple equivalent notation systems: DNA/RNA/protein sequences, variant notation, gene aliases, clinical laboratory values, units, normalized measurements, and free-text interpretations. BioXRep studies whether alignment and distillation signals across these parallel systems can teach models to represent the underlying biological fact rather than its surface form.

## Focused Literature

BioXRep sits closest to four adjacent literature areas: numeracy in language models, multi-view contrastive learning, biomedical entity normalization, and variant or clinical normalization. There is not yet a single settled SOTA for the exact BioXRep setting, but the papers below define the strongest neighboring design patterns.

### Numeracy and scientific text

- **xVal: A Continuous Number Encoding for Large Language Models**: motivates explicit magnitude-aware number representations rather than relying only on subword tokens.
- **Language Models Do Not Embed Numbers Continuously**: shows that ordinary token embeddings do not preserve numeric structure well, which supports adding numeric losses or numeric encoders for positions, values, and unit conversions.
- **NumericBench** and related numeracy benchmarks: provide useful evaluation patterns for held-out numeric ranges, formatting shifts, and quantitative robustness.

### Multi-view and contrastive representation learning

- **SimCLR**: reinforces that strong positive pairs and view construction matter more than architectural scale alone.
- **Supervised Contrastive Learning**: is especially relevant when many surface forms map to one biological fact, which fits the equivalence-class structure used in BioXRep.
- **CLIP** and later multi-view alignment work: support treating protein HGVS, nucleotide HGVS, aliases, IDs, and clinical text as distinct aligned views of the same underlying object.

### Biomedical entity normalization

- **BioSyn**: a strong biomedical synonym-normalization baseline that uses dense retrieval and synonym supervision.
- **SapBERT**: one of the clearest demonstrations that metric learning over synonyms and ontology-linked names yields robust biomedical concept representations.

These papers are the most relevant baselines for HGNC alias normalization and other name-heavy retrieval tasks in this repository.

### Variant and clinical normalization

- **tmVar / tmVar 2.0** and related variant-normalization systems: show that exact variant identity is still best handled with structured parsing and normalization rather than text similarity alone.
- **GA4GH variation representation and normalization** work: clarifies that variant equivalence is a canonicalization problem before it becomes a representation-learning problem.
- **Med-BERT**, **BEHRT**, **CLMBR**, **G-BERT**, and related clinical foundation model work: provide strong EHR representation baselines, while also highlighting that unit-invariant lab semantics remain underexplored.

### What this means for BioXRep

The literature points toward a hybrid structured-neural approach rather than a generic text encoder alone. The most justified next model combines:

- multi-view contrastive training over full equivalence classes,
- structured teacher signals from canonical IDs or parsed biomedical fields,
- auxiliary attribute prediction to preserve gene, variant, residue, position, unit, and value information,
- explicit numeric handling for quantities and positions,
- evaluation under held-out notation, held-out units, and held-out numeric ranges.

In other words, the literature supports improving the supervision and objective before simply scaling the encoder.

## Initial Tracks

- **Genomic/protein notation invariance:** learn shared representations across protein substitutions, HGVS-like notation, gene/transcript references, and textual variant descriptions.
- **Clinical/EHR lab-value invariance:** learn representations that preserve clinical quantity and interpretation across units, reference ranges, and text descriptions.

## Current Artifacts

- `docs/bioxrep_research_brief.md`: research framing, paper thesis, hypotheses, and method sketch.
- `docs/bioxrep_experiment_roadmap.md`: staged experiment plan from synthetic benchmark to real biomedical validation.
- `docs/bioxrep_hgvs_results.md`: frozen paper-grade HGVS result table, artifacts, and interpretation.
- `bioxrep/data/generate_synthetic.py`: deterministic generator for synthetic BioXRep equivalence classes.
- `data/bioxrep_synth.jsonl`: initial generated benchmark sample.

## Generate the Synthetic Benchmark

```bash
python3 -m bioxrep.data.generate_synthetic --output data/bioxrep_synth.jsonl --variant-count 40 --seed 13
```

## Prepare Forms and Splits

```bash
python3 -m bioxrep.data.prepare_forms --input data/bioxrep_synth.jsonl --output data/bioxrep_synth_forms.jsonl
python3 -m bioxrep.data.make_splits --input data/bioxrep_synth.jsonl --output data/bioxrep_split_cdna_heldout.jsonl --track variant --heldout-notation cdna_hgvs_like
python3 -m bioxrep.data.make_splits --input data/bioxrep_synth.jsonl --output data/bioxrep_split_lab_alt_unit_heldout.jsonl --track lab --heldout-notation alternate_unit
python3 -m bioxrep.data.make_splits --input data/bioxrep_synth.jsonl --output data/bioxrep_synth_protein_position_600plus_heldout.jsonl --track variant --heldout-numeric-range protein_position 600 1200
```

`make_splits` also supports held-out numeric ranges, which is useful for position and value generalization checks. For example, the synthetic variant split above writes `161` train rows and `95` held-out protein-position test rows.

## Run the First Retrieval Baseline

```bash
python3 -m bioxrep.baselines.char_ngram_retrieval --input data/bioxrep_synth.jsonl
python3 -m bioxrep.baselines.char_ngram_retrieval --input data/bioxrep_split_cdna_heldout.jsonl --track variant --query-split test --candidate-split train
python3 -m bioxrep.baselines.char_ngram_retrieval --input data/bioxrep_split_lab_alt_unit_heldout.jsonl --track lab --query-split test --candidate-split train --candidate-notation canonical_unit
```

Initial character n-gram retrieval results:

- All forms: top-1 `0.480`, top-5 `0.555`, MRR `0.529`.
- Held-out cDNA-like variant notation to train variant forms: top-1 `0.050`, top-5 `0.050`, MRR `0.077`.
- Held-out lab alternate-unit forms to canonical-unit forms: top-1 `0.571`, top-5 `0.929`, MRR `0.730`.

## Fetch Public Biomedical Sources

List known public sources:

```bash
python3 -m bioxrep.data.fetch_public --list
```

Preview planned downloads:

```bash
python3 -m bioxrep.data.fetch_public hgnc_complete_set clinvar_allele_gene --dry-run
```

Fetch a small first public source:

```bash
python3 -m bioxrep.data.fetch_public hgnc_complete_set
```

Fetch ClinVar sources explicitly when ready:

```bash
python3 -m bioxrep.data.fetch_public clinvar_variant_summary clinvar_hgvs clinvar_allele_gene
```

Downloaded files are written under `data/raw/`, which is ignored by git. A `data/raw/fetch_manifest.json` file records fetched source keys, URLs, paths, timestamps, and byte counts.

Build real HGNC gene-alias equivalence classes:

```bash
python3 -m bioxrep.data.build_hgnc_aliases --input data/raw/hgnc_complete_set/hgnc_complete_set.txt --output data/bioxrep_hgnc_aliases.jsonl
python3 -m bioxrep.data.prepare_forms --input data/bioxrep_hgnc_aliases.jsonl --output data/bioxrep_hgnc_aliases_forms.jsonl
python3 -m bioxrep.data.make_splits --input data/bioxrep_hgnc_aliases.jsonl --output data/bioxrep_hgnc_alias_symbol_heldout.jsonl --track gene_alias --heldout-notation alias_symbol
```

Initial HGNC real-data artifact sizes:

- `data/bioxrep_hgnc_aliases.jsonl`: 44,997 gene-alias equivalence classes.
- `data/bioxrep_hgnc_aliases_forms.jsonl`: 387,123 flattened forms.
- `data/bioxrep_hgnc_alias_symbol_heldout.jsonl`: 342,405 train rows and 44,718 held-out alias-symbol test rows.

Initial HGNC char n-gram smoke result:

```bash
python3 -m bioxrep.baselines.char_ngram_retrieval --input data/bioxrep_hgnc_alias_symbol_heldout.jsonl --track gene_alias --query-split test --candidate-split train --candidate-notation approved_symbol --max-queries 200
```

- Held-out alias-symbol to approved-symbol retrieval: top-1 `0.050`, top-5 `0.085`, MRR `0.070` over 200 queries.

Learned baseline (SapBERT dense and BioSyn-style dense+sparse hybrid) on the same HGNC alias tasks:

```bash
HF_HOME=.hf_cache OMP_NUM_THREADS=8 python3 -m bioxrep.baselines.sapbert_retrieval --input data/bioxrep_hgnc_alias_symbol_heldout.jsonl --track gene_alias --query-split test --candidate-split train --candidate-notation approved_symbol --max-queries 2000 --output outputs/sapbert_dense_hgnc_alias_symbol_heldout.json
HF_HOME=.hf_cache OMP_NUM_THREADS=8 python3 -m bioxrep.baselines.sapbert_retrieval --input data/bioxrep_hgnc_alias_symbol_heldout.jsonl --track gene_alias --query-split test --candidate-split train --candidate-notation approved_symbol --max-queries 2000 --scoring hybrid --hybrid-lambda 0.7 --output outputs/biosyn_hybrid_hgnc_alias_symbol_heldout.json
```

Comparison over `2,000` queries against the `44,997`-symbol pool (MRR): SapBERT dense beats the char n-gram lexical baseline most on the low-surface-overlap `alias_name` notation (`0.316` vs `0.042`), the BioSyn hybrid is best on the high-overlap `prev_symbol` notation (`0.250`), and all learned/lexical methods stay low on the fact-disjoint held-out `alias_symbol` task (≤ `0.134`) against a canonical-teacher upper bound of `1.000`. Regenerate the full off-the-shelf comparison (char n-gram, SapBERT dense, BioSyn hybrid, canonical teacher), table, and figure with:

```bash
HF_HOME=.hf_cache OMP_NUM_THREADS=8 PYTHONPATH=. python3 scripts/run_hgnc_alias_comparison.py
```

We also trained a lightweight BioXRep char-CNN student (byte-level CNN, hidden `64` / projection `128`, ~100k params, supervised contrastive over `22,364` HGNC equivalence classes with the `alias_symbol` test facts excluded) and scored it on the same flat pool via `bioxrep.baselines.student_retrieval`. **This is a negative result**: on the fact- and notation-disjoint `alias_symbol` task the student reaches only `0.051` MRR — below both SapBERT dense (`0.134`) and the char n-gram lexical floor (`0.077`). The student never sees `alias_symbol` forms in training (held-out *notation*, not just held-out facts), and a small surface-only encoder has neither n-gram lexical precision nor SapBERT's biomedical semantics; its in-domain validation top-1 of `0.985` confirms it trained correctly but does not transfer. Reproduce:

```bash
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
  --max-queries 2000 --output outputs/student_hgnc_alias_symbol_heldout.json
```

See `docs/bioxrep_sapbert_baseline.md` for the full five-method table, figure, and interpretation.

Build a ClinVar allele-to-gene bridge joined with HGNC:

```bash
python3 -m bioxrep.data.fetch_public clinvar_allele_gene
python3 -m bioxrep.data.build_clinvar_gene_bridge --clinvar-allele-gene data/raw/clinvar_allele_gene/allele_gene.txt.gz --hgnc data/raw/hgnc_complete_set/hgnc_complete_set.txt --output data/bioxrep_clinvar_gene_bridge_100k.jsonl --max-examples 100000
python3 -m bioxrep.data.prepare_forms --input data/bioxrep_clinvar_gene_bridge_100k.jsonl --output data/bioxrep_clinvar_gene_bridge_100k_forms.jsonl
python3 -m bioxrep.data.make_splits --input data/bioxrep_clinvar_gene_bridge_100k.jsonl --output data/bioxrep_clinvar_gene_bridge_100k_allele_heldout.jsonl --heldout-notation clinvar_allele_id
```

Initial ClinVar-HGNC bridge artifact sizes:

- `data/bioxrep_clinvar_gene_bridge_100k.jsonl`: 100,000 allele-gene equivalence classes.
- `data/bioxrep_clinvar_gene_bridge_100k_forms.jsonl`: 1,275,281 flattened forms.
- `data/bioxrep_clinvar_gene_bridge_100k_allele_heldout.jsonl`: 1,175,281 train rows and 100,000 held-out ClinVar AlleleID test rows.

Initial ClinVar bridge char n-gram smoke result:

```bash
python3 -m bioxrep.baselines.char_ngram_retrieval --input data/bioxrep_clinvar_gene_bridge_100k_allele_heldout.jsonl --track clinvar_gene --query-split test --candidate-split train --candidate-notation gene_symbol --max-queries 200
```

- Held-out ClinVar AlleleID to gene-symbol retrieval: top-1 `0.000`, top-5 `0.000`, MRR `0.00026` over 200 queries.

This task is intentionally hard for surface similarity because ClinVar AlleleIDs are opaque identifiers. It is a good target for structured canonicalization, teacher alignment, or supervised contrastive learning.

Build real ClinVar HGVS variant-notation equivalence classes:

```bash
python3 -m bioxrep.data.fetch_public clinvar_hgvs
python3 -m bioxrep.data.build_clinvar_hgvs_variants --hgvs data/raw/clinvar_hgvs/hgvs4variation.txt.gz --hgnc data/raw/hgnc_complete_set/hgnc_complete_set.txt --output data/bioxrep_clinvar_hgvs_variants_50k.jsonl --max-examples 50000
python3 -m bioxrep.data.prepare_forms --input data/bioxrep_clinvar_hgvs_variants_50k.jsonl --output data/bioxrep_clinvar_hgvs_variants_50k_forms.jsonl
python3 -m bioxrep.data.make_splits --input data/bioxrep_clinvar_hgvs_variants_50k.jsonl --output data/bioxrep_clinvar_hgvs_variants_50k_protein_heldout.jsonl --track clinvar_hgvs --heldout-notation protein_expression
```

Initial ClinVar HGVS artifact sizes:

- `data/bioxrep_clinvar_hgvs_variants_50k.jsonl`: 50,000 variation equivalence classes.
- `data/bioxrep_clinvar_hgvs_variants_50k_forms.jsonl`: 1,457,957 flattened forms.
- `data/bioxrep_clinvar_hgvs_variants_50k_protein_heldout.jsonl`: 1,337,406 train rows and 120,551 held-out protein-expression rows.

Initial ClinVar HGVS retrieval results:

```bash
python3 -m bioxrep.baselines.char_ngram_retrieval --input data/bioxrep_clinvar_hgvs_variants_50k_protein_heldout.jsonl --track clinvar_hgvs --query-split test --candidate-split train --candidate-notation nucleotide_expression --max-queries 200 --max-candidates 50000
python3 -m bioxrep.baselines.canonical_teacher_retrieval --input data/bioxrep_clinvar_hgvs_variants_50k_protein_heldout.jsonl --track clinvar_hgvs --query-split test --candidate-split train --candidate-notation nucleotide_expression --canonical-key variation_id --relevance-key variation_id --max-queries 200
```

- Char n-gram protein-expression to nucleotide-expression retrieval: top-1 `0.000`, top-5 `0.000`, MRR `0.0047` over 200 queries and 50,000 candidates.
- Canonical teacher protein-expression to nucleotide-expression retrieval: top-1 `1.000`, top-5 `1.000`, MRR `1.000` over 200 queries.

This is the first real BioXRep variant-notation benchmark: surface similarity fails to align protein HGVS with nucleotide HGVS, while structured variation IDs provide a perfect teacher signal.

Build ClinVar variant-summary clinical annotation classes:

```bash
python3 -m bioxrep.data.fetch_public clinvar_variant_summary
python3 -m bioxrep.data.build_clinvar_variant_summary --input data/raw/clinvar_variant_summary/variant_summary.txt.gz --output data/bioxrep_clinvar_summary_50k.jsonl --max-examples 50000
python3 -m bioxrep.data.prepare_forms --input data/bioxrep_clinvar_summary_50k.jsonl --output data/bioxrep_clinvar_summary_50k_forms.jsonl
python3 -m bioxrep.data.make_splits --input data/bioxrep_clinvar_summary_50k.jsonl --output data/bioxrep_clinvar_summary_50k_clinsig_heldout.jsonl --track clinvar_summary --heldout-notation clinical_significance
```

Initial ClinVar summary artifact sizes:

- `data/bioxrep_clinvar_summary_50k.jsonl`: 50,000 clinical variant-summary equivalence classes.
- `data/bioxrep_clinvar_summary_50k_forms.jsonl`: 1,080,027 flattened forms.
- `data/bioxrep_clinvar_summary_50k_clinsig_heldout.jsonl`: 1,030,055 train rows and 49,972 held-out clinical-significance rows.

Initial ClinVar summary retrieval results:

```bash
python3 -m bioxrep.baselines.char_ngram_retrieval --input data/bioxrep_clinvar_summary_50k_clinsig_heldout.jsonl --track clinvar_summary --query-split test --candidate-split train --candidate-notation clinvar_name --max-queries 200 --max-candidates 50000
python3 -m bioxrep.baselines.canonical_teacher_retrieval --input data/bioxrep_clinvar_summary_50k_clinsig_heldout.jsonl --track clinvar_summary --query-split test --candidate-split train --candidate-notation clinvar_name --canonical-key variation_id --relevance-key variation_id --max-queries 200
```

- Char n-gram clinical-significance to ClinVar-name retrieval: top-1 `0.000`, top-5 `0.000`, MRR `0.0024` over 200 queries and 50,000 candidates.
- Canonical teacher clinical-significance to ClinVar-name retrieval: top-1 `1.000`, top-5 `1.000`, MRR `1.000` over 200 queries.

This adds a clinical interpretation layer to the notation benchmark: the same ClinVar variation can now connect molecular notation, VCF-style alleles, clinical significance, phenotype text, and review status.

## Train the First Contrastive Student

Build teacher-positive pairs from ClinVar HGVS classes:

```bash
python3 -m bioxrep.data.build_pairs --input data/bioxrep_clinvar_hgvs_variants_50k.jsonl --output data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_20k.jsonl --left-notation protein_expression --right-notation nucleotide_expression --max-pairs 20000
```

Train a lightweight byte/character mean-pooling student:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_20k.jsonl --output-dir outputs/contrastive_student_hgvs_20k --max-pairs 20000 --epochs 5 --batch-size 256 --max-length 160
```

The `KMP_DUPLICATE_LIB_OK=TRUE` prefix is a local macOS workaround for duplicate OpenMP runtime initialization in this Python/PyTorch environment.

Initial student result:

- Valid pairs: 2,000.
- Epoch 1: top-1 `0.016`, top-5 `0.079`.
- Epoch 5: top-1 `0.069`, top-5 `0.246`.

This first student is intentionally small, but it confirms that a learned representation can improve on chance for protein-HGVS to nucleotide-HGVS alignment. The next model should replace mean pooling with a stronger character CNN or Transformer encoder.

Train a stronger character CNN student:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_20k.jsonl --output-dir outputs/contrastive_student_hgvs_cnn_10k --encoder cnn --max-pairs 10000 --epochs 3 --batch-size 128 --max-length 160 --hidden-dim 64 --projection-dim 128
```

Initial CNN student result:

- Valid pairs: 1,000.
- Epoch 1: top-1 `0.065`, top-5 `0.239`.
- Epoch 3: top-1 `0.457`, top-5 `0.903`.

This is the first strong learned BioXRep result: a modest character CNN learns to align protein HGVS with nucleotide HGVS far better than surface retrieval and far better than the mean-pooling student.

Train a fact-aware CNN student with structured variation supervision:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_20k.jsonl --output-dir outputs/contrastive_student_hgvs_cnn_varid_10k --encoder cnn --max-pairs 10000 --epochs 3 --batch-size 128 --max-length 160 --hidden-dim 64 --projection-dim 128 --attribute-fields variation_id --attribute-loss-weight 0.2
```

Updated fact-aware plus attribute-supervised CNN result:

- Valid pairs: 1,000.
- Epoch 1: top-1 `0.262`, top-5 `0.533`, variation-id accuracy `0.050` left / `0.040` right.
- Epoch 2: top-1 `0.618`, top-5 `0.871`, variation-id accuracy `0.261` left / `0.201` right.
- Epoch 3: top-1 `0.803`, top-5 `0.972`, variation-id accuracy `0.493` left / `0.427` right.

This is the current best BioXRep HGVS student result in the repository. Two changes matter: the contrastive loss now treats rows with the same `fact_id` as positives instead of false negatives, and the student also predicts structured `variation_id` labels during training.

Regenerate the ClinVar HGVS artifact with parsed scalar positions:

```bash
python3 -m bioxrep.data.build_clinvar_hgvs_variants --hgvs data/raw/clinvar_hgvs/hgvs4variation.txt.gz --hgnc data/raw/hgnc_complete_set/hgnc_complete_set.txt --output data/bioxrep_clinvar_hgvs_variants_50k_positions.jsonl --max-examples 50000
python3 -m bioxrep.data.build_pairs --input data/bioxrep_clinvar_hgvs_variants_50k_positions.jsonl --output data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_20k_positions.jsonl --left-notation protein_expression --right-notation nucleotide_expression --max-pairs 20000
```

The rebuilt real-data HGVS artifact now stores parsed `protein_position`, `nucleotide_position`, `cdna_position`, and `genomic_position` attributes. On the first 10k training pairs, coverage is complete for `protein_position`, `nucleotide_position`, and `cdna_position`, and `0.997` for `genomic_position`.

Create a held-out real HGVS position split:

```bash
python3 -m bioxrep.data.make_splits --input data/bioxrep_clinvar_hgvs_variants_50k_positions.jsonl --output data/bioxrep_clinvar_hgvs_variants_50k_protein_position_500plus_heldout.jsonl --track clinvar_hgvs --heldout-numeric-range protein_position 500 5000
```

This first real held-out-position split writes `911,040` train rows and `546,917` held-out test rows.

Build fact-disjoint held-out-position HGVS pair files:

```bash
python3 -m bioxrep.data.build_pairs --input data/bioxrep_clinvar_hgvs_variants_50k_protein_position_500plus_heldout.jsonl --output data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_train_10k.jsonl --left-notation protein_expression --right-notation nucleotide_expression --left-split train --right-split train --max-pairs 10000
python3 -m bioxrep.data.build_pairs --input data/bioxrep_clinvar_hgvs_variants_50k_protein_position_500plus_heldout.jsonl --output data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_test_10k.jsonl --left-notation protein_expression --right-notation nucleotide_expression --left-split test --right-split test --max-pairs 10000
```

The resulting held-out-position pair files are fact-disjoint in this larger evaluation slice: `309` train facts, `455` test facts, and `0` overlap.

Train the fact-aware CNN with explicit numeric position features:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_20k_positions.jsonl --output-dir outputs/contrastive_student_hgvs_cnn_varid_pos_explicit_10k --encoder cnn --max-pairs 10000 --epochs 3 --batch-size 128 --max-length 160 --hidden-dim 64 --projection-dim 128 --attribute-fields variation_id --attribute-loss-weight 0.2 --numeric-fields protein_position,nucleotide_position --numeric-feature-mode explicit
```

Explicit numeric position result:

- Valid pairs: 1,000.
- Epoch 3: top-1 `0.884`, top-5 `0.996`, variation-id accuracy `0.472` left / `0.420` right.

Train the fact-aware CNN with sinusoidal numeric position encoding:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_20k_positions.jsonl --output-dir outputs/contrastive_student_hgvs_cnn_varid_pos_sinusoidal_10k --encoder cnn --max-pairs 10000 --epochs 3 --batch-size 128 --max-length 160 --hidden-dim 64 --projection-dim 128 --attribute-fields variation_id --attribute-loss-weight 0.2 --numeric-fields protein_position,nucleotide_position --numeric-feature-mode sinusoidal
```

Sinusoidal numeric position result:

- Valid pairs: 1,000.
- Epoch 1: top-1 `0.991`, top-5 `1.000`, variation-id accuracy `0.866` left / `0.866` right.
- Epoch 3: top-1 `0.996`, top-5 `1.000`, variation-id accuracy `0.953` left / `0.953` right.

This is the current best real-data HGVS result. In this benchmark, parsed position features help substantially, and sinusoidal position encoding is much stronger than feeding normalized scalar positions directly.

Run a held-out-position OOD evaluation with explicit train and test pair files:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_train_10k.jsonl --valid-input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_test_10k.jsonl --output-dir outputs/contrastive_student_hgvs_pos500plus_baseline_10k_test10k --encoder cnn --epochs 3 --batch-size 128 --max-length 160 --hidden-dim 64 --projection-dim 128 --attribute-fields variation_id --attribute-loss-weight 0.2
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_train_10k.jsonl --valid-input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_test_10k.jsonl --output-dir outputs/contrastive_student_hgvs_pos500plus_sinusoidal_10k_test10k --encoder cnn --epochs 3 --batch-size 128 --max-length 160 --hidden-dim 64 --projection-dim 128 --attribute-fields variation_id --attribute-loss-weight 0.2 --numeric-fields protein_position,nucleotide_position --numeric-feature-mode sinusoidal
```

Held-out-position OOD comparison:

- No numeric position features, epoch 3: top-1 `0.3582`, top-5 `0.4787`.
- Sinusoidal numeric position features, epoch 3: top-1 `0.9999`, top-5 `0.9999`.

This OOD slice is stricter than the earlier in-distribution run because train and test facts are disjoint under a protein-position holdout, and it now covers `455` held-out test facts instead of the earlier tiny pilot slice. Validation variation-ID accuracy is not reported here because the held-out test variation IDs are unseen in the train vocabulary.

Follow-up ablations on the same 455-fact held-out-position split:

- `protein_position + cdna_position` with sinusoidal encoding: epoch 3 top-1 `1.0000`, top-5 `1.0000`.
- `protein_position + nucleotide_position` with sinusoidal encoding plus numeric consistency loss (`--numeric-loss-weight 0.1`): epoch 3 top-1 `0.9999`, top-5 `0.9999`.
- Digit-masked text with sinusoidal numeric features (`--text-transform mask_digits`): epoch 3 top-1 `0.9999`, top-5 `1.0000`.
- Numeric-only ablation with sinusoidal numeric features (`--text-weight 0.0`): epoch 3 top-1 `0.9999`, top-5 `1.0000`.

These ablations change the interpretation of the near-perfect OOD result. The gain is not primarily coming from raw digit strings in the text encoder, because masking all digits barely moves retrieval. It is also not sensitive to whether the coding coordinate is supplied as `nucleotide_position` or `cdna_position` in this dataset. The dominant signal is the structured numeric-position pathway itself.

Build hard HGVS retrieval candidate sets where every decoy shares the same parsed position fields as the query fact:

```bash
python3 -m bioxrep.data.build_clinvar_hgvs_variants --hgvs data/raw/clinvar_hgvs/hgvs4variation.txt.gz --hgnc data/raw/hgnc_complete_set/hgnc_complete_set.txt --output data/bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl --max-examples 50000
python3 -m bioxrep.data.build_hard_retrieval_benchmark --input data/bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl --output data/bioxrep_clinvar_hgvs_hard_protein_position_min10_1k.jsonl --query-notation protein_expression --candidate-notation nucleotide_expression --confound-fields protein_position --min-decoys 10 --max-queries 1000 --max-candidates 20 --seed 13
python3 -m bioxrep.data.build_hard_retrieval_benchmark --input data/bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl --output data/bioxrep_clinvar_hgvs_hard_protein_cdna_position_min10_1k.jsonl --query-notation protein_expression --candidate-notation nucleotide_expression --confound-fields protein_position,cdna_position --min-decoys 10 --max-queries 1000 --max-candidates 20 --seed 13
```

Run the lexical hard-set baseline:

```bash
python3 -m bioxrep.baselines.char_ngram_hard_retrieval --input data/bioxrep_clinvar_hgvs_hard_protein_position_min10_1k.jsonl --output outputs/char_ngram_hard_protein_position_min10_1k.json
python3 -m bioxrep.baselines.char_ngram_hard_retrieval --input data/bioxrep_clinvar_hgvs_hard_protein_cdna_position_min10_1k.jsonl --output outputs/char_ngram_hard_protein_cdna_position_min10_1k.json
```

Hard-set char n-gram results:

- Same `protein_position` decoys: top-1 `0.160`, top-5 `0.373`, MRR `0.293` over 1,000 queries with 11-20 candidates/query.
- Same `protein_position + cdna_position` decoys: top-1 `0.184`, top-5 `0.504`, MRR `0.346` over 1,000 queries with 11-20 candidates/query.

These hard candidate sets are the next evaluation target for learned BioXRep models because position fields are no longer sufficient to identify the correct variant.

Evaluate trained students on the hard candidate sets:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.eval.hard_student_retrieval --checkpoint outputs/contrastive_student_hgvs_pos500plus_baseline_10k_test10k/char_cnn_student.pt --input data/bioxrep_clinvar_hgvs_hard_protein_cdna_position_min10_1k.jsonl --output outputs/hard_eval_baseline_textonly_protein_cdna_min10_1k.json
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.eval.hard_student_retrieval --checkpoint outputs/contrastive_student_hgvs_pos500plus_sinusoidal_10k_test10k/char_cnn_student.pt --input data/bioxrep_clinvar_hgvs_hard_protein_cdna_position_min10_1k.jsonl --output outputs/hard_eval_sinusoidal_protein_cdna_min10_1k.json
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.eval.hard_student_retrieval --checkpoint outputs/contrastive_student_hgvs_pos500plus_sinusoidal_numeric_only_10k_test10k/char_cnn_student.pt --input data/bioxrep_clinvar_hgvs_hard_protein_cdna_position_min10_1k.jsonl --output outputs/hard_eval_numeric_only_protein_cdna_min10_1k.json
```

Hard-set learned-model results:

| Candidate set | Model | Top-1 | Top-5 | MRR |
| --- | --- | ---: | ---: | ---: |
| Same `protein_position` | text-only CNN | `0.143` | `0.459` | `0.302` |
| Same `protein_position` | sinusoidal text+numeric CNN | `0.378` | `0.880` | `0.572` |
| Same `protein_position` | numeric-only sinusoidal CNN | `0.342` | `0.873` | `0.542` |
| Same `protein_position + cdna_position` | text-only CNN | `0.166` | `0.612` | `0.365` |
| Same `protein_position + cdna_position` | sinusoidal text+numeric CNN | `0.149` | `0.446` | `0.308` |
| Same `protein_position + cdna_position` | numeric-only sinusoidal CNN | `0.126` | `0.392` | `0.279` |
| Same `protein_position + cdna_position` | masked-digit text+numeric CNN | `0.139` | `0.495` | `0.310` |

Interpretation: matching only `protein_position` is not enough because nucleotide or cDNA coordinates still provide a strong shortcut. Matching both `protein_position` and `cdna_position` removes most of that advantage; the numeric-only model falls below the text-only CNN, making this the stronger candidate set for paper-grade evaluation.

Train with hard-set validation reported at every epoch:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_train_10k.jsonl --valid-input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_test_10k.jsonl --hard-valid-input data/bioxrep_clinvar_hgvs_hard_protein_cdna_position_min10_1k.jsonl --output-dir outputs/contrastive_student_hgvs_hardvalid_textonly_10k --encoder cnn --epochs 3 --batch-size 128 --max-length 160 --hidden-dim 64 --projection-dim 128 --attribute-fields variation_id --attribute-loss-weight 0.2
```

The trainer logs `hard_valid_top1`, `hard_valid_top5`, `hard_valid_mean_reciprocal_rank`, and candidate-set statistics in each epoch record.

First strict hard-validation training matrix, using same-`protein_position + cdna_position` candidate pools:

| Training run | Old pair top-1 | Hard top-1 | Hard top-5 | Hard MRR |
| --- | ---: | ---: | ---: | ---: |
| text-only CNN | `0.4190` | `0.232` | `0.661` | `0.421` |
| sinusoidal text+numeric CNN, `protein_position + cdna_position` | `0.9999` | `0.123` | `0.447` | `0.289` |
| numeric-only sinusoidal CNN, `protein_position + cdna_position` | `0.9999` | `0.079` | `0.348` | `0.235` |
| masked-digit text+numeric CNN, `protein_position + cdna_position` | `0.9999` | `0.108` | `0.497` | `0.294` |

This is the first clean evidence that the old pair validation can be solved by structured numeric shortcuts while the strict candidate-pool task cannot. Current best strict hard-set model is the text-only CNN.

Train a leakage-safe equivalence-class-aware text-only model on the same train facts:

```bash
python3 -m bioxrep.data.filter_equivalence_classes --input data/bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl --reference data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_train_10k.jsonl --output data/bioxrep_clinvar_hgvs_variants_numeric_500plus_train_classes.jsonl
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --class-input data/bioxrep_clinvar_hgvs_variants_numeric_500plus_train_classes.jsonl --class-notations protein_expression,nucleotide_expression,protein_change,nucleotide_change --forms-per-class 4 --valid-input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_test_10k.jsonl --hard-valid-input data/bioxrep_clinvar_hgvs_hard_protein_cdna_position_min10_1k.jsonl --output-dir outputs/contrastive_student_hgvs_classaware_textonly_342facts --encoder cnn --epochs 5 --batch-size 64 --max-length 160 --hidden-dim 64 --projection-dim 128 --attribute-fields variation_id --attribute-loss-weight 0.2
```

Class-aware text-only result:

- Train classes: `342`.
- Epoch 5 old pair validation: top-1 `0.4202`, top-5 `0.5768`.
- Epoch 5 strict hard validation: top-1 `0.130`, top-5 `0.542`, MRR `0.324`.

This simple class-aware objective improves neither hard top-1 nor MRR over the pair-trained text-only CNN, though hard top-5 remains competitive. The next modeling step should use hard negatives directly rather than relying only on same-class positives.

Train directly on leakage-safe hard-negative pools from train facts only:

```bash
python3 -m bioxrep.data.build_hard_retrieval_benchmark --input data/bioxrep_clinvar_hgvs_variants_numeric_500plus_train_classes.jsonl --output data/bioxrep_clinvar_hgvs_hard_train_protein_cdna_position_min1.jsonl --query-notation protein_expression --candidate-notation nucleotide_expression --confound-fields protein_position,cdna_position --min-decoys 1 --max-candidates 20 --seed 17
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --hard-train-input data/bioxrep_clinvar_hgvs_hard_train_protein_cdna_position_min1.jsonl --valid-input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_test_10k.jsonl --max-valid-pairs 1000 --hard-valid-input data/bioxrep_clinvar_hgvs_hard_protein_cdna_position_min10_1k.jsonl --max-hard-valid-queries 250 --output-dir outputs/contrastive_student_hgvs_hardnegative_train342_min1_textonly_3ep --encoder cnn --epochs 3 --batch-size 16 --max-length 160 --hidden-dim 64 --projection-dim 128
```

Hard-negative pilot result:

- Leakage-safe train hard queries: `79`, average candidates/query `2.15`.
- Epoch 3 train hard top-1 `0.7475`, top-5 `1.000`, showing rapid fit to the small train pool.
- Epoch 3 capped strict hard validation: top-1 `0.128`, top-5 `0.404`, MRR `0.285`.

This confirms that hard-negative training is technically wired, but the current leakage-safe train pool is too small and too easy. The next data step is to construct larger train-only hard-negative pools, either by expanding the train fact set or by using less restrictive confounders during training while keeping strict `protein_position + cdna_position` pools for evaluation.

Build a denser train-only hard-negative pool by keeping same-`protein_position` matched decoys and filling remaining candidate slots with random train-only decoys:

```bash
python3 -m bioxrep.data.build_hard_retrieval_benchmark --input data/bioxrep_clinvar_hgvs_variants_numeric_500plus_train_classes.jsonl --output data/bioxrep_clinvar_hgvs_hard_train_protein_position_filled20.jsonl --query-notation protein_expression --candidate-notation nucleotide_expression --confound-fields protein_position --min-decoys 1 --max-candidates 20 --fill-random-decoys --seed 23
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --hard-train-input data/bioxrep_clinvar_hgvs_hard_train_protein_position_filled20.jsonl --valid-input data/bioxrep_clinvar_hgvs_protein_to_nucleotide_pairs_500plus_test_10k.jsonl --max-valid-pairs 1000 --hard-valid-input data/bioxrep_clinvar_hgvs_hard_protein_cdna_position_min10_1k.jsonl --max-hard-valid-queries 250 --output-dir outputs/contrastive_student_hgvs_hardnegative_train342_proteinpos_filled20_textonly_5ep --encoder cnn --epochs 5 --batch-size 8 --max-length 160 --hidden-dim 64 --projection-dim 128
```

Dense hard-negative result:

- Train hard queries: `237`, exactly 20 candidates/query.
- Train-pool char n-gram baseline: top-1 `0.110`, top-5 `0.409`, MRR `0.277`.
- Epoch 5 capped strict hard validation: top-1 `0.168`, top-5 `0.532`, MRR `0.339`.
- Full 1k strict hard evaluation: top-1 `0.134`, top-5 `0.505`, MRR `0.316`.

Dense hard-negative training improves over sparse hard-negative pilots, but still does not beat the pair-trained text-only CNN on full strict hard evaluation. This suggests the next gain is likely from broader train facts and richer hard negatives, not from the current 342-fact train pool alone.

Scale the HGVS fact split and hard benchmark:

```bash
python3 -m bioxrep.data.split_equivalence_classes --input data/bioxrep_clinvar_hgvs_variants_numeric_50k.jsonl --train-output data/bioxrep_clinvar_hgvs_variants_numeric_scaled_train_40k.jsonl --test-output data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl --required-notations protein_expression,nucleotide_expression --train-fraction 0.8 --max-examples 50000 --seed 31
python3 -m bioxrep.data.build_pairs --input data/bioxrep_clinvar_hgvs_variants_numeric_scaled_train_40k.jsonl --output data/bioxrep_clinvar_hgvs_scaled_train_pairs_50k.jsonl --left-notation protein_expression --right-notation nucleotide_expression --max-pairs 50000
python3 -m bioxrep.data.build_pairs --input data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl --output data/bioxrep_clinvar_hgvs_scaled_test_pairs_20k.jsonl --left-notation protein_expression --right-notation nucleotide_expression --max-pairs 20000
python3 -m bioxrep.data.build_hard_retrieval_benchmark --input data/bioxrep_clinvar_hgvs_variants_numeric_scaled_test_10k.jsonl --output data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl --query-notation protein_expression --candidate-notation nucleotide_expression --confound-fields protein_position,cdna_position --min-decoys 1 --max-queries 2000 --max-candidates 20 --fill-random-decoys --seed 37
```

Scaled data artifacts:

- `29,076` train equivalence classes and `7,270` test equivalence classes after requiring protein and nucleotide expressions.
- `50,000` train pairs and `20,000` test pairs.
- `2,000` strict-filled hard test queries with exactly `20` candidates/query.
- Scaled strict-filled hard test char n-gram baseline: top-1 `0.154`, top-5 `0.3995`, MRR `0.296`.

Train the scaled text-only CNN:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.train.train_contrastive_student --input data/bioxrep_clinvar_hgvs_scaled_train_pairs_50k.jsonl --valid-input data/bioxrep_clinvar_hgvs_scaled_test_pairs_20k.jsonl --max-valid-pairs 5000 --hard-valid-input data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl --max-hard-valid-queries 500 --output-dir outputs/contrastive_student_hgvs_scaled_pair_textonly_50k --encoder cnn --epochs 3 --batch-size 128 --max-length 160 --hidden-dim 64 --projection-dim 128 --attribute-fields variation_id --attribute-loss-weight 0.2
KMP_DUPLICATE_LIB_OK=TRUE python3 -m bioxrep.eval.hard_student_retrieval --checkpoint outputs/contrastive_student_hgvs_scaled_pair_textonly_50k/char_cnn_student.pt --input data/bioxrep_clinvar_hgvs_scaled_test_hard_protein_cdna_filled20_2k.jsonl --output outputs/hard_eval_scaled_pair_textonly_50k_protein_cdna_filled20_2k.json
```

Scaled text-only CNN result:

- Epoch 3 capped hard validation: top-1 `0.672`, top-5 `0.950`, MRR `0.799`.
- Full 2k strict-filled hard evaluation: top-1 `0.6995`, top-5 `0.9465`, MRR `0.812`.

This is the strongest BioXRep result so far. Scaling fact coverage, not adding numeric shortcuts, gives a large gain on the strict hard retrieval task.

Scaled ablation table on the same full 2k strict-filled hard benchmark:

| Model | Old pair top-1 | Hard top-1 | Hard top-5 | Hard MRR |
| --- | ---: | ---: | ---: | ---: |
| text-only CNN | `0.9574` | `0.6995` | `0.9465` | `0.812` |
| sinusoidal text+numeric CNN, `protein_position + cdna_position` | `1.0000` | `0.4875` | `0.9985` | `0.701` |
| numeric-only sinusoidal CNN, `protein_position + cdna_position` | `1.0000` | `0.3480` | `0.9840` | `0.610` |
| masked-digit text+numeric CNN, `protein_position + cdna_position` | `1.0000` | `0.5385` | `0.9995` | `0.731` |

Interpretation: the scaled strict benchmark is not solved by numeric features alone. Numeric features push top-5 very high, but they reduce top-1 compared with text-only training. Masked-digit text still improves over numeric-only, showing that non-digit sequence context carries useful notation-invariant signal. The main paper table should report text-only and numeric-feature models separately.

## Fetch MIMIC-IV Lab Tables

MIMIC-IV requires credentialed PhysioNet access. Before fetching, complete the PhysioNet credentialing flow for MIMIC-IV v3.1, including required training and the data use agreement.

BioXRep expects credentials in environment variables and does not store them:

```bash
export PHYSIONET_USERNAME="your_physionet_username"
export PHYSIONET_PASSWORD="your_physionet_password"
```

Alternatively, create a local `.env` file, which is ignored by git:

```bash
PHYSIONET_USERNAME=your_physionet_username
PHYSIONET_PASSWORD=your_physionet_password
```

Preview the MIMIC-IV lab downloads:

```bash
python3 -m bioxrep.data.fetch_public mimiciv_hosp_d_labitems mimiciv_hosp_labevents --dry-run
```

Fetch the minimum lab-value tables:

```bash
python3 -m bioxrep.data.fetch_public mimiciv_hosp_d_labitems mimiciv_hosp_labevents
```

Or with a local env file:

```bash
python3 -m bioxrep.data.fetch_public mimiciv_hosp_d_labitems mimiciv_hosp_labevents --env-file .env
```

Optional patient/admission context:

```bash
python3 -m bioxrep.data.fetch_public mimiciv_hosp_patients mimiciv_hosp_admissions
```

The most important BioXRep tables are `d_labitems.csv.gz`, which defines lab concepts and labels, and `labevents.csv.gz`, which contains measured values, units, flags, and timestamps. MIMIC-IV v3.1 is credentialed access on PhysioNet and is also available through BigQuery after PhysioNet access approval.

If fetching returns `HTTP 401` or `HTTP 403`, confirm that:

- the `.env` file contains the same username/password used to log in to PhysioNet,
- the account has approved MIMIC-IV access, not only a PhysioNet account,
- the required training and MIMIC-IV data use agreement are complete,
- the account can manually download `d_labitems.csv.gz` from the MIMIC-IV v3.1 PhysioNet page.

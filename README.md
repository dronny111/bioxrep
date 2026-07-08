# BioXRep

Cross-notation biomedical entity normalization benchmarks and baselines.

BioXRep asks whether a model can recover the underlying biological **fact** when
the same object appears as different surface notations: gene symbols, aliases,
former names, HGVS strings, opaque IDs, and free-text descriptions.

This repository is currently a **benchmark-and-baselines** contribution, not a
positive method result. The main finding is negative: lightweight surface-form
students fit in-distribution pairs but do not generalize to a fact- and
notation-disjoint HGNC alias task.

## Highlights

- Fact- and notation-disjoint HGNC gene-alias retrieval benchmark.
- Parallel ClinVar HGVS protein-to-nucleotide numeric pilot.
- Baselines: character n-gram, SapBERT dense, BioSyn-style hybrid, canonical
  teacher oracle, char-CNN student, and byte-Transformer student.
- Leakage checks via [`scripts/verify_no_leakage.py`](scripts/verify_no_leakage.py).
- Standalone visual explainer:
  [`docs/bioxrep_visualization_demo.html`](docs/bioxrep_visualization_demo.html).

## Key Results

### HGNC `alias_symbol` Held-Out Retrieval

Queries are held-out `alias_symbol` forms; candidates are the full
`44,997` approved-symbol inventory.

| Method | top-1 | top-5 | MRR |
| --- | ---: | ---: | ---: |
| Canonical teacher | 1.000 | 1.000 | 1.000 |
| SapBERT dense | 0.080 | 0.190 | 0.134 |
| BioSyn hybrid | 0.077 | 0.182 | 0.129 |
| Character n-gram | 0.047 | 0.108 | 0.077 |
| Char-CNN student | 0.041 ± 0.002 | 0.075 ± 0.001 | 0.059 ± 0.001 |
| Byte-Transformer student | 0.024 ± 0.002 | 0.055 ± 0.003 | 0.040 ± 0.001 |

The char-CNN reaches `0.985` in-domain validation top-1, but only
`0.059 ± 0.001` MRR on the held-out notation. Attention distillation changes MRR
by about `0.001`, within seed variance.

### HGVS Numeric Pilot

On matched-only position-confounded hard retrieval:

| Numeric input mode | top-1 | MRR |
| --- | ---: | ---: |
| Text-only / digit-token | 0.9195 | 0.957 |
| xVal-style continuous tokenization | 0.9060 | 0.951 |
| Explicit normalized scalar | 0.9000 | 0.947 |
| Sinusoidal Fourier features | 0.4980 | 0.708 |

Numeric features look helpful on easy random-fill pools, but not when candidates
share the same parsed protein and cDNA positions.

## Install

```bash
conda create -n bioxrep-sapbert python=3.11 -y
conda activate bioxrep-sapbert
pip install torch numpy scikit-learn transformers pytest
```

SapBERT/BioSyn runs download `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` on
first use. Set `HF_HOME=.hf_cache` to cache locally.

## Quickstart

Run the deterministic synthetic benchmark and first lexical baseline:

```bash
python3 -m bioxrep.data.generate_synthetic --output data/bioxrep_synth.jsonl --variant-count 40 --seed 13
python3 -m bioxrep.data.prepare_forms --input data/bioxrep_synth.jsonl --output data/bioxrep_synth_forms.jsonl
python3 -m bioxrep.baselines.char_ngram_retrieval --input data/bioxrep_synth.jsonl
```

Run the HGNC comparison table:

```bash
python3 -m bioxrep.data.fetch_public hgnc_complete_set
python3 -m bioxrep.data.build_hgnc_aliases \
  --input data/raw/hgnc_complete_set/hgnc_complete_set.txt \
  --output data/bioxrep_hgnc_aliases.jsonl
python3 -m bioxrep.data.make_splits \
  --input data/bioxrep_hgnc_aliases.jsonl \
  --output data/bioxrep_hgnc_alias_symbol_heldout.jsonl \
  --track gene_alias --heldout-notation alias_symbol
HF_HOME=.hf_cache OMP_NUM_THREADS=8 PYTHONPATH=. python3 scripts/run_hgnc_alias_comparison.py
```

## Repository Map

```text
bioxrep/data/       public-source fetchers, form builders, split builders
bioxrep/baselines/  lexical, SapBERT/BioSyn, canonical teacher, student retrieval
bioxrep/models/     byte-level char-CNN and Transformer encoders
bioxrep/train/      supervised-contrastive student trainer
bioxrep/eval/       hard-set retrieval and invariance metrics
scripts/            reproducible experiment drivers and leakage checks
docs/               frozen results, review notes, roadmap, visual demo
tests/              unit tests for metrics, encoders, and attention distillation
```
<!-- 
## More Detail

- HGNC baseline writeup: [`docs/bioxrep_sapbert_baseline.md`](docs/bioxrep_sapbert_baseline.md)
- HGVS numeric pilot: [`docs/bioxrep_hgvs_results.md`](docs/bioxrep_hgvs_results.md)
- Work-so-far review: [`docs/bioxrep_review_and_gaps.md`](docs/bioxrep_review_and_gaps.md)
- Research roadmap: [`docs/bioxrep_experiment_roadmap.md`](docs/bioxrep_experiment_roadmap.md)
- Visual demo: [`docs/bioxrep_visualization_demo.html`](docs/bioxrep_visualization_demo.html) -->

## Reproducibility Notes

- `data/`, `outputs/`, `.hf_cache/`, and `paper/` are git-ignored.
- Use `scripts/verify_no_leakage.py` after rebuilding any split.
- Retrieval commands support `--bootstrap` for 95% bootstrap CIs.
- Learned student runs support `--seed` for multi-seed mean ± std reporting.

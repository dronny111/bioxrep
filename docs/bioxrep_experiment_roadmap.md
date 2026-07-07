# BioXRep Experiment Roadmap

## Guiding principle

Start with controlled data where the invariances are known, then move to real biomedical data after the method and metrics are stable.

The first goal is not to beat a large biological foundation model. The first goal is to show that cross-notation distillation produces representations that are measurably more invariant, more numerically faithful, and more robust under notation shift than standard token-based models.

## Phase 0: BioXRep foundation

Deliverables:

- Keep the active repository focused on the BioXRep framing and implementation.
- Maintain project history in Git rather than carrying historical prototype code in the working tree.
- Add a clean `bioxrep/` package for experiments.
- Add `docs/` research artifacts for the paper framing and roadmap.
- Define a minimal configuration format.
- Add deterministic data generation and evaluation scripts.

Success criterion:

- A fresh user can run a small synthetic benchmark end to end.

## Phase 1: BioXRep-Synth benchmark

This phase creates synthetic-but-biologically-grounded equivalence classes.

### Track A: protein and variant notation

Generate examples with fields:

- `gene`
- `transcript`
- `protein_position`
- `reference_aa_1`
- `reference_aa_3`
- `alternate_aa_1`
- `alternate_aa_3`
- `cdna_position`
- `reference_base`
- `alternate_base`
- `mutation_type`

Example forms:

```text
BRAF V600E
BRAF p.Val600Glu
BRAF Val600Glu
NM_004333.6:c.1799T>A
missense variant in BRAF changing valine to glutamic acid at protein position 600
```

Initial controlled splits:

- Held-out genes.
- Held-out amino-acid substitutions.
- Held-out notation families.
- Held-out numeric positions.
- Compositional split: seen gene and seen substitution, unseen pairing.

Implementation status:

- `bioxrep.data.make_splits` now supports `--heldout-numeric-range NAME MIN MAX`, which can be used to create held-out protein-position and other numeric-range evaluation sets on both synthetic and real-data artifacts.

### Track B: clinical lab values

Generate examples with fields:

- `lab_name`
- `canonical_value`
- `canonical_unit`
- `surface_value`
- `surface_unit`
- `reference_low`
- `reference_high`
- `status`
- `interpretation_text`

Example forms:

```text
glucose 126 mg/dL
glucose 7.0 mmol/L
fasting glucose is elevated
glucose above diabetes diagnostic threshold
```

Initial controlled splits:

- Held-out units.
- Held-out lab names.
- Held-out value ranges.
- Held-out wording templates.
- Hospital-style reference range shift.

## Phase 2: baseline models

Implement small, cheap baselines first:

1. Character CNN or character Transformer encoder.
2. Byte/subword-style encoder.
3. Digit-token numeric encoder.
4. Scientific-notation normalized encoder.
5. Continuous number placeholder encoder inspired by xVal.
6. Contrastive encoder without distillation.
7. Attribute prediction encoder without contrastive equivalence.

Keep all baselines small enough to run locally for early iteration.

## Phase 3: BioXRep objective

Start with a canonical-field teacher rather than a neural teacher.

Losses:

- `L_contrastive`: equivalent forms close, non-equivalent forms separated.
- `L_attr`: reconstruct canonical fields.
- `L_numeric`: preserve numeric magnitude after unit or notation conversion.
- `L_align`: align surface spans to canonical fields when deterministic alignments are available.

Then add neural teacher distillation:

- Train a seq2seq notation translator.
- Extract cross-attention or token alignment distributions.
- Distill those distributions into the student.
- Compare deterministic alignment, neural attention, and hybrid teachers.

## Phase 4: evaluation

Primary metrics:

- Equivalent-form retrieval top-1/top-5 accuracy.
- Canonical fact identification accuracy.
- Attribute exact match and macro-F1.
- Numeric MAE/RMSE after unit conversion.
- OOD generalization gap.
- Invariance ratio: mean between-class distance divided by mean within-class distance.

Stress tests:

- Notation held out during training.
- Unit held out during training.
- New value ranges.
- New genes.
- New amino-acid substitutions.
- Mixed text plus symbolic notation.
- Ambiguous or alias-heavy forms.

## Phase 5: real-data bridge

Only after synthetic experiments are working, add real data.

Candidate sources:

- ClinVar or related public variant resources for variant notations and gene/variant metadata.
- HGNC for gene symbol and alias normalization.
- UniProt for protein names, accessions, and sequence/protein metadata.
- MIMIC-IV for clinical labs, if access is available and the research question requires real EHR validation.

Real-data goals:

- Confirm that synthetic findings survive realistic noise.
- Evaluate alias ambiguity and incomplete notation.
- Test clinical unit variation and lab reference range variation.

Current real-data bridge status:

- HGNC complete gene set has been converted into gene-alias equivalence classes.
- ClinVar allele-to-gene mappings have been joined with HGNC to create allele-to-gene grounding classes.
- ClinVar HGVS expressions have been converted into variant-notation equivalence classes linking variation IDs, allele IDs, gene identity, nucleotide HGVS, protein HGVS, and HGNC metadata.
- ClinVar variant summary has been converted into clinical annotation equivalence classes linking variation IDs, variant names, VCF-style alleles, clinical significance, phenotype text, review status, and gene metadata.
- Initial surface retrieval baselines fail sharply on held-out alias, opaque allele ID, and protein-to-nucleotide HGVS retrieval.
- Canonical teacher retrieval reaches the expected upper bound when ranking by structured biomedical IDs such as HGNC ID, GeneID, and ClinVar VariationID.
- A first lightweight byte/character contrastive student has been trained on protein-HGVS to nucleotide-HGVS pairs. It improves from top-5 `0.079` at epoch 1 to top-5 `0.246` at epoch 5 on a 2,000-pair validation split, showing that the learned-student loop has signal even with a deliberately small encoder.
- A stronger character CNN student reaches top-1 `0.457` and top-5 `0.903` after 3 epochs on a 10k-pair protein-HGVS to nucleotide-HGVS run, giving the first strong learned BioXRep result.
- Updating the objective to use fact-aware multi-positive contrastive training plus `variation_id` attribute supervision raises the 10k-pair CNN result to top-1 `0.803` and top-5 `0.972` after 3 epochs, with validation variation-ID accuracy reaching `0.493` on left forms and `0.427` on right forms.
- Parsing scalar HGVS positions into the real ClinVar artifact and feeding `protein_position` plus `nucleotide_position` as numeric features improves the same 10k-pair CNN further: explicit normalized numeric features reach top-1 `0.884` and top-5 `0.996`, while sinusoidal numeric position encoding reaches top-1 `0.996` and top-5 `1.000` with validation variation-ID accuracy `0.953` on both left and right forms.
- On a larger fact-disjoint held-out protein-position split with `309` train facts and `455` test facts, the same 10k-train setup reaches top-1 `0.3582` and top-5 `0.4787` without numeric position features, versus top-1 `0.9999` and top-5 `0.9999` with sinusoidal position encoding.
- Follow-up ablations on that same OOD split show that the effect is robust to digit masking and even to zeroing out the text encoder: `protein_position + cdna_position` also reaches top-1 `1.0000`, adding a numeric consistency loss leaves retrieval essentially unchanged at top-1 `0.9999`, digit-masked text remains at top-1 `0.9999`, and a numeric-only ablation also reaches top-1 `0.9999`.
- Caveat: the near-perfect numeric-feature results should currently be treated as diagnostic or teacher-feature upper bounds, not as final evidence that the text encoder has learned notation-invariant biological representations. Because numeric-only features can solve the current OOD retrieval setup, the next evaluation must include candidate sets where position fields are not uniquely identifying, or report numeric-feature models separately from representation-learning models.
- A first hard-candidate HGVS benchmark now constructs per-query candidate pools where all decoys share the query's parsed position fields. Same-`protein_position` pools give the char n-gram baseline top-1 `0.160`, top-5 `0.373`, MRR `0.293`; same-`protein_position + cdna_position` pools give top-1 `0.184`, top-5 `0.504`, MRR `0.346`. These are the first non-shortcut candidate sets for rerunning learned BioXRep models.
- Hard-set learned-model evaluation confirms the shortcut diagnosis. On same-`protein_position` pools, the sinusoidal text+numeric CNN reaches top-1 `0.378` and numeric-only reaches top-1 `0.342`, so a coordinate shortcut remains. On same-`protein_position + cdna_position` pools, text-only reaches top-1 `0.166`, text+numeric reaches top-1 `0.149`, and numeric-only drops to top-1 `0.126`, making this the preferred current evaluation for representation learning.
- A first training-time hard-validation matrix strengthens the conclusion. On same-`protein_position + cdna_position` pools, the text-only CNN reaches hard top-1 `0.232`, top-5 `0.661`, MRR `0.421`. The sinusoidal text+numeric model reaches old pair top-1 `0.9999` but hard top-1 only `0.123`; numeric-only reaches old pair top-1 `0.9999` but hard top-1 `0.079`; masked-digit text+numeric reaches old pair top-1 `0.9999` but hard top-1 `0.108`. This makes strict hard-set retrieval the main metric for future HGVS representation learning.
- A first leakage-safe class-aware text-only run trained on `342` train equivalence classes with four sampled forms per class. It reaches old pair top-1 `0.4202`, top-5 `0.5768`, but strict hard top-1 only `0.130`, top-5 `0.542`, MRR `0.324`. This suggests that simply adding more same-fact positives is not enough; the next objective needs explicit hard-negative pressure from same-position and same-cDNA decoys.
- Hard-negative training is now wired and was tested on a leakage-safe train-only strict pool. The train pool contains only `79` queries with average `2.15` candidates/query. A 3-epoch pilot fits the small train hard pool quickly, reaching train hard top-1 `0.7475`, but capped strict hard validation remains low at top-1 `0.128`, top-5 `0.404`, MRR `0.285`. The objective is useful, but the current train-only hard pool is too sparse for final conclusions.
- A denser train-only hard-negative pool keeps same-`protein_position` matched decoys and fills remaining slots with random train-only decoys, producing `237` queries with exactly `20` candidates/query. This pool is harder for lexical retrieval than the sparse pool: char n-gram top-1 `0.110`, top-5 `0.409`, MRR `0.277`. A 5-epoch text-only hard-negative model reaches capped strict hard top-1 `0.168`, top-5 `0.532`, MRR `0.339`, but full 1k strict evaluation is top-1 `0.134`, top-5 `0.505`, MRR `0.316`. This improves over sparse hard-negative pilots but remains below the pair-trained text-only CNN on full strict evaluation.
- Scaling the fact-disjoint HGVS split changes the result substantially. A new split requiring protein and nucleotide expressions yields about `29k` train classes and `7k` test classes, with `50,000` train pairs and `20,000` test pairs. On the full 20-candidate strict-filled hard pool, numeric features can look strong because most decoys are random fill; on the stricter matched-only projection, the text-only byte encoder is strongest (`0.9195` hard top-1), ahead of xVal-style continuous tokenization (`0.9060`), explicit scalar features (`0.9000`), and sinusoidal Fourier features (`0.4980`).
- Scaled ablations clarify the role of numeric features. Position features are useful for rejecting easy random-fill decoys but do not help discriminate among candidates that share the same parsed `protein_position` and `cdna_position`; in that matched-only setting the numeric component is a confound rather than a discriminator. The main table should therefore separate full-pool convenience metrics from matched-only representation-learning metrics.

## Phase 6: paper experiments

Minimum viable paper table:

| Experiment | Track | Claim |
| --- | --- | --- |
| Equivalent-form retrieval | Variant + lab | BioXRep learns notation-invariant representations |
| Held-out notation | Variant | BioXRep generalizes to unseen symbolic forms |
| Held-out unit | Lab | BioXRep preserves quantity across unit shift |
| Attribute probing | Both | Representations retain useful biological attributes |
| Ablation | Both | Alignment/distillation losses matter |
| Real-data validation | At least one track | The method survives realistic biomedical noise |

## Phase 7: target venues

Near-term workshops:

- NeurIPS AI4Science.
- ML4H.
- MLCB.
- BioNLP workshop at ACL.

Full-paper targets:

- ACL, EMNLP, NAACL for NLP framing.
- ICLR or NeurIPS if the method and benchmark become broadly compelling.
- ISMB or RECOMB if biological validation becomes the main contribution.

## Immediate next steps

Completed credibility fixes:

- Split-aware pair generation now fails fast when split filters are requested on unsplit equivalence-class JSONL, preventing accidental held-out-form leakage.
- Numeric-field training now fails fast when requested numeric fields have no numeric values in the training data, and uses only validated numeric fields for tensors, heads, and feature encoders.
- Hard HGVS retrieval candidate-set construction now supports same-attribute decoys and filters query forms against parseable confound values, preventing isoform-level position mismatches from entering the benchmark.
- Trained BioXRep students can now be evaluated directly on hard candidate sets from saved checkpoints, including older checkpoints without auxiliary numeric-head weights.
- The contrastive student trainer now accepts `--hard-valid-input` and reports `hard_valid_top1`, `hard_valid_top5`, and `hard_valid_mean_reciprocal_rank` during training, so new runs can optimize against the strict candidate-pool evaluation instead of relying only on pair retrieval.
- The trainer now supports leakage-safe equivalence-class-aware training through `--class-input`, `--class-notations`, and `--forms-per-class`, with supervised contrastive positives over all same-fact forms in the batch.
- The trainer now supports `--hard-train-input` for direct hard-negative training on per-query candidate pools.
- Fact-level equivalence-class splitting now supports scaled train/test creation through `bioxrep.data.split_equivalence_classes`.

Current priorities:

1. Use `docs/bioxrep_hgvs_results.md` as the frozen first paper-grade HGVS table.
2. Optimize hard-negative training batching so 5k-20k hard query pools can train efficiently.
3. Add stronger biomedical synonym/entity-normalization baselines such as SapBERT-style or BioSyn-style retrieval for alias-heavy tasks.
4. Add residue-change and gene-context confounders to the hard candidate sets, beyond position-only matching.
5. Start the analogous scaled clinical/lab-value benchmark once accessible data is ready.

## Literature-derived implementation checklist

This checklist translates the strongest adjacent literature into concrete engineering work for the next BioXRep iterations.

### Representation and supervision

1. Upgrade from pair-only contrastive training to equivalence-class-aware batching with multiple positives per anchor.
2. Add auxiliary attribute heads for gene identity, variant identity, residue or base substitution, position, mutation type, unit, and lab status where available.
3. Keep canonical biomedical identifiers such as HGNC ID, GeneID, ClinVar AlleleID, and ClinVar VariationID as structured teacher signals and upper-bound references.
4. Separate surface-form supervision from canonical-field supervision so the model can be evaluated on invariance without discarding factual precision.

### Numeric handling

1. Add an explicit numeric pathway for positions, laboratory values, and converted quantities rather than relying only on character or subword embeddings.
2. Implement a numeric consistency loss for unit-converted or notation-converted values.
3. Add held-out numeric-range splits for both the synthetic and real-data tracks.

### Baselines to add or strengthen

1. Add a SapBERT-style or BioSyn-style learned biomedical synonym baseline for gene alias and name normalization tasks.
2. Keep lexical retrieval baselines for calibration, but do not treat them as the only non-neural reference.
3. Maintain canonical-teacher retrieval as the structured upper bound on real-data tasks.

### Evaluation additions

1. Report notation-held-out, unit-held-out, and numeric-range-held-out results separately from in-distribution retrieval.
2. Add attribute probing or exact-match evaluation for the fields used by auxiliary heads.
3. Measure whether invariance is gained without collapsing biologically meaningful distinctions such as position, residue identity, or abnormality status.
4. Keep the invariance ratio and numeric MAE or RMSE in the core table rather than as optional diagnostics.

### Recommended near-term execution order

1. Build a larger train-only hard-negative dataset for HGVS.
2. Add residue-change and gene-context confounders to same-`protein_position + cdna_position` candidate pools.
3. Add stronger biomedical baselines for alias-heavy tracks.
4. Consolidate hard-set and hard-negative training results into a first paper table.
5. Only after those objective and evaluation changes, test a stronger encoder such as a character Transformer.

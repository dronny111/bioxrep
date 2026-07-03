# BioXRep: Cross-Notation Distillation for Invariant Biological Representation Learning

## One-sentence thesis

Biological information is often expressed through multiple equivalent notations, and models trained only on surface tokens fail to learn the invariant biological object underneath; BioXRep learns notation-invariant representations by distilling alignment signals across parallel biological and numerical systems.

## Motivation

Modern language and biological foundation models still struggle with numerical values, scientific notation, unit changes, aliases, and domain-specific symbolic systems. In biology and medicine, this weakness is not cosmetic. The same biological fact may appear as a genomic coordinate, HGVS notation, protein substitution, gene alias, clinical laboratory value, normalized measurement, or free-text interpretation.

The original 2021 project asked whether a student model could learn universal numerical representations from a transliteration teacher over parallel number systems. BioXRep generalizes that idea: transliteration is treated as one instance of a broader problem, cross-notation alignment.

## Central research question

Can alignment signals from parallel biological notation systems teach a student model to represent equivalent biological facts similarly, while preserving task-relevant distinctions such as value, unit, position, mutation effect, gene identity, and clinical interpretation?

## Core hypothesis

A student model trained with cross-notation distillation will generalize better than ordinary token-based models under notation shift, unit shift, alias shift, and out-of-distribution numeric ranges.

## Why this is timely

Current LLM and scientific foundation model research has exposed three related gaps:

1. Standard tokenization is brittle for numbers and scientific measurements.
2. Biological foundation models often depend on arbitrary representation choices, such as gene order, sequence encoding, value binning, and naming conventions.
3. Biomedical systems need robustness to notation changes because real clinical and biological data is heterogeneous.

BioXRep sits at the intersection of numeracy, biomedical representation learning, cross-lingual/cross-notation distillation, and model robustness.

## Conceptual framing

Biology does not have one language. It has many partially aligned languages:

- DNA sequence
- RNA sequence
- codon sequence
- amino-acid sequence
- protein mutation notation
- genomic coordinates
- gene symbols and aliases
- lab values and units
- normalized omics measurements
- clinical text interpretations

BioXRep treats these as parallel notation systems. The model should learn the biological entity, event, or quantity underneath the surface representation.

## Main paper shape

The first paper should be one method with two biological instantiations:

1. Main track: genomic/protein notation invariance.
2. Generalization track: clinical/EHR lab-value and unit invariance.

This keeps the paper coherent while showing that the method is not limited to a single domain.

## Proposed contributions

1. A cross-notation distillation objective for invariant biological representation learning.
2. BioXRep-Bench, a benchmark of equivalent biological facts expressed across multiple notation systems.
3. A genomic/protein notation benchmark covering gene aliases, HGVS-like variants, protein substitutions, genomic coordinates, and textual descriptions.
4. A clinical lab-value benchmark covering unit conversions, reference ranges, normalized values, and textual interpretations.
5. Evaluation of robustness under held-out notations, held-out units, held-out genes, held-out numeric ranges, and compositional notation shifts.
6. Interpretability analysis showing whether learned representations separate biological identity, numeric magnitude, units, and surface form.

## Method sketch

Each training example is an equivalence class:

```text
fact_id: BRAF_V600E
forms:
  - BRAF V600E
  - BRAF p.Val600Glu
  - NM_004333.6:c.1799T>A
  - chr7:g.<position>A>T
  - valine to glutamic acid substitution at residue 600 in BRAF
attributes:
  gene: BRAF
  protein_position: 600
  reference_amino_acid: V
  alternate_amino_acid: E
  mutation_type: missense
```

The student encoder maps all forms into a shared representation. A teacher or alignment module provides soft signals over token correspondences, entity correspondences, and numeric fields.

Training losses:

- Task loss: prediction, retrieval, classification, or normalization.
- Contrastive equivalence loss: equivalent forms should be close; non-equivalent forms should be separated.
- Distillation loss: student alignments should match teacher alignments or canonical field alignments.
- Numeric consistency loss: values related by unit conversion or notation conversion should preserve magnitude/order.
- Attribute reconstruction loss: representation should retain gene, variant, position, unit, and value attributes.

## Teacher options

1. Seq2seq notation translator:
   Train a teacher to translate one notation form into another and distill its attention/alignment structure.

2. Structured canonicalizer:
   Parse each form into canonical fields and use field-level alignments as supervision.

3. Hybrid teacher:
   Use deterministic biological parsers where available, then use a neural teacher for ambiguous text forms.

The first implementation should start with a structured canonicalizer plus contrastive learning. Neural attention distillation can be added once the benchmark is stable.

## Track A: genomic/protein notation invariance

### Task families

- Variant canonicalization: map multiple variant notations to the same canonical variant.
- Equivalent-form retrieval: retrieve all notations describing the same variant.
- Held-out notation generalization: train without one notation family and test on it.
- Gene alias robustness: resolve gene symbols and aliases to canonical identifiers.
- Mutation attribute prediction: predict gene, position, reference residue/base, alternate residue/base, and mutation class.

### Example equivalence class

```text
BRAF V600E
BRAF p.Val600Glu
BRAF Val600Glu
BRAF c.1799T>A
missense mutation changing valine to glutamic acid at residue 600 in BRAF
```

### Why this track is the conceptual spine

This is closest to the original transliteration idea. DNA, RNA, codons, proteins, and variant notations are parallel symbolic systems with strong biological semantics.

## Track B: clinical/EHR lab-value invariance

### Task families

- Unit-invariant lab representation.
- Lab value normalization across hospitals or reference ranges.
- Equivalent-form retrieval for values and interpretations.
- Abnormality classification under unit shift.
- Clinical text grounding: align text descriptions such as "elevated creatinine" with numeric values and reference ranges.

### Example equivalence class

```text
glucose 126 mg/dL
glucose 7.0 mmol/L
fasting blood glucose is elevated
glucose above diabetes diagnostic threshold
```

### Why this track matters

It provides practical healthcare impact and tests whether BioXRep generalizes beyond genomic notation into numerically grounded biomedical data.

## Baselines

- Character-level encoder.
- Subword transformer encoder.
- Digit-level numeric tokenization.
- Scientific notation tokenization.
- xVal-style continuous numeric token placeholder.
- Contrastive learning without distillation.
- Canonical field prediction without contrastive equivalence.
- Teacher-only translator without student representation learning.

## Primary metrics

- Top-k equivalent-form retrieval accuracy.
- Canonicalization accuracy.
- Attribute F1 or exact match.
- Numeric error after unit conversion.
- OOD generalization gap under held-out notation or held-out unit.
- Embedding invariance score: within-equivalence distance versus between-equivalence distance.
- Calibration under numeric ranges and notation shifts.

## Interpretability checks

- Probe whether embeddings encode gene identity, mutation position, amino-acid substitution, unit, magnitude, and abnormality status.
- Test whether unit information is retained separately from biological quantity.
- Visualize equivalence classes in embedding space.
- Ablate alignment/distillation losses and measure which invariances disappear.

## First implementation milestone

Build a synthetic-but-biologically-grounded benchmark before depending on restricted biomedical datasets.

Minimum viable benchmark:

1. Generate protein substitution examples from a curated list of genes and amino acids.
2. Generate multiple notation forms for each example.
3. Generate lab-value examples with unit conversions and reference ranges.
4. Train simple encoders with contrastive and attribute losses.
5. Compare against character and subword baselines.
6. Evaluate held-out notation and held-out unit generalization.

This gives a fast experimental loop while preserving a path to real datasets later.

## Near-term decision

The first code milestone should not attempt to repair all 2021 training code. Instead:

1. Keep the original code as historical prototype.
2. Add a clean `bioxrep/` package.
3. Add deterministic data generators for Track A and Track B.
4. Add a minimal encoder and benchmark harness.
5. Only then reintroduce teacher attention distillation.

## Annotated bibliography

### Numeracy and scientific text

- **xVal: A Continuous Number Encoding for Large Language Models**
   Why it matters: argues that plain token embeddings are a weak default for quantitative reasoning.
   Design implication: add explicit numeric machinery for positions, values, and converted quantities instead of treating them only as characters or subwords.

- **Language Models Do Not Embed Numbers Continuously**
   Why it matters: shows that standard language-model representations do not preserve numeric proximity well.
   Design implication: BioXRep should include numeric consistency losses or magnitude-aware encoders when the task depends on value fidelity.

- **NumericBench** and related numeracy benchmarks
   Why they matter: provide evaluation patterns for held-out numeric ranges, formatting changes, and out-of-distribution quantity understanding.
   Design implication: evaluate notation invariance separately from numeric generalization instead of relying only on aggregate retrieval metrics.

### Multi-view contrastive learning

- **SimCLR**
   Why it matters: establishes that the quality of views and positive pairs is often more important than scaling the encoder early.
   Design implication: the main supervision object in BioXRep should be the equivalence class, not isolated string pairs alone.

- **Supervised Contrastive Learning**
   Why it matters: directly matches the setting where multiple surface forms map to one underlying fact.
   Design implication: move from pairwise contrastive training toward class-aware batching with multiple positives per anchor.

- **CLIP** and later multi-view alignment work
   Why it matters: shows that heterogeneous aligned views can share a useful embedding space when each view preserves different signal.
   Design implication: treat protein HGVS, nucleotide HGVS, aliases, structured IDs, and clinical text as aligned but distinct views of one object.

### Biomedical entity normalization

- **BioSyn**
   Why it matters: a strong biomedical synonym-normalization baseline built around dense retrieval and synonym supervision.
   Design implication: BioXRep should compare against at least one learned biomedical synonym baseline for alias-heavy tasks.

- **SapBERT**
   Why it matters: demonstrates that metric learning over synonym sets and ontology-linked names produces robust biomedical concept embeddings.
   Design implication: SapBERT-style training is a better reference baseline than lexical retrieval for HGNC alias and concept-normalization slices.

### Variant and clinical normalization

- **tmVar / tmVar 2.0** and related variant-normalization work
   Why it matters: shows that exact variant identity still depends heavily on structured parsing and normalization.
   Design implication: canonical biomedical identifiers should remain the teacher or upper bound for variant tasks.

- **GA4GH variation representation and normalization**
   Why it matters: gives the cleanest standards-based notion of when two variant forms refer to the same underlying event.
   Design implication: stronger canonicalization produces cleaner equivalence classes and more defensible supervision.

- **Med-BERT**, **BEHRT**, **CLMBR**, **G-BERT**, and related clinical foundation model work
   Why they matter: define strong representation-learning baselines for structured clinical data while leaving unit-invariant scalar semantics relatively underexplored.
   Design implication: the lab-value track should be framed as invariance under unit and wording shift, not just generic EHR representation learning.

### Reading order for project decisions

1. SapBERT.
2. BioSyn.
3. xVal.
4. Language Models Do Not Embed Numbers Continuously.
5. SimCLR.
6. Supervised Contrastive Learning.
7. tmVar and related variant-normalization work.
8. GA4GH variation representation and normalization work.
9. Clinical foundation model papers for the lab-value framing.

### Synthesis

Taken together, the literature favors a hybrid structured-neural approach: multi-view contrastive learning over full equivalence classes, structured supervision from canonical IDs or parsed fields, auxiliary attribute prediction, explicit numeric handling, and held-out notation or unit evaluation. That is the nearest adjacent SOTA pattern for the problem BioXRep is trying to define.

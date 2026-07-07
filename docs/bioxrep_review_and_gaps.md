# BioXRep: Work-So-Far Review and Gap Analysis

A review of the project state as of this snapshot: research brief, staged roadmap,
the two frozen results docs (HGNC alias, HGVS), the `bioxrep/` package, the paper
draft (`main.tex`), and the saved artifact store.

## What's solid

The project is in good shape as a **benchmark-and-honest-baselines** effort.

- **Two real biomedical benchmarks built from public sources.** HGNC gene-alias
  equivalence classes (44,997 classes -> ~387k forms) and ClinVar HGVS
  protein<->nucleotide classes (29,076 train / 7,270 test), both with genuinely
  fact-disjoint splits and a `verify_no_leakage.py` guardrail that fails loudly
  on fact *or* notation leakage.
- **Rigorous evaluation design.** The HGVS "strict hard pool" (decoys share the
  query's parsed `protein_position` + `cdna_position`) is the standout
  contribution: it correctly exposes numeric-position features as shortcuts
  (top-5 ~= 1.0 but hard top-1 *drops*), and the matched-only projection shows
  the text-only byte encoder strongest once random-fill decoys are removed
  (0.9195 vs 0.906 xVal, 0.900 explicit, and 0.498 sinusoidal).
- **Intellectually honest reporting.** The HGNC char-CNN student is written up as
  a clean negative result (0.059±0.001 MRR over 5 seeds, below both the lexical
  floor 0.077 and SapBERT dense 0.134), with the in-domain 0.985 top-1 shown as the trap a
  disjoint split avoids.
- **A near-submittable short paper** (`main.tex`) with CrossRef/OpenAlex/arXiv-
  verified references, reframed around the negative result.

## Gaps (roughly by severity)

### 1. The full neural namesake method is not implemented (highest severity)
The repo is *"...by attention distillation,"* and the brief's Phase 3 core
hypothesis is that distilling a neural teacher's cross-attention/alignment into
the student produces notation invariance. The code now includes an optional
deterministic byte/digit alignment teacher and bidirectional attention KL loss in
`bioxrep.train.train_contrastive_student` (`--attention-distillation-weight`), so
the repo has an implemented alignment-distillation baseline. It now also includes
a **trained neural teacher** path (`--attention-teacher-checkpoint`): a frozen,
higher-capacity char-CNN student (valid top-1 0.677 vs the student's 0.582) whose
learned attention `softmax(f_s f_t^T / sqrt(d))` is the KL target. A three-arm
ablation (contrastive-only / byte-rule teacher / neural teacher, identical student
config, seed 13) has now been **run and recorded** (`outputs/ab_eval/`,
`ab_summary_3arm.json`): held-out `alias_symbol` MRR 0.050 / 0.054 / 0.050 with
overlapping bootstrap intervals, and invariance ratio *worsening* monotonically
0.753 -> 0.769 -> 0.804. **Neither teacher closes the gap.** The still-absent
variant is distillation from a *semantically pretrained* teacher (SapBERT-strength),
which the surface-only attention teachers here do not supply. The central thesis
is therefore no longer *under-instrumented*: the namesake mechanism was tested in
both heuristic and trained-neural forms and *did not help*; the limitation is the
surface-only student's representational capacity, not the teacher's quality.

### 2. There is no positive method result
HGNC is a negative result; the HGVS "win" is a scaling/ablation finding (text-only
+ more data beats numeric shortcuts), not evidence the *proposed* objective beats
a strong off-the-shelf baseline. No experiment shows a BioXRep-trained encoder
beating SapBERT on a disjoint task. As written, the affirmative contribution is a
benchmark + a negative result — a legitimate BioNLP paper, but not the method
paper the brief describes.

### 3. Track B numeric-consistency loss — TESTED, negative (was: scaffolding only)
**Update.** The numeric-consistency loss (`--numeric-loss-weight`) is now
evaluated on a real HGVS cross-notation task (ClinVar, protein↔nucleotide
expression) with a position-confounded hard benchmark. Holding input to text-only
and varying only the loss weight (0.0 / 0.1 / 0.5, 5 seeds' worth of scaffolding
reused), hard top-1 is `0.7275 → 0.705 → 0.7145` — all CIs overlap, so the
auxiliary loss does **not** improve hard retrieval. Combined with the corrected
numeric-*feature* ablation (explicit, sinusoidal, and xVal-style position inputs
all fail to beat text-only on the strict matched-only pool), Track B's numeric
objective is a closed negative result: shared position is a confound, not a
discriminator. See `docs/bioxrep_hgvs_results.md`
(§Numeric-consistency-loss ablation) and `scripts/run_trackb_numeric_loss.sh`.
A pipeline fix landed alongside: the `_numeric_50k` source had 702 duplicate
`fact_id`s that could leak across the split; dedup-by-`fact_id` restores a
fact-disjoint split (leakage check passes).

Still genuinely absent: real MIMIC-IV lab-value / unit-held-out results. The
brief's "one method, two instantiations" shape now has the HGVS numeric
instantiation tested (negative), not the lab-value one.

### 4. Interpretability and invariance metrics are only partly implemented
The brief promises attribute probing, an invariance ratio (between-class /
within-class distance), and embedding visualization. The trainer *has*
attribute/numeric heads, and `bioxrep.eval.invariance_ratio` now computes the
distance ratio for saved student checkpoints. The attention-distillation table in
`main.tex` reports invariance ratio for the three HGNC arms; broader probing
tables and embedding visualizations are still absent.

### 5. Statistical rigor — partly addressed (was: none in the frozen tables)
Both docs mention `--bootstrap` CIs and multi-seed sweeps as *available*, and most
headline numbers were single-seed point estimates. **Update.** The
attention-distillation ablation is now reported as mean ± std over 5 seeds
(`{13,17,23,42,101}`, `scripts/run_multiseed_ablation.sh`,
`outputs/multiseed/aggregate_multiseed.json`): MRR `0.055±0.002` (contrastive) vs
`0.056±0.002` / `0.056±0.001` (byte-rule / neural) — the ~0.001 gap is within one
seed σ, strengthening the negative over the earlier single-seed bootstrap. The
Track B numeric-loss arms carry bootstrap CIs. **The main HGNC results table is now
multi-seeded too**: the headline held-out `alias_symbol` char-CNN student row is the
mean over 5 seeds (`{13,17,23,42,101}`, `scripts/run_multiseed_hgnc_main.sh`,
`outputs/multiseed_hgnc/aggregate_hgnc_multiseed.json`) — MRR `0.059±0.001`
(top-1 `0.041±0.002`, top-5 `0.075±0.001`), still below the char n-gram lexical
floor `0.077` and SapBERT `0.134` by >15 seed σ, so the headline negative is robust
to seed. The deterministic lexical/SapBERT/BioSyn rows are seed-invariant; the
train-seen (`†`) student rows remain single-seed (not the headline claim).

### 6. Reproducibility gap between docs and stored artifacts
The HGNC comparison outputs and HGNC student checkpoint are saved as artifacts,
and the current HGVS/Track-B JSONs and checkpoints exist locally under
`outputs/trackb*`. Because `outputs/` is git-ignored, an auditable paper release
should commit or otherwise publish the small result JSONs and logs (not the large
checkpoints) so the HGVS tables are backed by durable provenance.

### 7. Baseline coverage is narrower than the brief
Implemented: char n-gram, char-CNN, SapBERT dense, BioSyn hybrid, canonical
teacher. **RESOLVED (numeric side):** the numeric input-feature ablation now runs
four arms on the corrected deduped split — `none` (text-only byte encoder, which
doubles as the **digit-token** baseline since each digit is its own byte),
`explicit` scalar, `sinusoidal` Fourier, and an **xVal-style continuous numeric
baseline** (Golkar et al. 2023) — closing the "xVal cited but not benchmarked"
gap. Finding: on the full 20-candidate pool the value-aware features look like wins
(`explicit` 0.834, `xval` 0.846 vs text-only 0.680 hard top-1), but each pool has
only ~2.25 position-matched decoys against ~16.75 random-fill decoys, so that gain
is trivial random-decoy rejection. On the **strict matched-only pool** (positive
scored against only its position-matched hard decoys) the ranking inverts:
text-only **0.9195** [0.908, 0.930] > `xval` 0.906 > `explicit` 0.900, and
`sinusoidal` collapses to 0.498. The honest negative — value-aware numeric features
do not help discriminate position-confounded hard negatives — holds across xVal.
Full numbers in `docs/bioxrep_hgvs_results.md`; eval JSONs
`outputs/trackb_featmode/eval/hard_feat_*.json` and `strict_matched_only.json`.
Still missing from the brief: a subword-transformer encoder baseline.

### 8. Tests / CI are minimal
A lightweight `pytest` suite and GitHub Actions workflow now cover retrieval
metrics, invariance-distance helpers, and the attention-distillation math. This
closes the complete-absence gap, but coverage is still narrow and does not yet
exercise public-data builders or end-to-end training runs.

### 9. Task well-posedness worth a sentence
On `alias_symbol` every method scores MRR <= 0.134, and the canonical teacher
trivially scores 1.000 by matching the structured `symbol` field it's keyed on. A
short headroom/ceiling analysis would preempt the reviewer question of whether the
hardest task is near-unsolvable as posed.

### 10. Minor hygiene
Real PhysioNet credentials sit in plaintext at `bioxrep/.env`. Verified: it is
gitignored and was **never committed** (low risk), but it lives in a writable
workspace folder. A root `.env.example` with empty keys is now present for safer
setup.

## Code audit addendum (verified against the repo, not the artifact snapshot)

A direct read of `bioxrep/` confirms the reconciliation above and adds four
implementation-level findings:

- **The distillation teacher is a fixed heuristic, not a learned model.**
  `byte_alignment_teacher_probs` builds the target attention distribution by rule:
  exact byte match gets mass `2.0`, any digit-to-digit match `1.0`, and rows with
  no match fall back to uniform over valid target positions. The student attention
  is `softmax(source_features . target_features^T / sqrt(d))` over token features,
  distilled bidirectionally (L->R and R->L averaged) via KL. This is genuine
  attention-*map* distillation in structure, but the "teacher" carries no learned
  parameters. If the title keeps "by attention distillation," the rule-based vs.
  neural-teacher distinction must be explicit or it reads as an overclaim.
- **Both auxiliary losses default to off; the attention path is now exercised.**
  `--attention-distillation-weight` and `--numeric-loss-weight` both default to
  `0.0`. The attention-distillation path is no longer only wired and unit-tested:
  the three-arm ablation above switches it on (weight 0.1) for both the byte-rule
  and neural-teacher forms and records the result. Gap #3 (numeric loss, Track B)
  remains "under-instrumented, not disproven": no frozen result exercises
  `--numeric-loss-weight > 0` yet.
- **Byte-offset digit check is correct (not a bug).** The teacher tests
  `id in [ord("0")+1, ord("9")+1]`; `encode_text_tensors` maps every byte to
  `byte + 1` (reserving `0` for `padding_idx`), so the `+1` offset is intentional
  and consistent end-to-end.
- **`CharCNNEncoder.forward` has an undocumented channel-coupling assumption.**
  The pooling loop strides by `self.convs[0].out_channels`, implicitly requiring
  every conv to share `out_channels == hidden_dim`. It holds under the current
  constructor but would silently mis-pool if kernel widths ever had heterogeneous
  channel counts; an assert or comment would harden it.

Test coverage is real but shallow: the suite exercises teacher-probability math,
loss finiteness, ranking metrics, and invariance-helper determinism (the pure
functions), but touches neither the data builders (`build_*`, `fetch_public`) nor
any end-to-end training step. CI installs torch/sklearn and runs `pytest -q` on
push and PR.

One live-hygiene note: `bioxrep/.env` (59 bytes, real PhysioNet credentials) still
exists in the writable workspace folder. It is untracked and gitignored (low leak
risk), but now that `.env.example` exists the real file no longer needs to persist
between sessions.

## Suggested priorities

1. **Done — neural attention-distillation teacher tested.** A frozen
   higher-capacity char-CNN teacher was trained and its learned attention distilled
   into the student; the three-arm ablation (contrastive / byte-rule / neural) is
   recorded in `outputs/ab_eval/` and written into `main.tex` (§ ablation). Neither
   teacher closes the disjoint `alias_symbol` gap. The remaining, higher-cost
   variant is distillation from a *semantically pretrained* (SapBERT-strength)
   teacher on the same byte grid — the surface-only teachers tested here do not
   substitute for it.
2. **Polish for the current paper:** keep the release-facing docs synchronized
   with the multi-seed/CI tables, and ensure the invariance-ratio definition is
   consistently described as between/within (higher is better).
3. **Provenance:** publish the small HGVS result JSONs/logs in a committed or
   release artifact location so the flagship table is durable outside local
   `outputs/`.
4. ~~**Add the xVal baseline** — cited as motivation but not benchmarked.~~
   **DONE** — benchmarked in the 4-arm feature-mode ablation (see section 7);
   negative holds on the strict matched-only pool.

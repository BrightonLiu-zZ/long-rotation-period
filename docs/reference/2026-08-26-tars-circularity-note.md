# The TARS circularity — why today's rotation score cannot answer Theissen's question

**Written:** 2026-08-26 · **Task:** T0 (Lane R) · **Feeds:** R6' (Theissen update) and the meeting he
offered · **Evidence trail:** Q17 in `experiments/proposed_open_questions.md` (lines 624–713)

## 1. The circularity, stated plainly

Asked for a TESS-appropriate astrophysics question, Theissen named *measuring rotation periods below
what is already done, and potentially longer than the observing baseline*, and cited **Boyle, Bouma &
Mann 2026, ApJS 284, 75** (arXiv:2603.05586) — the TESS All-Sky Rotation Survey, **TARS** — as the bar.

TARS is also our rotation ground truth (`docs/labels-sources.md` §TARS, `tars_table_2.feather`). The
catalogue he named as the bar is the catalogue our labels come from. Three consequences:

    our rotation score = agreement with TARS
    perfect agreement  = TARS
    -> the current evaluation cannot in principle demonstrate "below what is already done"

- A star the model calls a rotator that TARS does not is **scored as a false positive today**. The
  detections the question asks for are currently counted as errors.
- The mechanism is already written down: absence from TARS means *"no detectable rotation period in a
  single-sector TESS baseline"*, not "confirmed non-rotator". That is detection incompleteness, and
  `labels-sources.md` (known limitations, §Rotation) says it "caps the achievable upper bound".
- T1 measures the size of the affected set: **104,827** of 119,754 corpus stars with windows are
  TARS-null. Every one of them is a negative in today's eval and a candidate under his question.

## 2. Two different baselines are being conflated

The phrase "longer than the time baseline" refers to two different quantities:

| baseline | length | who it belongs to | status |
|---|---|---|---|
| observation baseline | one sector, ~27 d (TARS recovers to 25 d) | TARS | beating it needs multi-sector stitching — forbidden by a CLAUDE.md hard constraint; would need an ADR (**T5**) |
| **model input baseline** | `4 windows x 1024 cadences x 2 min` = **5.69 d** | our encoder | **labels already exist for periods no tensor the encoder sees contains one full cycle of** |

**Limb A** targets the model's baseline and is evaluable now. **Limb B / T5** target TARS's and are not.
Any sentence sent to Theissen must name which one it means.

## 3. Three routes out of the circularity (limb B)

A claim of detections *below TARS's threshold* needs a reference TARS cannot provide:

1. **Injection–recovery** — inject synthetic spot modulation below TARS's amplitude/period threshold
   into corpus light curves, measure recovery against a Lomb–Scargle/TARS-like baseline. Shares
   machinery with the synthetic-injection *selection* protocol already on the table from the Yue Ma
   call: pay once, use twice.
2. **An independent catalogue on the overlap** — Kepler/McQuillan, ground-based, or spectroscopic
   `v sin i`.
3. **Hand-vetting** a sample of model-positive / TARS-null stars.

Available and unused: **`tars_table_4`** (~40 GB) identifies stars TARS *analysed and found no signal
in* — a genuine high-confidence-negative subset, currently out of v1 scope. Our `rotation = 0` is
absence-from-catalogue, which is weaker.

## 4. What this means for the paper (W7 disposition)

Yue Ma's W7 point — Tmag < 10 stars are already well studied, so "identify unlabeled data" is dead as a
selling point — is accepted **as a narrowing, rejected as a killing**. The label catalogs are
incomplete *by construction* (Villanova = S1–26 only; TARS is threshold-limited; Gao is ML-derived;
flatwrm2 is precision-first), so model-positive / catalog-null stars are **candidates, not false
positives**; today's eval merely scores them as FP. The discovery claim is therefore **not dead, it is
unprintable until one of the three routes above exists** — recorded as a v2 route in
`docs/roadmap/2026-08-25-ml4ps-pivot-roadmap.md` §1 (W7) and §Deferred. The ML4PS paper claims
classification improvement only.

## 5. What is evaluable now, and what T0 feeds

**T2 (limb A)** is the part we can run on cached µ with no new training. T1 sized it: **4,679** corpus
rotators carry a TARS period beyond the model's 5.69 d baseline, **4,546** of them reachable in the R8
`r8_scoreable` pool (3,357 in 5.7–13 d, 1,189 at ≥13 d). The frozen v1 subset holds only 158, of which
25 sit in its test split — so T2 scores on the R8 pool and reports the v1 subset as a contrast row.

Three controls, none skippable:

| question | control | what its outcome means |
|---|---|---|
| is this better than a random encoder? | untrained-encoder floor | below it, there is no result |
| is this period inference or activity amplitude? | **amplitude-only baseline** | mandatory: `evr_pc1` = 0.926, and residualising amplitude already costs rotation 0.558 → 0.424 |
| how much of the recoverable signal did we get? | A1 engineered-feature ceiling | the gap, not the absolute, is the claim |

The **G-rot** gate is already pre-registered in `docs/roadmap/2026-08-15-post-yue-ma-roadmap.md` and is
referenced here, not re-registered. Its partial-pass branch — beats untrained but not amplitude-only —
is reported as *"the beyond-baseline signal is amplitude/activity, not period inference"*: a real,
printable negative. `docs/architecture.md` pre-registers the same fallback for the stratified reading:

> Monotonic degradation across buckets is a paper finding in its own right — it demonstrates that
> encoder behavior matches the physical limit of the input rather than overfitting to label artifacts.

**Scope note:** ADR-0004 caps v1 rotation metrics at `P ≤ 5 d` precisely to avoid this regime, and Q13
proposes *enforcing* that cap. T2 deliberately crosses it. The two must be decided together — that is
**T3**, and it is not done yet.

**This note feeds:** R6' (Theissen update) and the meeting. It does not commit us to limb B.

# Handoff: the new-project line with Prof. Theissen (rotation periods, or new planets)

**Written:** 2026-10-02. **For:** a fresh session whose job is to prepare what Brighton discusses with
Prof. Theissen about the **next** project. This is explicitly **not** the ML4PS/arXiv line.

You have latitude here. This document is context, not a task list. Decide for yourself what to read,
what to re-verify, and what to produce.

---

## 1. The state of this line in one paragraph

Theissen asked for a project that works backward from a clarified higher-level astronomy question,
and named two candidates: **predict rotation periods beyond the light-curve length limit**, and
**find new planets** (he framed the second as a subtype and called the area saturated). A full
20-minute meeting brief was written on 2026-09-10 and **has not been delivered or revised since**.
Brighton's current steer: rotation is provisionally chosen, planets are **not** fully dropped.

**The brief is the starting artifact, not this file:**
`tmp/handoff/2026-09-10-theissen-next-direction-prep.md` (151 lines). It holds the three-baseline
restatement of his question, a seven-row state-of-the-art table, the proposed framing and method, the
label-free evaluation, the two pre-registered outcomes, a 20-minute script, four domain asks for him,
and a sourced reference list. Read it first. Do not re-derive what is already in it; do re-check
anything it asserts that has aged.

---

## 2. Decisions Brighton has already made (treat as settled unless he reopens them)

- **New repo, new corpus.** The v1 hard constraints in this repo's `CLAUDE.md` (SPOC 2-min only,
  PDCSAP, no stitching, frozen architecture) do **not** carry over.
- **The world model is the method under test**, other methods are baselines. But the posture in the
  meeting is "work backward from the astro problem, I am flexible on the model", because that rule is
  Theissen's own.
- **Direction criterion:** pick where self-supervision is most likely to actually work.
- **Compute:** undecided. Local RTX 4060 is the floor; he will try for the UCSD HDSI cluster.
- **Time budget:** never fixed. Worth settling, since it decides corpus size.

---

## 3. What changed since the brief was written, and why each matters here

**ML4PS was submitted on 2026-09-19** (deadline was extended one week from the 12th). **Decision is
due 2026-10-10**, so as of today it is still pending and you should not assume an outcome. The
workshop is non-archival and, this year, not affiliated with NeurIPS. An arXiv version, four new
appendices, a code release and a tagged public repo were all completed between 09-20 and 09-27. Prof.
Theissen and Yue Ma are both co-authors.

**Theissen reviewed the draft on 2026-09-19 and this tells you how to pitch him.** Verbatim notes are
in `tmp/Chris Theissen slack suggestion 9_19.md`. Three things to carry into the new-project pitch:

- He corrected a physics framing that we had invented: *"TESS doesn't trade sampling rate for sky
  coverage. The sampling rate is even higher than Kepler... What it does trade for sky coverage is
  spatial resolution, which is why contamination is a bigger problem for TESS than for Kepler."*
  Expect him to check causal claims about instruments, not just numbers. Contamination being worse in
  TESS than Kepler is also a real design consideration for any faint-M-dwarf FFI corpus.
- He said the draft never set up *"why do we care"*, and called that *"more of an astronomer question
  than a data scientist question."* For a project he is being asked to co-own, the astrophysical
  motivation should lead, not follow the method.
- He flagged vagueness hard (*"I don't even know what the gap is"*). Quantities and stakes beat
  adjectives with him.

**New evidence about rotation from our own work (2026-09-26), which cuts against the easy story.**
The arXiv Phase 1 rotation-pool analysis built an unconfounded pool and re-scored it. Rotation-period
regression gain **+0.034 ± 0.002** counts; rotation classification **+0.055 ± 0.006** but the
untrained control also gained **+0.027 ± 0.003**, so by the pre-registered rule it does **not** count;
XGBoost fusion gain was negative on both rows, and fusion sat behind a plain MLP on rotation period
(0.752 vs 0.773). See `docs/STATUS.md` (top entry), `docs/plans/2026-09-20-rotation-pool-design.md`
and `experiments/rotation_pool/`. This is the most recent measurement of how well our approach does on
rotation specifically, and the brief predates it. Decide for yourself whether it strengthens the
"the current stack cannot answer his question, so build a new one" argument, or weakens the case that
self-supervision is the right instrument for rotation at all. Either reading is defensible; what is
not defensible is pitching rotation without having looked at it.

---

## 4. The one trap specific to this line

Our rotation labels come from **TARS**, and TARS is also the bar Theissen named. A star our model
calls a rotator that TARS does not is scored today as a false positive, so the detections his question
asks for are counted as errors. Written up in `docs/plans/2026-08-26-tars-circularity-note.md`, with
the evidence trail at Q17 in `experiments/proposed_open_questions.md`. A related in-house measurement
(T2) found the encoder's period estimate saturating at the input length, with amplitude outranking the
latent beyond it. The brief's proposed fix is independent truth (Kepler, ZTF) plus a label-free
forecasting gate. That fix should survive scrutiny, but it is the part most worth attacking before he
does.

---

## 5. Open questions you may need to resolve with Brighton

- **When is the meeting**, and is it a 20-minute slot like last time? The brief's script assumes 20.
- Does the ML4PS outcome on 10-10 change the pitch, or is this independent of it?
- How hard is the commitment to rotation? He says planets stay on the table; the brief parks them in
  one paragraph (world model as the stellar-variability noise model, against `nuance` as baseline).
- Time budget and compute, from section 2.

---

## 6. Where things live

| What | Where |
|---|---|
| The meeting brief (start here) | `tmp/handoff/2026-09-10-theissen-next-direction-prep.md` |
| TARS circularity write-up | `docs/plans/2026-08-26-tars-circularity-note.md` |
| Rotation pool design + results | `docs/plans/2026-09-20-rotation-pool-design.md`, `experiments/rotation_pool/`, `docs/STATUS.md` |
| Theissen's 2026-09-19 review notes | `tmp/Chris Theissen slack suggestion 9_19.md` |
| Open-questions ledger (Q17, Q18) | `experiments/proposed_open_questions.md` |
| Career/grad-school hub (separate repo) | `C:\BrightonLiu\grad_app_and_career` |
| Memory for this line | `project_next_direction_rotation_beyond_baseline.md`, `project_theissen_astro_question.md` |

Handoff docs for this project belong in `tmp/handoff/`, by the user's standing preference.

---

## 7. Working style that applies

- Answer in caveman style, per the user's global instructions. Full technical accuracy, no filler.
- He asks for 3 to 6 critical questions with a recommended option attached to each, rather than open
  menus. He pushes back when a recommendation is unexplained.
- Web search is encouraged on this line; the brief's SOTA table came out of it. Astronomy moves, and
  several of those papers are recent.

## 8. Suggested skills

- `theissen-report` (project skill): the clarification gate for anything going to Theissen.
- `read-the-methods-section`: before any claim about how an external catalogue, survey or benchmark
  produced its data. This line is dense with them, and one such claim has already been corrected by
  him in print.
- `option-set-hygiene`: when building the option set for his decisions.
- `grilling`: to stress-test the proposed framing before he does.
- `answer-the-question-asked`: for anything long or table-heavy.

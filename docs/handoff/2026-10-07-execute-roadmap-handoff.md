# Handoff: execute the long-rotation-period roadmap, step by step

**Written:** 2026-10-07. **For:** a fresh agent whose job is to run `docs/ROADMAP.md` Steps 1→6 in
order, stopping where this document says to stop.
**Repo:** `C:\git_repo\long-rotation-period` (git-initialised, **no commit yet**, all files untracked).

---

## 0. Read these first, in this order

1. `CLAUDE.md` (repo root) — constraints. Short.
2. `docs/ROADMAP.md` — the six steps, the Step 3 decision table, and two honesty rules for Steps 4/5.
   **This handoff does not repeat it; it adds what you need to execute it.**
3. `docs/meetings/2026-10-02-theissen-summary.md` — what Prof. Theissen decided, with transcript
   timestamps. The raw transcript is `docs/meetings/2026-10-02-theissen-transcript.txt`; its speaker
   labels are **swapped** ("Instructor" = the user, Brighton; "Student" = Theissen).

Optional background only: `docs/prep/` (pre-meeting research for an image/pixel idea Theissen did
**not** adopt) and `docs/reference/` (copies from the old repo).

---

## 1. The situation in one paragraph

Theissen (UCSD astronomer, informal collaborator, not an ML engineer) asked whether rotation periods
longer than the TESS observing window can be recovered. The user wanted to work on TESS **images**
(to gain computer-vision / video world-model experience). In the 2026-10-02 meeting Theissen said:
start with **SPOC light curves**, images only if the light-curve route proves insufficient. The
user's response (2026-10-03): spend **as little time as possible** on this project, do it roughly
well, and steer back toward CV/WM **only if the evidence supports it**. The honest route back is
Roadmap Step 3 (SAP vs PDCSAP). Do not bias Step 3 toward the outcome the user would like.

---

## 2. Decisions already made — do not reopen

| decision | source |
|---|---|
| Data = TESS SPOC 2-min light curves. No FFI, no TESScut, no image work. | Theissen, transcript 32:57, 36:00 |
| Method = model predicts continuation of a light curve → extended curve → periodogram. He calls it "GAN"; user will try **both** a reused SWM forecaster (Step 4) and a real small GAN (Step 5). | transcript 46:07–49:28; user answer 2026-10-03 |
| Baseline = Lomb–Scargle (LS) periodogram, take the peak. | transcript 42:40 |
| Short-period truth = TARS. **Long-period truth = MEarth** (Newton+2016 north, Newton+2018 south). TARS is never used as long-period truth. | transcript 43:53; `docs/reference/2026-08-26-tars-circularity-note.md` |
| Validate inside TESS only: held-out stars; first half of a sector predicts the second half. No other surveys. | transcript 09:59–12:24, 22:58 |
| Don't restrict to M dwarfs (MEarth happens to be M dwarfs; fine). | transcript 18:34 |
| No Earth/Moon geometry input. But flag any period near 13.7 d as a likely systematic. | transcript 20:46–22:02 |
| Compute: build locally on ~10 stars first, estimate storage + time; Theissen can add it to his supercomputer allocation later. | transcript 14:06–16:11 |
| New repo, independent of Stellar-World-Model (SWM). SWM hard constraints do not apply. Copy code, never import across repos. | user, 2026-10-03 |
| User's weekly time budget: **not fixed**. Keep each step small. | user, 2026-10-03 |

**Two questions the user has not answered yet — use these defaults, and say so in your report:**
- Phase 1 = Steps 1–3, then draft a Slack update to Theissen (Step 6 brought forward). Default: yes.
- "Reuse SWM model" = **copy** the model code into this repo and retrain on new data. The SWM
  checkpoint is not reusable as-is: it was trained on 5.69-day inputs of MAD-normalised PDCSAP.

---

## 3. Environment and existing assets

**Conda envs (Windows):**
- `astro` — data work (astroquery, lightkurve, astropy, pandas, numpy, scipy, tenacity). CPU torch.
  Invocation that works on this machine: `conda.exe run -n astro python -u ...` from PowerShell.
  Do not call a python.exe directly.
- `swm` — training (`C:\Users\user1\miniconda3\envs\swm`, torch 2.5.1+cu121, RTX 4060 8 GB). Has
  hydra + wandb. Machine has 24 GB RAM.

**Reusable from `C:\git_repo\Stellar-World-Model` (read, copy what you need, do not write there):**

| asset | path | note |
|---|---|---|
| TARS table | `data/tars_table_2.feather` (231 MB) | columns `TICID`, `adopted_period`; loader example `src/build_variability_labels.py` lines 176–192 |
| SPOC FITS download (MAST curl scripts, retry, resume) | `src/build_sequences_bulk.py` (`fetch_curl_script`, `download_fits`, `read_fits`, `_call_with_retry`) | it reads only PDCSAP and **deletes the FITS after use** (line 482) — you need SAP_FLUX too, so keep the FITS this time |
| SPOC sector map | `processed/spoc_sector_map.csv` (`tic_id,tmag,sector`) | **Tmag < 10 only.** MEarth mid/late M dwarfs are mostly fainter → this map will miss most of them. Query MAST yourself. |
| Forecasting model | `src/swm/models/{encoder,decoder,dynamics,world_model}.py`, training loop `src/swm/train/` | Conv1D-VAE encoder + GRU dynamics; rollout exists |
| Retry/resume pattern for network loops | project skill `astroquery-resilience` (SWM `.claude/skills/`) | mandatory for any batch query/download |

There is **no code yet** in this repo. Suggested layout (keep it small): `src/` scripts,
`data/` (gitignored), `results/` (small CSVs + PNGs, committed).

---

## 4. Step-by-step execution notes

The ROADMAP has the "what". Below is the "how" and the traps.

### Step 1 — truth table
- MEarth periods: VizieR. Newton+2016 north = 387 stars, 0.1–140 d (https://arxiv.org/abs/1511.00957,
  VizieR `J/ApJ/821/93`, verify the exact table id before relying on it). Newton+2018 south:
  https://arxiv.org/abs/1807.09365 — read its abstract/VizieR entry for the count and period range;
  not yet verified.
- Cross-match to TIC (by coordinates; account for proper motion — these are nearby, high-PM stars).
- For each star: which sectors have **SPOC 2-min** light curves (MAST `Observations` query,
  `obs_collection="TESS"`, `dataproduct_type="timeseries"`, provenance SPOC). Never pass
  `columns=` to `Catalogs.query_criteria()` (raises `InvalidQueryError`); filter in pandas.
- Short-period control: ~50–100 TARS stars with P < 10 d that have SPOC 2-min data.
- **Output:** `results/step1_truth_table.csv` + a short count summary: N MEarth stars, N with ≥1 SPOC
  sector, sectors per star, period histogram in bins < 13.7 / 13.7–27 / > 27 d.
- **Stop and report** if fewer than ~30 MEarth stars with P > 13.7 d have SPOC data. The whole long-
  period evaluation depends on this number; the user and Theissen need to know before more work.

### Step 2 — LS baseline on PDCSAP
- Download SPOC FITS for the Step 1 stars; **keep them** (`data/fits/`, gitignored).
- Per star-sector: QUALITY == 0, drop NaN, normalise by median, LS (astropy `LombScargle`) over
  roughly 0.1–100 d; take the peak. Also run on all sectors of a star **combined with a per-sector
  offset** (Theissen: sectors have flux offsets; Hattori+2025 handled this with offset terms).
- Score: fraction within 10% of truth, also count half/double-period aliases separately. Report per
  period bin (< 13.7, 13.7–27, > 27 d).
- **Output:** `results/step2_ls_pdcsap.csv`, a 3-row summary table, and ~6 example periodogram PNGs
  (mix of hits and misses; mark the truth period and 13.7 d).
- Start with ~10 stars; measure download size + runtime per star-sector and write it down — Theissen
  asked for exactly this to size a supercomputer request.

### Step 3 — SAP vs PDCSAP (the decision step)
- Same stars, same LS code, `SAP_FLUX` instead of `PDCSAP_FLUX`. SAP keeps scattered light and
  other systematics, so expect more junk peaks near 13.7 d — count them.
- Compare per star: is the true long period recovered in SAP but not PDCSAP? In both? In neither?
- **Output:** `results/step3_sap_vs_pdcsap.csv` + the 3-outcome count table matching ROADMAP's
  decision table.
- The claim "PDCSAP removes sector-scale signal" appears in the literature but **no primary source
  was located**; this step measures it. Do not cite it as established fact in anything for Theissen.
- **Stop here and report to the user** (this is the end of Phase 1). Draft, don't send, the Slack
  update: plain language, what was done, the three-bin table, which Step 3 outcome occurred, what
  that implies, one or two domain questions for him. Use the `theissen-report` skill's gate.

### Step 4 — reused SWM forecaster (only after the user says go)
- Data: SPOC PDCSAP binned to 30 min, full sectors, split into the two ~13.7 d orbits.
- Train to predict orbit 2 from orbit 1 (first-half → second-half, as Theissen suggested).
- Required comparison rows (ROADMAP "honesty" section): (a) LS without extension, (b) extension by a
  fitted sinusoid, (c) persistence/copy-last forecast. The model counts only if it beats (b).
- Held-out **stars** for test, never held-out windows of training stars.
- **Training is never run by you.** Smoke-test the exact command tiny (`max_epochs=2`, wandb
  disabled, a handful of stars), then print the full command (env activation, `PYTHONPATH`,
  `PYTHONUNBUFFERED=1`, all overrides) for the user to run in their own terminal, and wait for them
  to say it finished.

### Step 5 — small GAN (only after Step 4 works)
- Same data, splits and comparison rows as Step 4. Keep it small (1D conv generator conditioned on
  the observed half; a discriminator on real vs generated continuations). Same training handoff rule.

### Step 6 — report
- One table: method × period bin, plus the Step 3 outcome. Slack draft via `theissen-report`.
- Only propose SPOC Target Pixel Files (the CV route back) if Step 3 came out "signal in SAP, gone
  in PDCSAP". Otherwise say plainly that SPOC light curves look sufficient.

---

## 5. Rules that bite on this project

- **Instrument claims need a source or our own measurement.** Theissen corrected an invented TESS
  trade-off in the user's paper (`docs/reference/2026-09-19-theissen-slack-review.md`). TESS orbit is
  13.7 d; pixels 21"/px vs Kepler 4"/px; these are sourced. Anything about what SPOC/PDC does to
  long signals is to be measured, not asserted.
- **Extension creates no new observation.** Never describe a model-extended light curve as data.
- **Any loop > ~1 min** gets `tqdm` with `desc=`/`total=`; heavy non-training commands run
  backgrounded with `PYTHONUNBUFFERED=1`; never pipe through `head`/`tail` or `2>/dev/null` (a hook
  blocks it).
- Python: type hints; `uv`, not `pip`, for any install.
- Network batch scripts: tenacity retry, `status="error"` rows on failure, verify outputs on `--resume`.
- Commits: conventional commits, **no Co-Authored-By line** (user's rule overrides any harness
  attribution reminder). The first commit of this repo has not been made; ask the user before the
  first commit/push. Raw transcript is gitignored on purpose.
- Communicate with the user tersely, in Chinese if they write Chinese. Reports for Theissen are in
  English, plain language, no ML jargon.

---

## 6. Open items you can't resolve alone

- TESS sector coverage of MEarth stars in SPOC 2-min (Step 1 answers it).
- Whether TARS used SPOC or FFI light curves: the 09-10 notes say FFI; not checked against the TARS
  methods section (https://arxiv.org/abs/2603.05586). Only matters if it goes in a report.
- Supercomputer request: Theissen wants storage + compute estimates from the ~10-star run.

---

## 7. Suggested skills

| when | skill |
|---|---|
| any MAST / VizieR / astroquery batch | `astroquery-resilience` (lives in SWM `.claude/skills/`; read it by path if not listed here) |
| writing analysis code | `human-style-code` |
| any statement about how MEarth, TARS or SPOC/PDC produced data | `read-the-methods-section` |
| numbers or tables going to Theissen | `reporting-measured-numbers`, then `theissen-report` |
| plots | `dataviz` |
| if a notebook is used | `notebook-direct-edit` + `notebook-clarity` |
| presenting choices to the user | `option-set-hygiene` |
| unexpected failures | `diagnosing-bugs` |

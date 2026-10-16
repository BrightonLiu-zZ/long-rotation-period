# Next-direction meeting prep — Prof. Theissen (20 min)

**Written:** 2026-09-10 · **Purpose:** choose the astro question + ML method for the post-ML4PS project, and the one paper to share.
**Decisions already taken by Brighton (this session):** new repo, new corpus, v1 hard constraints do not carry over; world model is the method *under test*, other methods are baselines; posture in the meeting = "work backward from the astro problem, I am flexible on the model"; pick the direction where SSL / world model is most likely to work.

---

## 0. Recommendation in three lines

- **Framing:** *Can a model trained only on TESS photometry measure rotation periods longer than the light curve it sees — P > 13 d from one sector, P > 27 d from two or three — for K/M dwarfs, validated against independent long-baseline truth (Kepler, ZTF)?*
- **Method:** self-supervised latent-dynamics forecaster pretrained on millions of unlabeled TESS FFI light curves; period read from the observed + rolled-out sequence. Label-free evaluation = held-out future-sector prediction, which removes the TARS circularity. Baselines: amplitude-only, Astraea-style RF (features + Gaia), Claytor-style CNN, Lomb–Scargle.
- **Paper to share:** Hattori, Angus, Foreman-Mackey, Lu & Colman 2025, *Measuring Long Stellar Rotation Periods (>10 days) from TESS FFI Light Curves is Possible*, arXiv:2505.10376.

Why rotation and not planets: it is his stated preference, it is un-saturated (two published attempts at true beyond-baseline recovery are both weak, §2), it fits Cool Star Lab (M-dwarf ages, habitability), and it is the direction where unlabeled data and a forecasting objective give SSL a real edge (§4).

---

## 1. Restating his question precisely (say this first, 2 min)

He said: *"measure rotation periods below what is already done, potentially longer than the time baseline of observations,"* citing TARS (Boyle, Bouma & Mann 2026, ApJS 284, 75) as the bar.

Three different "baselines" hide in that sentence. Name them so the goal is unambiguous:

| baseline | length | what limits recovery there | who has done it |
|---|---|---|---|
| TESS orbit | 13.7 d | scattered-light systematics; PDCSAP detrending removes sector-scale signal | TARS to ~12 d single-sector, 25 d with multi-sector consistency |
| one sector | 27.4 d | window function; ≤ 1 cycle observed | Claytor+2024 (CVZ stacks, 35–80 d), Hattori+2025 (12–13 sectors, ~50 d) |
| **1–2 sectors, P > baseline** | 27–55 d | genuinely extrapolative: < 1 cycle, spot evolution | **Lu+2020 Astraea 55% on real TESS; Claytor+2022 Kepler-field test "unable to recover reliably"** |

The third row is his question as asked and it is open. Everything above it is "beat the 13.7-d systematics with a long stack", which the CVZ papers have already done.

**The circularity we found (T0/T2, Aug 2026):** TARS was our label source, so a model-positive/TARS-null star scored as a false positive. Our own T2 measurement on the v1 corpus showed the encoder's period estimate saturates at the 5.7-d input length and that amplitude alone outranks the latent past that point. New project fixes both: independent truth (Kepler/ZTF) and a label-free objective (forecasting).

---

## 2. State of the art (one slide, 5 min)

| work | data | method | reach | independent truth? |
|---|---|---|---|---|
| TARS, Boyle+2026 | FFI S1–96, 1.05 M stars, T<16, <500 pc | LS + 2 random forests (systematics, half-period alias) | reliable to ~25 d; completeness falls past ~10 d | no (self-consistency) |
| Claytor+2022 (ApJ 927, 219) | TESS 2-min, simulated spots ⊗ real systematics | CNN on wavelet, heteroskedastic | 46% within 10% on best half (sim); **Kepler field, 2 sectors: not recovered** | Kepler McQuillan, failed |
| Claytor+2024 (ApJ 962, 47) | SCVZ, ~1 yr stacks | CNN on wavelet | 7,245 periods; 35 d GK, 80 d M; recovers 10–20 d gap | partial |
| Hattori+2025 (arXiv:2505.10376) | NCVZ K/M dwarfs, G 13–16, 12–13 sectors, unpopular FFI | LS with per-sector offset columns | 66% within 10% (81% with power cut) up to ~50 d; 16% half-period aliases | **ZTF DR22, 272 stars** |
| Lu+2020 Astraea (AJ 160, 168) | Kepler cut into 27-d segments + Gaia DR2 | random forest | 9% on Kepler segments to 150 d; **55% on real TESS** | Kepler, 195 stars |
| Boyle, Mann & Bush 2025 K2–TESS limits (ApJ, 2504.13262) | 23k K2–TESS overlap | LS | <3% error below 10 d; >80% reliable to 10 d; stitching improves precision not reliability | K2 |
| Lares-Martiz+2025 (2507.08266) | wide binaries, gyro constraint | LS + gyro consistency | reliable < 8 d, hard > 10 d | co-eval pairs |

Reading: the 13.7-d barrier is beaten with year-long stacks. **Nobody has recovered P > baseline from one or two sectors using the light curve alone**; the one method that tries (Astraea) leans on Gaia stellar parameters, i.e. a gyrochronology prior, and lands at 55% on TESS.

Why it matters astrophysically (his language): most field M dwarfs rotate slower than 27 d, so the sample TARS *cannot* see is the sample that carries gyrochronology ages for planet hosts; the intermediate period gap sits at 10–20 d for cool stars, straddling the barrier; Claytor+2024 states the 13-d limit "puts the majority of stars beyond reach".

---

## 3. Proposed framing + method (6 min)

**Question.** For K/M dwarfs with 1–3 TESS sectors, how much rotation-period information does the photometry carry past the observing baseline, and can a learned dynamics model extract it?

**Design.**

1. **Corpus:** TESS FFI light curves (QLP or unpopular; SPOC 2-min covers only ~30% of M dwarfs, QLP >85%). Own detrending, no PDCSAP. Millions of unlabeled curves.
2. **Pretraining (self-supervised):** latent-dynamics model trained to forecast the next window / next sector from the past. This is the world-model recipe from the summer project, re-targeted: the loss is prediction, not reconstruction, so the latent must track phase and period, not amplitude.
3. **Readout:** period from the observed + imagined sequence (periodogram on the rollout) and, as a second arm, a linear probe on the latent. Optional Gaia-colour conditioning = Astraea's prior, kept as a separate arm so its contribution is measurable.
4. **Truth sets (independent of TESS):**
   - Kepler field stars with McQuillan+2014 / Santos+2019 periods (to ~70 d), observed by TESS in S14/15/26/40/41/53/54/55 → the 1–3-sector regime with 4-yr truth.
   - NCVZ K/M dwarfs with ZTF periods (Hattori's 272; Lu+2022's 40,553 ZTF periods, 9,535 > 10 d) → the multi-sector regime.
5. **Baselines:** amplitude-only; Lomb–Scargle; Astraea-style RF; Claytor-style CNN on wavelet; untrained encoder.
6. **Label-free gate:** held-out future-sector prediction error vs persistence and vs a Lomb–Scargle extrapolation. If the model cannot forecast the next sector better than LS, it has not learned the dynamics and no period claim is made.
7. **Mechanism test first (inherited Q18):** does the period ceiling move with the input length? Train at 13, 27, 54 d inputs; if the ceiling tracks the input, the information is there and the estimator is the bottleneck; if not, the ceiling is physical.

**Two publishable outcomes, pre-registered.**
- Positive: periods recovered past the baseline at stated accuracy on Kepler/ZTF truth.
- Negative: "information limit of N-day TESS photometry" measured with the strongest available estimator — the same shape as Boyle, Mann & Bush 2025's limits paper, but with a learned estimator instead of LS. Either way it answers "what is the length limit" for TESS.

**Flexibility line for the meeting:** "The forecaster is my candidate because it uses the unlabeled data; if the CNN or the periodogram beats it on the Kepler truth set, the paper reports that. The astro question fixes the evaluation, not the model."

---

## 4. Why this is where SSL / world model is most likely to work (Brighton's criterion)

- **Unlabeled data is the asset.** Millions of FFI curves, zero labels needed for forecasting. Claytor trained on simulations because real labels are scarce; SSL on real curves is the untried axis.
- **The objective points away from the confound that beat us.** T2 showed amplitude outranking the latent past the baseline under a reconstruction-style objective. A forecasting loss is phase-sensitive by construction.
- **Label-free evaluation exists.** Next-sector prediction is a score no catalog can be circular with.
- **Honest risks (say them before he does):** Claytor's 2-sector Kepler test failed; Astraea got 55%; spot lifetimes bound the forecast horizon (weeks for K, longer for M dwarfs); T2 found the beyond-baseline signal was not SSL-specific on the v1 corpus. The mechanism test in §3.7 is designed to fail fast if the same thing happens here.

---

## 5. Alternative kept on the table (1 min, only if he asks)

**Planets around active / rapidly rotating M dwarfs**, world model as the stellar-variability noise model, transit = forecast residual; baseline = `nuance` (Garcia+2024, GP + linear search on 438 rapidly rotating M dwarfs). Un-saturated subtype and Cool Star Lab-adjacent (SPECULOOS). Not recommended now: he called planets saturated, our v1 transit signal was the weakest task, and a GP already does the noise model well. Revisit if rotation fails the §3.6 gate.

---

## 6. Data flexibility note (LSST)

Rubin survey operations started 2026-06-29; DP2 (commissioning data, Apr 2025–Jan 2026) ships Oct–Dec 2026; DR1 is one year of survey data, due ~June 2028. Too immature for a Oct 2026–Mar 2027 project. The sparse-cadence version of the same question is LSST's future problem, and ZTF is its rehearsal, so the ZTF truth set in §3.4 doubles as the forward-looking hook. Keep TESS as the primary dataset, as he said.

---

## 7. The one paper to share, and why

**Hattori, Angus, Foreman-Mackey, Lu & Colman 2025, arXiv:2505.10376.**

- It is the newest rung above TARS on exactly his axis: periods > 10 d from FFI, reaching ~50 d, > 27 d shown possible.
- It supplies the evaluation protocol we adopt: ZTF as independent truth, 10% agreement, power-cut reliability, explicit half-period-alias accounting.
- It quantifies the current limit (66% / 81%, 12–13 sectors) and does **not** attempt few-sector recovery, which is our gap.
- Same group as Astraea (Lu is co-author), so it is the natural citation chain: Astraea (2020) → Hattori (2025) → this project.

Names to say aloud as backups: Claytor+2024 (CNN, 80 d in the SCVZ) and Lu+2020 Astraea (the only true beyond-baseline attempt).

---

## 8. 20-minute script

| min | beat |
|---|---|
| 0–3 | Restate his question as the three-baseline table (§1). Say plainly that our summer evaluation could not answer it because TARS was the label source; new project uses Kepler/ZTF truth. |
| 3–8 | SOTA table (§2). Land the sentence: "nobody has recovered periods past the baseline from one or two sectors with the light curve alone." |
| 8–14 | Framing + design (§3). Emphasise the label-free gate and the two-outcome plan. Deliver the flexibility line. |
| 14–18 | Asks (§9). |
| 18–20 | Next steps: new repo, corpus choice, first milestone = mechanism test on the Kepler-field truth set. Share the Hattori paper. |

---

## 9. Asks for him (domain expertise only)

1. **Which period regime matters to him:** the 10–20 d gap region (gyro physics) or the 30–100 d slow M dwarfs (ages for planet hosts)? Determines the truth set and the input-length sweep.
2. **Independent truth from Cool Star Lab:** does the lab hold v sin i / radii (v sin i + R → period upper bound) or ages for any TESS-observed M dwarfs? Any contact to Bouma (TARS) or the Angus/Lu group?
3. **FFI pipeline sanity for M dwarfs:** QLP vs unpopular vs TESS-SPOC FFI for faint red stars; contamination cuts he trusts.
4. **Is the negative outcome publishable in his view** (an "information limit of TESS photometry" paper), and which venue he would target (AJ/ApJ letter, RNAAS, AAS 249 poster).

---

## 10. Career fit (for Brighton, not for the meeting)

Throughline "latent dynamics from board games to physical stars" continues unchanged; the new phrase is *world model as imagination for measurement*: dream the unobserved part of the cycle, then measure it. Fits the scientific-time-series latent-variable faculty on the list (Mandt, Kuleshov, Grover, Rose Yu) and the state-estimation framing for ECE (Atanasov). Code reuse from the summer repo: windowing, the encoder recipe, the eval harness and the seed/gate discipline; nothing from the v1 constraints.

---

## Sources

- TARS: https://arxiv.org/abs/2603.05586 · https://iopscience.iop.org/article/10.3847/1538-4365/ae6657
- Hattori+2025: https://arxiv.org/abs/2505.10376
- Claytor+2022: https://arxiv.org/abs/2104.14566 · Claytor+2024: https://arxiv.org/abs/2307.05664
- Lu+2020 Astraea: https://arxiv.org/abs/2008.03351 · Lu+2022 ZTF periods: https://arxiv.org/abs/2210.06604
- Boyle, Mann & Bush 2025 K2–TESS limits: https://arxiv.org/abs/2504.13262
- Lares-Martiz+2025: https://arxiv.org/abs/2507.08266
- nuance (Garcia+2024): https://arxiv.org/abs/2402.06835
- ExoMiner++: https://arxiv.org/abs/2502.09790 · RAVEN: https://arxiv.org/abs/2603.22597 · ExoNet: https://arxiv.org/abs/2604.15560
- FALCO: https://arxiv.org/abs/2504.20290 · Scaling law in light curves: https://arxiv.org/abs/2405.17156 · LightPred: https://arxiv.org/abs/2407.06858 · SELDON: https://arxiv.org/abs/2603.04392
- Rubin DP2/DR1 schedule: https://dp2.lsst.io/index.html · https://community.lsst.org/t/early-data-preview-2-edp2-release-date-2026-07-27/12226
- Theissen / Cool Star Lab: https://ctheissen.physics.ucsd.edu/ · https://www.coolstarlab.org/

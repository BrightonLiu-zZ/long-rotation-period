# long-rotation-period

Can rotation periods longer than the TESS observing window be measured from TESS light curves?
Informal project with Prof. Chris Theissen (UCSD), started 2026-10. Split out of
`C:\git_repo\Stellar-World-Model` on 2026-10-03; the two projects share no code constraints.

## Current scope (set in the 2026-10-02 meeting)

- **Data:** SPOC light curves. Images are out of scope until the light-curve route is shown to be insufficient.
- **Method:** a model predicts the continuation of a light curve; the extended curve goes into a periodogram.
- **Baseline:** Lomb–Scargle periodogram.
- **Truth:** TARS for short periods; MEarth rotation periods for long ones.
- **Validation:** inside TESS only (held-out stars; first half of a sector predicts the second half).

## Where things are

| what | where |
|---|---|
| Plan | `docs/ROADMAP.md` |
| Meeting summary + transcript | `docs/meetings/` |
| Pre-meeting research (pixel-video idea, related work, links) | `docs/prep/` |
| Notes copied from the SWM repo | `docs/reference/` |

# CLAUDE.md

Guidance for this repo. Keep it lean.

## Project

Rotation periods beyond the TESS observing window, with Prof. Theissen (informal, no weekly meeting).
Read `docs/ROADMAP.md` and `docs/meetings/2026-10-02-theissen-summary.md` before starting work.

## Constraints

- **Minimal time.** The user wants this done roughly well with as little time as possible. Do the
  smallest thing that answers the step in the roadmap; no extra sweeps, gates or write-ups.
- **SPOC light curves only** until Roadmap Step 3 says otherwise. Do not start image / FFI work on
  your own initiative.
- **TARS is never the truth set for long periods** (it is the bar Theissen named, so scoring against
  it counts new detections as errors). Long-period truth = MEarth.
- The Stellar-World-Model hard constraints (PDCSAP only, no stitching, frozen architecture,
  window/seq sizes) do **not** apply here. Code may be copied from that repo; do not import across repos.
- Anything said to Theissen about how an instrument or pipeline behaves needs a source or a
  measurement of our own. He checks these.

## Environment

- Data scripts: conda env `astro`. Model training: conda env `swm` (CUDA, RTX 4060).
- Network batch scripts (astroquery / lightkurve / Vizier) need retry, error status and resume.

## The user's wider goal

Computer-vision / video world-model experience. The honest route back to pixels is Roadmap Step 3
(SAP vs PDCSAP), then SPOC Target Pixel Files. Do not oversell it: if SPOC light curves turn out
to be sufficient, say so.

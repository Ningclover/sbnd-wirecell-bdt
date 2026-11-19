# release/: freezing, scoring and validating a trained νe model

- `freeze_release.py`: turns a trained level-1 folder and a level-2 model into a self-contained release under
  `releases/<name>/` (models cut to the trees actually used, identity-checked, plus `manifest.json`).
- `score_release.py`: scores the builder's tables with a release, from its manifest alone.
- `compare_chain.py`: compares the scores a Wire-Cell chain run stores with the standalone release on the same events.

Releases: round 4 (2026-11-19), three models: `releases/README.md`; `releases/nue_round4_ruleA/` (A and A-nogap), `releases/nue_round4_ruleB/` (B).

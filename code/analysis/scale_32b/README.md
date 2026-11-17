# 32B helper and receiver pair on OBQA: analysis code

Research ID E2b; the paper's appendix on the 32B helper and receiver pair. CPU only; see `code/pipeline/scale_32b/README.md` for the layout as run.

- `e2b_analyze.py`: pilot review, full-fit thresholds, calibration tests with the sequential label queries and the
  frozen deployments, development and held-out evaluation (Clopper-Pearson bounds, paired bootstrap), the timing panel,
  the report and the recheck exports (`e2b_recheck/`). Its Chinese report text and conclusion labels are translated.
- `export_e2_recheck.py`: compatibility exports of the recheck files (also used for the batched-serving bundle).

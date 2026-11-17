# CPU re-analyses (gain retention, per-discipline certification, single-model baselines, lost-correction test)

`recompute.py` re-analyses stored records only (no model forward pass): per-discipline MMLU-Pro certification,
gain retention on unseen questions, single-model baselines and the medium/OBQA/Text lost-correction test. As run, it lived in `rebuttal/r0_cpu/` of a tree with this repository's `code/`
and the dataset's `records/`, `results/` and `splits/`; set `P2_RELEASE_ROOT` to such a tree and `DATA_DIR` to the
directory holding the OpenBookQA parquet (`c2c_reproduction_assets/datasets/allenai--openbookqa/main/`). It loads the
frozen calibration module `code/analysis/mmlu_pro_breadth/src/risk.py` (added with this release, docstring translated)
and the answer parser `code/pipeline/answer_scoring_v2/scoring_v2.py`. It writes its four CSV files next to itself.

`export_medium_ledger.py` was written at release time: it re-runs the medium/OBQA/Text steps of `recompute.py` with the same
frozen `risk.py` and parser, writes the per-candidate ledger (j = 1..20) and the 36-label query log released in the dataset
(`results/cpu_reanalysis/results/medium_text_calibration_ledger.csv`, `medium_text_label_queries.jsonl`), and asserts that every
summary value equals the saved `medium_text_lost_correction.csv`.

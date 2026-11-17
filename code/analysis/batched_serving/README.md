# Batched serving: analysis code

Research ID E2a v2; the paper's appendix on batched serving. `analyze_e2a_v2.py` computes the 36 arm cells, the 36 paired
comparisons (20,000 batch-paired bootstrap replicates), the answer, message, score and route consistency tables and the
sensitivity analysis. Its Chinese report text and conclusion labels are translated: "positive benefit established",
"positive benefit not established", "negative benefit established", "negative benefit", "non-negative point estimate".
The recheck bundle was written by `code/analysis/scale_32b/export_e2_recheck.py e2a_v2`.

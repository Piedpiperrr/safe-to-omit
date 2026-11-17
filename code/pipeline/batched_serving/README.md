# Batched serving: pipeline code

Research ID E2a v2; the paper's appendix on batched serving. The analysis code is in `code/analysis/batched_serving/`.

As run, these files were in the same directory as the 32B code (see `code/pipeline/scale_32b/README.md`) and import
`e2b_gate.py` and `e2_debug_ops.py` from there. `prepare_e2a_v2.py` starts from the configuration of the earlier E2a
preparation (frozen thresholds and model pairs of the paper's deployments; the earlier E2a variant is not part of this
release) and fixes the full development sets (OBQA 742, ARC 299).

- `e2a_v2_batch.py`: the three arms (synchronous, early return, reference) for one batch, and the arm rotation.
- `run_e2a_v2.py`: the replay over model pairs, datasets and batch sizes 1, 8 and 32 on one exclusive node.
- `e2a_v2_gate.py`: freeze check and unit start (the per-unit claim and the cluster-node assertion are removed).

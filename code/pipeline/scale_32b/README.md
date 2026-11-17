# 32B helper and receiver pair on OBQA: pipeline code

Research ID E2b; the paper's appendix on the 32B helper and receiver pair. The analysis code is in `code/analysis/scale_32b/`.

These files are the code as run, with comments, names, paths and scheduler-only parts cleaned for release (no
computation, prompt, constant or threshold changed). As run, all E2 files (this folder, `code/analysis/scale_32b/`,
`code/pipeline/batched_serving/`, `code/analysis/batched_serving/`) were one directory `rebuttal/e2/` inside a copy of
this repository, with the frozen configurations (`e2b_config.json`, `e2b_formal_config.json`, plans), the prepared
inputs (`inputs/`), the freeze receipts and the run directories next to them. The modules `e2b_gate.py` and
`e2_debug_ops.py` are shared with the batched-serving code.

- `prepare_e2b.py`, `prepare_e2b_formal.py`, `prepare_e2b_formal_inputs.py`: inputs and frozen configuration (the
  last one restores two option texts that an earlier path anonymization had altered; see the dataset's DEVIATIONS.md).
- `hf32_runtime.py`, `hf32_formal_runtime.py`: the four-GPU layer-sharded runtime: helper message, receiver answer,
  Text answer, helper-alone answer, the log-space probe score, and the timed policy / reference / receiver-alone arms.
- `run_e2b_formal.py`, `e2b_runtime_checks.py`: the per-shard worker, its two-question consistency preflight and the
  non-statistical warmups of the timing segments.
- `e2b_gate.py`, `e2b_formal_gate.py`, `e2_debug_ops.py`, `e2b_plan45.py`: freeze and stage-release checks, the shard
  plan overlay and completion checks. Removed for release: job submission, queue and job-record checks, the per-unit
  claim helper on the shared filesystem, the resource re-estimate, and the cluster-node assertions.

The freeze checks compare file hashes with the research freeze receipts, which are not released; the edited files
therefore no longer match those receipts.

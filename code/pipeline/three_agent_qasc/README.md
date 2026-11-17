# Three-agent system on QASC: pipeline code

Research ID E1; the paper's appendix on the three-agent system on QASC. The analysis scripts are in `code/analysis/three_agent_qasc/`.

These files are the code as run, with comments, names, paths and scheduler-only branches cleaned for release (no
computation, prompt, constant or threshold changed). As run, the files of this folder and of the analysis folder were
one directory `<E1>/code/`, next to `<E1>/configs/`, `<E1>/prompts/`, `<E1>/splits/`, `<E1>/data/qasc/` (written by
`data_prep.py` from the QASC parquet files), `<E1>/outputs/`, `POLICY_FREEZE.json` and `FROZEN_B_PROMPT.json`. `<E1>`
was `rebuttal/e1/` inside a copy of this repository, whose `code/pipeline/` holds the four frozen paper files that
`e1lib.py` checks by SHA-256 on import. To run the scripts, recreate that layout from this repository and the dataset
folders `configs/three_agent_qasc/` and `splits/three_agent_qasc/`.

Order of use: `data_prep.py` -> `make_splits.py` -> `build_config.py` -> `cpu_preflight.py` -> `e1_generate.py` (per
split and shard; A, B, C, the probe and the controls) -> calibration and evaluation scripts (analysis folder) ->
`layout_check.py` and `replay_e2e.py` (timing panel). `clusterA_check.py` is the cross-cluster consistency gate that ran
before shards were generated on the second cluster.

`e1lib.py` is unchanged from the run (SHA-256 6417a0a728dbf05317f00b355ccde9657c1e974c686ae226cbec8adb44349c04, the value
recorded in every generation status file). In `e1_generate.py` the cross-host shard lock (scheduling only) is removed.

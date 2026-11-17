# Three-agent system on QASC: analysis code

Research ID E1; the paper's appendix on the three-agent system on QASC. CPU only. As run, these scripts sat in the same directory as the
pipeline files (`code/pipeline/three_agent_qasc/`, see its README) and import `e1lib.py` from there.

- `collect_and_thresholds.py`, `fit_diagnostics.py`, `fit_accuracy_and_p1p2_prediction.py`, `fit_controls_accuracy.py`:
  fit-split thresholds, score diagnostics, the label-free P0 prediction and the P1/P2 predictions.
- `calibrate.py`: the change-rate and lost-correction tests on calibration, the label query rule, and the policy freeze.
- `dev_report.py`: development accuracies, controls and ablations.
- `sealed_gate.py`, `sealed_eval.py`: the version gate and the one-time sealed-test evaluation (R1, R2, paired bootstrap).
- `timing_report.py`: the 128-question timing panel (paired savings, bootstrap intervals).
- `leak_stats.py`, `unused_audit.py`, `deviations_timeline_evidence.py`: evidence behind the deviations table.
- `export_recheck.py`: the recheck CSV export. `smoke_report.py`, `tune_report.py`: pre-freeze smoke and prompt-tuning
  reports.

Scheduler job ids that the run used as directory names or in output strings are replaced by neutral labels (`job-NN`).

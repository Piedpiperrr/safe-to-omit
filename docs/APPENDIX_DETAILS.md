# Details not printed in the paper

The paper's appendix is a shortened version of an earlier, fuller record of every experiment. The
details below were left out of the appendix for length. Each bullet gives the detail, with its
numbers exactly as in that record, and the file that holds it (paths are relative to the dataset
root; see [RECORDS.md](RECORDS.md)). The earlier text
used "omit" for what the paper calls "skip"; quotations keep the original word.

## Appendix A — Method and protocol

- **Role of the learned predictor.** The learned predictor (internal name `D`) is a control, not a
  candidate for the main method. Record: `configs/confidence_boundaries/frozen_config.json` (key `D`).

## Appendix C — All settings and robustness of the boundary

- **Medium-pair development diagnostics** (OBQA/Text, OBQA/C2C, ARC/Text, ARC/C2C): average
  precision 0.494, 0.502, 0.412, 0.611. Changed answers that became correct / became incorrect /
  stayed incorrect: 0/0/0, 7/1/3, 0/0/0, 2/1/1. INVALID parses of R/reference/policy: 8/1/1,
  8/16/14, 2/0/0, 2/12/10. Record:
  `results/boundaries_medium/execution_retry1_20260915T164957Z/summary/compact_summary.csv`.
- **Additional diagnostics.** For the small and large pairs, the released records also provide
  panel-level comparisons with an undeployable post-hoc agreement reference, valid-only rankings
  and INVALID decompositions, and paired uncertainty for accuracy and cost differences.
  Record: `results/boundaries_large_small/summary/` (`panel_cost.csv`, `ranking.csv`,
  `invalid_decomposition.csv`, `paired_bootstrap.csv`).
- **Clopper–Pearson 0.999 bound** for the calibration statistics of Table 5, in its row order: 0.232,
  0.161, 0.419, 0.296; 0.096, 0.049, 0.102, 0.031; 0.033, 0.038, 0.035, 0.048, 0.041, 0.045. Record:
  `results/r1_cpu_prep/results/item2_calibration_table.csv`.
- **MMLU-Pro.** "Both references accept q=0.40 (threshold u≤3.8385×10⁻⁵) … Text changes 67 answers
  (p=1.34×10⁻⁷, CP 0.0406) and C2C 75 (p=1.01×10⁻⁵, CP 0.0446)". Record:
  `results/mmlu_pro_breadth/calibration/40_test_ledger.csv`.
- **Small- and medium-pair MMLU-Pro runs.** "the 96,256 requests then ran with no duplicate or
  missing key and no input mismatch." Record: `results/e08/SUMMARY.md`.
- **Same-target baselines.** The per-component latency table (Figure 4b) is in
  `code/figures/appendix/record_tables/baselines.tex`.
- **Historical routers.** "Earlier complete end-to-end routers and auxiliary scores retain their
  original sources and are not merged with the unified panels." Record:
  `results/boundaries_large_small/summary/historical_e2e_reference_reused.csv`.

## Appendix D — Predicting the boundary

- **Small-sample convention.** Samples use "development questions for Text+fact, which has no
  fit-split reference outputs"; "a sample with no disagreement predicts deployment, and one with no
  agreement predicts fallback." Record: `results/e13/PREREG_E13.md`.
- **Fit-split saving.** "unlike [the paper's saving equation], the gap is not conditioned on
  omission." Record: `results/e15/PREREG_E15.md`.
- **Helper-aware score, Text+fact.** For Text+fact the helper prefill omits the fact, as frozen; a second,
  descriptive row with the fact in the prefill is in the released records. Record:
  `results/e09b_e09c/analysis/E9C_RESULTS.json`.

## Appendix E — Accuracy under skipping

- **"Other" and "Unused (omit/keep)".** "Other counts omitted questions where R and the reference
  are wrong but the other communication action is correct, and unused counts questions where some
  action is correct but the policy is not, split by side." Values:
  `code/figures/appendix/record_tables/omit_keep.tex` (plotted in part in Figure 6b).

## Appendix F — Headroom and the null control

- **All eight correctness patterns** of R/Text/C2C (1 = correct; INVALID counts as incorrect;
  development):

  | Population | 000 | 100 | 010 | 001 | 110 | 101 | 011 | 111 |
  |---|---|---|---|---|---|---|---|---|
  | small/OBQA | 244 | 29 | 66 | 77 | 37 | 46 | 70 | 173 |
  | small/ARC | 95 | 12 | 25 | 39 | 12 | 27 | 30 | 59 |
  | medium/OBQA | 149 | 11 | 59 | 21 | 37 | 38 | 23 | 404 |
  | medium/ARC | 51 | 10 | 21 | 6 | 15 | 11 | 4 | 181 |
  | large/OBQA | 72 | 5 | 37 | 7 | 30 | 13 | 9 | 569 |
  | large/ARC | 19 | 1 | 8 | 2 | 5 | 3 | 2 | 259 |
  | large/MMLU-Pro | 1083 | 54 | 121 | 110 | 150 | 75 | 45 | 1003 |

  Questions with C2C correct and Text wrong: 123, 66, 59, 17, 20, 5, and 185 in the order of the
  table. Record: `code/figures/appendix/record_tables/patterns.tex` (plotted in Figure 8a).
- **Correct answers, R / rotated options / another question's message** (same order): 285/290/316,
  110/123/130, 490/482/507, 217/216/220, 617/599/624, 268/260/271, 1282/1286/1306. Record:
  `results/e01_e02_e04/results/e4_null_controls.csv`.
- **Unmatched null oracle** over R and the two perturbed inputs: 488, 200, 596, 245, 679, 277, 1599
  correct; null headroom 23.18, 23.41, 11.99, 8.36, 7.41, 2.01, 11.09 points. Record: same file.
- **Forced** (mean number of shared questions per draw on which the two joint nulls must agree):
  84.3, 37.0, 20.9, 8.3, 3.7, 1.0, 133.3 (Appendix F prints the range). Record:
  `results/e09a/results/e9a_ratios.csv`.
- **Joint-null headroom, mean [2.5, 97.5 percentiles over draws]:** 138.1 [128, 148], 56.3 [49, 64],
  90.9 [83, 97], 25.6 [22, 28], 40.1 [34, 46], 7.6 [6, 9], 260.8 [244, 273]. Record:
  `results/e13/results/item1_best_fixed_headroom.csv`.

## Appendix G — Extensions

### G.1 Cross-family pairs

- **Llama pair.** "Receiver-only outputs and ProbeMax scores are those of the small pair (all 5,626
  inputs matched)." Record: `results/x01_x02_cross_family/x1/notes/prepare_checks.json`.
- **Run date.** "The official aligner writes the run date into the helper's prompt header, so the
  C2C helper prompts carry the date of the run …, unlike the fixed date of the Text messages; exact
  reproduction requires that date." Record: `results/x01_x02_cross_family/DEVIATIONS.md` (items C1,
  C16).
- **Llama-8B.** "The rendered template already begins with a BOS token, so inputs are tokenized
  without adding another; with Qwen3-8B as receiver, this code reproduced 16 saved fit rows bit for
  bit, and a 16-question smoke test with the new receiver had no unparseable answer." Record:
  `results/x03_llama_receiver/DEVIATIONS.md`, `results/x03_llama_receiver/results/smoke/SMOKE_CHECK.json`.

### G.2 Stronger helper

- **Smallest p-values.** "0.26 (OBQA, q=0.05, 26 changes among 601 omitted calibration questions)
  and 0.026 (ARC, q=0.05, 6 of 256)". Record: `results/e16/results/analysis_e16_5/ledger_40.csv`.
- **Unparseable answers.** "development: 8 and 2 for the receiver alone, none with the messages".
  Record: `results/e16/results/analysis_e16_5/E16_5_RESULTS.json`.

### G.3 SQuAD

- **Protocol.** The full-run protocol added "Text outputs for the fit questions, used only for
  re-splits"; "an empty answer is INVALID, and two INVALID answers agree." Record:
  `results/e20_full/PREREG_E20F.md`.
- **Pilot.** "No ineligible setting has P above 0.192 at 3,000, so eligibility did not change the
  choice." "Text changes the receiver's exact-match accuracy by −3.3 to +4.3 points across the eight
  settings, and by +0.8 on the selected one (65.5 to 66.3%)." Record:
  `results/e20_pilot/PILOT_STATS.csv`, `results/e20_pilot/results/ACCURACY_E20P.json`.
- **Surface form.** "the median token F1 between the two answers among changes is 0.517, and
  extraction did more than normalization in 4 of 792 outputs. Over all development questions, raw
  strings differ on 23.5%, normalized answers on 15.4%, and 6.0% remain with containment counted as
  agreement. Median answers have two words on both paths; 6 receiver-only and 2 Text answers reach
  the 32-token limit; helper messages have a median of 36 tokens". Record:
  `results/e20_full/RESULTS_E20F.md`.
- **Gain and all changes.** "Text adds 1.0 point [−0.9, 2.9] on development and 1.8 [−0.003, 3.8] on
  test"; "Among the 154 development changes, Text corrects 49 answers and breaks 39 … On test, the
  figures are 54 and 36 among 155 changes". Record: same file.

### G.4 GSM8K

- **Extractor.** "commas, currency and percent signs and a trailing period are stripped, and the
  result is compared as a decimal with trailing zeros removed"; "an answer that does not parse would
  take u=1 and never be omitted." Record: `results/e10/PREREG_E10.md`.
- **Whole-continuation score.** "well spread (0.0163 to 0.1952)". Record: `results/e10/SUMMARY.md`.

### G.5 Sampled answers

- **Changes by uncertainty.** At u=0, 2 of 354 (OBQA) and 0 of 217 (ARC) skipped answers change,
  "against 17 of 218 and 8 of 67 of the others." Record:
  `results/x04_sampled_answers/results/analysis/u0_split_E14_2.csv`.
- **Disagreement difference.** "sampling adds −0.27 [−0.94, 0.40] and −0.33 [−1.00, 0.00] points
  (paired bootstrap)". Record: `results/x04_sampled_answers/results/analysis/X4_SUMMARY.md`.

## Appendix H — Latency

- **Cold first probe.** "The first formal OBQA/Text policy probe took 2022.73 ms and remains in its
  57.64 ms mean probe/selector cost; the other three settings had means between 42.09 and
  42.37 ms." Record: `results/dev_e2e_replay/evidence/WRITING_SYNTHESIS_RECEIPT.json`.
- **First request after loading.** "3,811.7, 936.5, and 3,384.3 ms in the three large-pair replays
  (OBQA Text policy), 708.5, 665.9, and 736.6 ms in the medium-pair replays (fixed C2C), and 1,658.2,
  1,703.9, and 1,622.4 ms in the MMLU-Pro replays (fixed Text)". Every arm:
  `results/r1_figures/E3_FIRST_REQUESTS.md`.
- **Pauses.** "In repeat 1, the MMLU-Pro replay took 22.3 instead of 6.4 minutes because of 181
  pauses longer than one second between requests; … no request was removed." Record:
  `results/e01_e02_e04/results/E3_repeat_mmlu_diagnosis.csv`.
- **Component estimate versus replay.** "For OBQA/Text, a reference-time shift of +28.29 ms was
  offset by +15.60 ms in the selected action and +14.46 ms in online control. The corresponding
  shifts were +15.41, +8.10, and −1.09 ms for OBQA/C2C; +29.69, +9.70, and −0.52 ms for ARC/Text; and
  +18.20, +9.21, and −0.75 ms for ARC/C2C." Record: `results/dev_e2e_replay/summary/component_vs_e2e.csv`.
- **Probe stages** (small/large OBQA panels): "tokenization and prefix construction take 0.87 and
  0.84 ms, transfer 0.13 and 0.12 ms, … label probabilities 0.56 and 0.55 ms, the score 0.08 ms, and
  the selector 0.001 ms" (ARC totals 32.85 and 43.07 ms; medium and MMLU-Pro totals 33.1, 33.2, and
  46.3 ms). Record: `records/boundaries_large_small/records/` (`small_obqa_dev_probes.jsonl.gz`,
  `panel_per_question.jsonl.gz`).
- **Medium-pair replay.** Thresholds "u≤1.3113×10⁻⁶ for OBQA and u≤3.5763×10⁻⁷ for ARC"; the first
  formal OBQA request took "3,269 ms on the fixed path and 814 ms on the policy path"; "Startup
  (75.6 s including runtime import) was excluded". Record: `results/medium_e2e_replay/protocol/`,
  `results/medium_e2e_replay/summary/runtime_diagnostics.csv`.
- **MMLU-Pro replay.** "The first formal fixed-Text request took 5,233 ms"; "Startup (92.6 s) was
  excluded". Record: `results/mmlu_pro_e2e_replay/summary/runtime_diagnostics.csv`.
- **Port check.** Under our parser the small-pair receiver alone answers 191 of 500. Record:
  `results/e01_e02_e04/results/e1_accuracy.csv`.

## Appendix I — Out-of-sample tests

- **Sealed ARC thresholds.** Text "u≤0.01800704002380371"; C2C "u≤0.0007095932960510254". Record:
  `results/sealed_arc_test/protocol/` (`fit_thresholds_large_arc.json`, `large_arc_C.json`).
- **Later tests.** "τ=0.00937 for Llama OBQA at q=0.60, 0.0150 for Llama ARC at q=0.70, and
  3.58×10⁻⁷ for medium ARC at q=0.60". Record: `results/e16/results/analysis_oos/oos_results.csv`.
- **Sealed run.** "Runtime startup (46.4 s) was excluded". Text: "1,073 neutral questions correct and
  80 incorrect under both paths; median latency 324.5 ms (policy) versus 1030.5 ms (fixed)". C2C:
  "1,054 correct and 108 incorrect … median latency 324.4 versus 382.0 ms". Record:
  `results/sealed_arc_test/summary/` (`accuracy_decomposition.csv`, `e2e_latency_utility.csv`,
  `runtime_diagnostics.json`).
- **Download.** "download and file checks handled undecoded Parquet bytes, and the gold column was
  never decoded or joined." Record: `results/sealed_arc_test/evidence/DOWNLOAD_ATTEMPT.json`.
- **Interrupted analysis (later tests).** "One analysis run stopped in a threaded file read after the
  seal receipts were written". Record: `results/e16/DEVIATIONS.md`.

## Appendix J — What the certificates rest on

- **Unparseable share by split.** The ranges printed in Appendix J (0.000 to 0.707 over the
  eighteen main settings, 0.187 to 0.495 over the seven OLMo and Llama settings) are calibration
  values; on development they are 0.000 to 0.709 and 0.159 to 0.403. Pooled over the seven OLMo and
  Llama settings: 0.318 (calibration) and 0.304 (development). Record:
  `results/e11/results/E11a_cell3_share_aggregate.csv`.
- **Filter versus deployable rule.** "The two differ because the non-deployable filter also removes
  questions on which the reference is the unparseable one, and there the receiver's answer genuinely
  differs from what the deployed system returns." Record: `results/e11/E11C_SUMMARY.md`.
- **Precision.** FP32 used "SDPA, memory-efficient backend, TF32 off". "over the others the smallest
  [non-top mass] is 6.04×10⁻⁸, and every zero has a top-two logit gap of at least 16.75". With the
  logit-gap tie-break, "medium/OBQA/C2C deploys q=0.50 instead of 0.55 (Jaccard 0.915 with the
  original omitted set; coverage 48.1 against 52.6%)". Record: `results/e16/RESULTS.md`,
  `results/e16/results/analysis_e16_3/E16_3_RESULTS.json`.

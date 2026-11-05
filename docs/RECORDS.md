# Records the paper points to

Each row is an item the paper says is released, and where it is. Paths are relative to the root of
the dataset [AgentsSci/ICLR2027-SafeToOmit](https://huggingface.co/datasets/AgentsSci/ICLR2027-SafeToOmit)
(`make data` puts it in `data/`); `configs/`, `splits/` and `code/` are also in this repository.
Nothing here was recomputed: every entry points at a stored record.

| # | item | path(s) | notes |
|---|---|---|---|
| 1 | Every prespecified calibration grid test of every setting, incl. rejected thresholds and fallback settings (routed *n*, changed *k*, exact binomial *p*, Clopper–Pearson 0.999 upper bound) | `results/frozen_pipeline/confidence_boundaries/calibration_all_160.csv` (160 = 8 settings x 20 q); per-setting `configs/confidence_boundaries/deployments/{pair}_{obqa,arc}_{T,C}_tests.csv`; medium `results/boundaries_medium/execution_retry1_20260915T164957Z/calibration/{obqa,arc}_{T,C}.csv` (80); MMLU-Pro `results/mmlu_pro_breadth/calibration/40_test_ledger.csv` (40) and `results/e08/analysis/calibration/80_test_ledger.csv` (80); earlier screen `results/historical_large_obqa_screen/summary/calibration_100.csv` (100); declared family `results/zero_gold_controls/summary/calibration_comparison_100.csv` (100 = 60 new + 40 reused from the screen); gold budget `results/historical_large_obqa_gold_budget/summary/calibration_new_300.csv` (300) | 860 ledger rows, 820 distinct tests; every row carries `n_R`/`routed`, `changed`, `p_value`, `CP_upper_0_999`, `accepted`, including the rejected q and the fallback settings, whose selected q is 0 |
| 2 | Medium-pair full 80-test calibration ledger; MMLU-Pro complete calibration ledger (all pairs) | medium: the four 20-row files above (80 rows); MMLU-Pro: `results/mmlu_pro_breadth/calibration/40_test_ledger.csv` (large) + `results/e08/analysis/calibration/80_test_ledger.csv` (small and medium) = 120 tests | |
| 3 | Calibration counts behind the zero-uncertainty (*u* = 0) analysis (E17-1/E17-2) | `results/e17/results/E17_1_u0_rule.csv`, `E17_1_tau0_candidates.csv`, `E17_2_upos_ledgers.csv`, `E17_2_upos_certification.csv` | |
| 4 | Split question IDs: OBQA fit 2,100 / calibration 1,366 / development 742; ARC and MMLU-Pro splits | `splits/development_and_calibration/obqa_{fit,cal,dev}_representatives.json` (2,100 / 1,366 / 742 — counts verified); `arc_{fit,cal,dev}_representatives.json` (670 / 448 / 299); MMLU-Pro `results/mmlu_pro_breadth/splits/{fit,cal,dev}_ids.json` and `*_groups.json.gz`; sealed ARC `splits/sealed_arc_test/`; panels `splits/timing_panels/` | |
| 5 | Model/tokenizer identities with revisions, and hashes of the model config and tokenizer files; label-token sets; prefix and template hashes; first-request checks; per-question probability and timing records | `configs/confidence_boundaries/MODEL_SOURCE_INDEX.json`, `configs/model_identities.json`, `configs/answer_scoring_v2/LABEL_MANIFEST.json`; `configs/confidence_boundaries/probe_protocol_{large,small}/` (`chat_template.jinja`, `FIRST_INPUT_CHECK.json`, prefix/template hashes); probabilities and timings in `records/*/…_probes.jsonl.gz` (`p_labels`, `ProbeMax`, `probe_core_ms`, …) | |
| 6 | Agreement-reference panel comparisons, valid-only rankings, INVALID decompositions, paired-uncertainty diagnostics | `results/frozen_pipeline/confidence_boundaries/{invalid_decomposition.csv, disagreement_base_rates.csv, panel_source_checks.json, paired_bootstrap.csv}`; `results/boundaries_large_small/summary/ranking.csv`; `results/answer_parser_v2/primary_paired_comparisons.csv`; `results/e19/results/E19_2_invalid_changes.csv` | |
| 7 | Label-light certificate (E21) out-of-sample coverage, *W* and delta-accuracy per policy | `results/e21/results/E21_1_oos.csv`, `E21_1_dev.csv`, `E21_1_cert_summary.csv`, `E21_1_cert_candidates.csv`, `E21_1_panel.csv`, `E21_1_appendix_table.csv`; rerun `results/e21b/results/primary_policies.csv` | |
| 8 | Timestamps of the hashed small-sample (AUROC-cut) analysis record (predates E8 and GSM8K calibration results) | `results/e13/results/item4_overall.csv`, `item4_per_setting.csv`, `item4_repro_checks.csv`, `item4_draws.csv.gz`; the recorded file times are in `results/e13/SUMMARY.md` ("File times" under Item 4); the hashed protocol is `results/e13/PREREG_E13.md` with `PREREG_E13.sha256`. Later settings: `results/e16/PREREG.sha256` + `results/e16/RESULTS.md`, `results/x03_llama_receiver/PREREG_X3.sha256`, `results/e20_pilot/PREREG_E20P.sha256`, `results/e20_full/PREREG_E20F.sha256` | The AUROC pattern was first stated in the authors' manuscript draft of 2026-09-20 04:15 UTC (not part of this repository); E13 (item 4) records the later result files' timestamps and fixes the 0.80 cut in its hashed protocol. Verified values below. |
| 9 | Helper-aware score (E9c): the descriptive Text+fact row with the fact in the helper prefill | `results/e09b_e09c/analysis/E9C_RESULTS.json` (keys `helper_prefill_panel_latency`, `settings`, `family`) and `E9C_LEDGERS.json` (row labelled `fact [with-fact prefill, descriptive]`); protocol in `results/e09b_e09c/PROTOCOL_FREEZE_E9C.md` | |
| 10 | All change-direction counts of the rate-matched null control (V1/V2 per population; E22 tables) | `results/e22/results/E22_2_directions.csv` (28 rows: `wrong_to_right`, `wrong_to_wrong`, `changes_S_plus_right_to_wrong`, INVALID splits, `null_g_minus`), `E22_2_share_minus_g.csv`, `E22_1_point.csv`; the V1/V2 null answers themselves in `records/e01_e02_e04/results/e4/*.jsonl.gz` (`variant`, `parsed_original`) | |
| 11 | First formal request of every arm in the three repeated replays (E3) | `records/e03_mmlu_c2c_full/results/records/four_arm_requests_with_correctness.jsonl.gz`, `records/e03_repeat{1,2,3}/mmlu/results/records/four_arm_requests_with_correctness.jsonl.gz` and `records/e03_repeat{1,2,3}/{large,medium}/records/e2e_requests.jsonl.gz` (10 files). Every record carries `order_position` and `parts_ms`; the MMLU-Pro and medium records mark the first request with `arm`, `first_formal_request_global` and `native_first_input_ids`, the large records with `mode` and `cold_first_request` | |
| 12 | The run date written into the Llama-pair C2C helper prompts by the official aligner (X2) | C2C: `Today Date: 19 Sep 2026` (the run date) on all 5,626 C2C rows — recorded in `results/x01_x02_cross_family/DEVIATIONS.md` item C16. Text: the pinned fixed date `Today Date: 26 Jul 2024`, in the three stored rendered helper prompts in `results/x01_x02_cross_family/x1/notes/prepare_checks.json` (`C_examples[*].llama_helper_prompt_rendered`, `fixed_date_present: true`) | |
| 13 | Fully rendered prompt examples per receiver and benchmark, plus prompt-code hashes | `results/e13/PROMPTS_AND_PARSER.md` (OBQA and MMLU-Pro, helper / Text-reading / receiver-only / probe / Text+fact, with the token-count and saved-id checks); SQuAD in `results/e20_pilot/PROMPTS_E20.md`; GSM8K and the Qwen3-8B, Llama-3.1-8B and OLMo-2 receivers in `results/e19/PROMPTS_FOR_APPENDIX.md`; rendered examples also in `results/e01_e02_e04/results/e1/prompts/*.json` and `results/e01_e02_e04/results/e4/prompts_*.json` | |
| 14 | Parser regexes and cue-word lists; released parser SHA-256 must equal `d05978f4…bfcc2f9` | `code/pipeline/answer_scoring_v2/scoring_v2.py` — **computed SHA-256 `d05978f400cf69aa6bddc56168a260ac3ea9158abaa640e9331602dc9bfcc2f9`, matches.** Rules prose in `results/e13/PROMPTS_AND_PARSER.md`; freeze in `configs/answer_scoring_v2/RULE_FREEZE_V2.json` | see note below |
| 15 | Per-question ProbeMax scores and routes, raw action outputs, per-request timing records, hash manifests | `records/**/*_probes.jsonl.gz` (25 files, `ProbeMax` and `p_labels`); routes (87 `*_routes.jsonl` files, e.g. `records/boundaries_large_small/records/*_routes.jsonl`); raw action outputs and requests; timing records (`e2e_requests.jsonl.gz` and others: `latency_ms`, `parts_ms`, component timings); hash lists (`SHA256SUMS`, `*.sha256`) | |

## Note on item 14

The released parser file is byte-identical to the one whose SHA-256 the paper publishes, and the
value agrees with the `parser_sha256` field recorded inside the label records.

## Note on item 1

"Every setting" includes the settings that fell back: their ledgers are present with every q
rejected and a selected q of 0. The lineage on the 1,366 large/OBQA/Text calibration questions is complete: 100 in the declared
family (`zero_gold_controls`; 60 new ProbeMax/ProbeEntropy/WordD, 40 reused D/R), 60 more in the
earlier screen (`historical_large_obqa_screen`; Diff, H, Random) and 300 correctness-trained heads
under label budgets (`historical_large_obqa_gold_budget`; R32/R128/R512 x r0-r4) — 460 distinct
tests. The 40 reused D/R rows are stored in both the screen's and the declared family's ledger, so
they are counted once in the 460.

## Item 8 — the verified timestamps

All values were read from the released files; none was adjusted.

| what | recorded value | expected |
|---|---|---|
| E13 PREREG SHA-256 | `2cc25a61f8ba57a2e129319e9aee1dad3c3df93321377130e38da109b92d3b3e` | `2cc25a61...` — matches |
| E13 PREREG hashed at | `2026-09-21T03:16:48Z` | `03:16:48Z` — matches |
| The 0.80 cut, in the hashed PREREG | "predict \"deploy\" if the ProbeMax AUROC for receiver-reference disagreement, estimated on the sample, is >= .80, otherwise \"fallback\"" (Item 4) | — |
| E8 earliest file with a certification decision, p-value or dev AUROC for the four small/medium MMLU-Pro settings (`analysis/calibration/80_test_ledger.{csv,json}`, `analysis/deployments/deployment_configs.json`) | `2026-09-20T19:10:30Z` | about `19:10Z` — matches |
| E10 GSM8K (`analysis/SEALED_CAL_TESTS.jsonl`) | `2026-09-21T00:11:26Z` | about `00:11Z` — matches |

Settings that ran after E13's protocol was hashed (`2026-09-21T03:16:48Z`):

| setting | hashed protocol | first output / certificate |
|---|---|---|
| E16-5 stronger helper (x2) | `PREREG.md` `17ca4713...`, `2026-09-21T18:44:07Z` | first E16 model output `18:58:58Z`; E16-5 shard env `18:59:19Z`, `18:59:26Z` |
| X3 Llama-3.1-8B receiver (x2) | `PREREG_X3.md` `8a2f6c6d...`, `2026-09-21T06:36:04Z` | step-0 access/download checks from `06:08:43Z`; run outputs after the hash |
| E20 SQuAD | pilot `PREREG_E20P.md` `c02cf958...`, `2026-09-21T23:30:43Z`; full `PREREG_E20F.md` `4c63e4ab...`, `2026-09-22T03:09:28Z` | pilot outputs `2026-09-21T23:58:29Z`; full certificate `2026-09-22T03:24:07Z` |

All five are after `2026-09-21T03:16:48Z`.

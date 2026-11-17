# Files added for version 2

Code, configurations and splits for the four folders added in version 2: the three-agent system on QASC, the 32B helper and receiver pair, batched serving, and the CPU re-analyses (gain retention on unseen questions, per-discipline MMLU-Pro certification, single-model baselines, the medium/OBQA/Text lost-correction test). The per-question records and results are in the Hugging Face dataset (version 2) under the same folder names. Each code folder has a README on how it was laid out and run and what was removed for release (scheduler, lock and submission code).

research IDs: three_agent_qasc = E1, scale_32b = E2b, batched_serving = E2a (version 2), cpu_reanalysis = R0.

| path | what it is | paper element |
|---|---|---|
| `code/analysis/batched_serving/README.md` | how the folder was laid out and run, and what was removed for release | batched serving |
| `code/analysis/batched_serving/analyze_e2a_v2.py` | arm cells, paired comparisons (20,000 batch-paired bootstrap), consistency and sensitivity tables, report | batched serving |
| `code/analysis/cpu_reanalysis/README.md` | how the folder was laid out and run, and what was removed for release | CPU re-analyses: gain retention on unseen questions, per-discipline MMLU-Pro certification, single-model baselines, medium/OBQA/Text lost-correction test |
| `code/analysis/cpu_reanalysis/export_medium_ledger.py` | export, written when the release was built, of the medium/OBQA/Text ledger (j = 1..20) and 36-label query log, using recompute.py's functions | medium/OBQA/Text lost-correction test |
| `code/analysis/cpu_reanalysis/recompute.py` | CPU re-analyses: per-discipline MMLU-Pro certification, gain retention on unseen questions, single-model baselines, medium/OBQA/Text lost-correction test | CPU re-analyses: gain retention on unseen questions, per-discipline MMLU-Pro certification, single-model baselines, medium/OBQA/Text lost-correction test |
| `code/analysis/mmlu_pro_breadth/src/risk.py` | frozen 20-candidate risk calibration (thresholds, masks, binomial tests) | per-discipline MMLU-Pro certification (frozen calibration test loaded by cpu_reanalysis/recompute.py) |
| `code/analysis/scale_32b/README.md` | how the folder was laid out and run, and what was removed for release | 32B helper and receiver pair |
| `code/analysis/scale_32b/e2b_analyze.py` | pilot review, thresholds, calibration tests and label queries, deployments, development/held-out evaluation, timing, report, recheck exports | 32B helper and receiver pair |
| `code/analysis/scale_32b/export_e2_recheck.py` | compatibility recheck exports for the 32B and batched-serving results | 32B helper and receiver pair |
| `code/analysis/three_agent_qasc/README.md` | how the folder was laid out and run, and what was removed for release | three-agent system on QASC |
| `code/analysis/three_agent_qasc/calibrate.py` | calibration tests (change rate, lost corrections at epsilon 0.01/0.02), label query rule, policy freeze | three-agent system on QASC |
| `code/analysis/three_agent_qasc/collect_and_thresholds.py` | merge check of fit records and the 20 frozen candidate thresholds | three-agent system on QASC |
| `code/analysis/three_agent_qasc/dev_report.py` | development accuracies, controls, ablations and policy summaries | three-agent system on QASC |
| `code/analysis/three_agent_qasc/deviations_timeline_evidence.py` | evidence behind the deviations timeline | three-agent system on QASC |
| `code/analysis/three_agent_qasc/export_recheck.py` | export of the recheck CSV files (fit, calibration, thresholds, policies, label queries) | three-agent system on QASC |
| `code/analysis/three_agent_qasc/fit_accuracy_and_p1p2_prediction.py` | fit accuracies of A+C and A+B+C and the plug-in P1/P2 prediction | three-agent system on QASC |
| `code/analysis/three_agent_qasc/fit_controls_accuracy.py` | fit accuracies of the controls "C alone" and "C reads the facts" (descriptive) | three-agent system on QASC |
| `code/analysis/three_agent_qasc/fit_diagnostics.py` | fit-split score diagnostics and the label-free P0 prediction | three-agent system on QASC |
| `code/analysis/three_agent_qasc/leak_stats.py` | shares of A messages naming an option and B messages containing the correct option text | three-agent system on QASC |
| `code/analysis/three_agent_qasc/sealed_eval.py` | one-time sealed-test evaluation: R1, R2, gains with paired bootstrap | three-agent system on QASC |
| `code/analysis/three_agent_qasc/sealed_gate.py` | version gate before the sealed-test evaluation | three-agent system on QASC |
| `code/analysis/three_agent_qasc/smoke_report.py` | pre-freeze smoke-test checks and cost estimate | three-agent system on QASC |
| `code/analysis/three_agent_qasc/timing_report.py` | timing-panel report: paired savings with bootstrap intervals, component breakdown | three-agent system on QASC |
| `code/analysis/three_agent_qasc/tune_report.py` | pre-freeze B-prompt tuning report | three-agent system on QASC |
| `code/analysis/three_agent_qasc/unused_audit.py` | audit that the 1,051 unused questions were never used | three-agent system on QASC |
| `code/pipeline/batched_serving/README.md` | how the folder was laid out and run, and what was removed for release | batched serving |
| `code/pipeline/batched_serving/e2a_v2_batch.py` | one batch under the synchronous, early-return and reference arms; arm rotation | batched serving |
| `code/pipeline/batched_serving/e2a_v2_gate.py` | freeze check and unit start for the batched replay | batched serving |
| `code/pipeline/batched_serving/prepare_e2a_v2.py` | frozen batched-serving configuration on the full development sets (OBQA 742, ARC 299) | batched serving |
| `code/pipeline/batched_serving/run_e2a_v2.py` | batched replay over model pairs, datasets and B = 1, 8, 32 on one exclusive node | batched serving |
| `code/pipeline/scale_32b/README.md` | how the folder was laid out and run, and what was removed for release | 32B helper and receiver pair |
| `code/pipeline/scale_32b/e2_debug_ops.py` | shard units, completion checks, one-smaller-retry layout, preflight barrier (scheduler functions removed) | 32B helper and receiver pair |
| `code/pipeline/scale_32b/e2b_formal_gate.py` | stage-release checks between generation stages | 32B helper and receiver pair |
| `code/pipeline/scale_32b/e2b_gate.py` | freeze check and the fixed question order of every stage | 32B helper and receiver pair |
| `code/pipeline/scale_32b/e2b_plan45.py` | shard-plan overlay and the analysis/worker/export entry used for the formal stages (submission removed) | 32B helper and receiver pair |
| `code/pipeline/scale_32b/e2b_runtime_checks.py` | two-question consistency preflight and non-statistical warmups | 32B helper and receiver pair |
| `code/pipeline/scale_32b/hf32_formal_runtime.py` | formal log-space probe score over A-D and the timed policy/reference/receiver-alone arms | 32B helper and receiver pair |
| `code/pipeline/scale_32b/hf32_runtime.py` | four-GPU layer-sharded HF runtime: prompts, helper message, receiver/Text/helper-alone answers, smoke probe | 32B helper and receiver pair |
| `code/pipeline/scale_32b/prepare_e2b.py` | prepared label-free inputs and the base E2b configuration | 32B helper and receiver pair |
| `code/pipeline/scale_32b/prepare_e2b_formal.py` | formal E2b configuration (score, policies, timing order, deadlines) | 32B helper and receiver pair |
| `code/pipeline/scale_32b/prepare_e2b_formal_inputs.py` | restores two option texts altered by an earlier anonymization (pre-freeze input repair) | 32B helper and receiver pair |
| `code/pipeline/scale_32b/run_e2b_formal.py` | per-shard worker for pilot, fit, calibration, development, held-out and timing replay | 32B helper and receiver pair |
| `code/pipeline/three_agent_qasc/README.md` | how the folder was laid out and run, and what was removed for release | three-agent system on QASC |
| `code/pipeline/three_agent_qasc/build_config.py` | writes the frozen E1 configuration (models, label-token sets, prefix ids, score, parser) | three-agent system on QASC |
| `code/pipeline/three_agent_qasc/clusterA_check.py` | cross-cluster consistency gate on 4 fit questions before generating on the second cluster | three-agent system on QASC |
| `code/pipeline/three_agent_qasc/cpu_preflight.py` | CPU preflight: label-token sets re-derived from the vocabulary, prompt renderings | three-agent system on QASC |
| `code/pipeline/three_agent_qasc/data_prep.py` | QASC download check, field checks and de-duplication; writes the label-free query file | three-agent system on QASC |
| `code/pipeline/three_agent_qasc/e1_generate.py` | per-shard generation of A, B, C (skip and reference paths), the probe and the controls, batch size 1 | three-agent system on QASC |
| `code/pipeline/three_agent_qasc/e1lib.py` | shared library: agent prompts (A, B, C on every path), model calls, ProbeMax probe and log-space score u, candidate thresholds, skip rule | three-agent system on QASC |
| `code/pipeline/three_agent_qasc/freeze_b_prompt.py` | records the approved B prompt and its SHA-256 | three-agent system on QASC |
| `code/pipeline/three_agent_qasc/layout_check.py` | two-GPU layout test on 4 fit questions before the timing panel | three-agent system on QASC |
| `code/pipeline/three_agent_qasc/make_splits.py` | seeded split into fit 2,000 / calibration 3,000 / development 1,000 / sealed 2,000 and the 1,051 unused ids | three-agent system on QASC |
| `code/pipeline/three_agent_qasc/replay_e2e.py` | live end-to-end timing panel: policy and reference alternate per request, per-component times | three-agent system on QASC |
| `configs/batched_serving/README.md` | where the prompts and probes of the replayed settings are | batched serving |
| `configs/batched_serving/e2a_config.json` | base configuration extended by e2a_v2_config.json | batched serving |
| `configs/batched_serving/e2a_v2_config.json` | frozen version-2 configuration (batch sizes, arms, pairs, deployments) | batched serving |
| `configs/batched_serving/model_identities.json` | model repositories and revisions, runtime settings, hardware | batched serving |
| `configs/batched_serving/chat_templates/large.helper.Qwen2.5-7B-Instruct.jinja` | chat template of the model, as used | batched serving |
| `configs/batched_serving/chat_templates/large.receiver.Qwen3-8B.jinja` | chat template of the model, as used | batched serving |
| `configs/batched_serving/chat_templates/llama.helper.Qwen2.5-7B-Instruct.jinja` | chat template of the model, as used | batched serving |
| `configs/batched_serving/chat_templates/llama.receiver.Llama-3.1-8B-Instruct.jinja` | chat template of the model, as used | batched serving |
| `configs/cpu_reanalysis/analysis_parameters.json` | test levels, bootstrap and threshold rule of the re-analyses | CPU re-analyses: gain retention on unseen questions, per-discipline MMLU-Pro certification, single-model baselines, medium/OBQA/Text lost-correction test |
| `configs/cpu_reanalysis/model_identities.json` | model repositories and revisions, runtime settings, hardware | CPU re-analyses: gain retention on unseen questions, per-discipline MMLU-Pro certification, single-model baselines, medium/OBQA/Text lost-correction test |
| `configs/scale_32b/OBQA_TEMPLATE_CHECK.json` | check that the receiver templates render identical OBQA prompts | 32B helper and receiver pair |
| `configs/scale_32b/e2b_formal_config.json` | frozen configuration (models, device maps, decoding, label tokens, inputs, input repairs) | 32B helper and receiver pair |
| `configs/scale_32b/helper_download_verified.json` | verification of the 27 helper repository files at the frozen revision | 32B helper and receiver pair |
| `configs/scale_32b/label_token_sets.json` | probe prefix, prefix token ids and the label token sets | 32B helper and receiver pair |
| `configs/scale_32b/model_identities.json` | model repositories and revisions, runtime settings, hardware | 32B helper and receiver pair |
| `configs/scale_32b/chat_templates/helper.Qwen2.5-32B-Instruct.jinja` | chat template of the model, as used | 32B helper and receiver pair |
| `configs/scale_32b/chat_templates/receiver.Qwen3-32B.jinja` | chat template of the model, as used | 32B helper and receiver pair |
| `configs/scale_32b/rendered_examples/rendered_examples.json` | prompts of every model and path for two example questions, rendered when the release was built (tokenizers only) and checked against the stored token counts; see results/scale_32b/DERIVED.md in the dataset | 32B helper and receiver pair |
| `configs/scale_32b/rendered_examples/rendered_examples.txt` | prompts of every model and path for two example questions, rendered when the release was built (tokenizers only) and checked against the stored token counts; see results/scale_32b/DERIVED.md in the dataset | 32B helper and receiver pair |
| `configs/three_agent_qasc/CODE_HASHES.txt` | SHA-256 of the code files that build the inputs and the score | three-agent system on QASC |
| `configs/three_agent_qasc/dataset_identity.json` | QASC source, revision, file hashes, licence, fields used, de-duplication rule | three-agent system on QASC |
| `configs/three_agent_qasc/e1_config.json` | frozen configuration (models with file hashes, decoding, probe, parser, threshold rule) | three-agent system on QASC |
| `configs/three_agent_qasc/label_token_sets.json` | probe prefix, prefix token ids and the label token sets | three-agent system on QASC |
| `configs/three_agent_qasc/model_identities.json` | model repositories and revisions, runtime settings, hardware | three-agent system on QASC |
| `configs/three_agent_qasc/split_config.json` | split and de-duplication rule | three-agent system on QASC |
| `configs/three_agent_qasc/policies/P0.json` | frozen policy (P0, P1, P2) or the policy-freeze record | three-agent system on QASC |
| `configs/three_agent_qasc/policies/P1.json` | frozen policy (P0, P1, P2) or the policy-freeze record | three-agent system on QASC |
| `configs/three_agent_qasc/policies/P2.json` | frozen policy (P0, P1, P2) or the policy-freeze record | three-agent system on QASC |
| `configs/three_agent_qasc/policies/POLICY_FREEZE.json` | frozen record | three-agent system on QASC |
| `configs/three_agent_qasc/chat_templates/helper.Qwen2.5-7B-Instruct.jinja` | chat template of the model, as used | three-agent system on QASC |
| `configs/three_agent_qasc/chat_templates/receiver.Qwen3-8B.jinja` | chat template of the model, as used | three-agent system on QASC |
| `configs/three_agent_qasc/rendered_examples/rendered_example_fit.json` | fully rendered inputs of every agent and path for example questions | three-agent system on QASC |
| `configs/three_agent_qasc/rendered_examples/rendered_example_fit.txt` | fully rendered inputs of every agent and path for example questions | three-agent system on QASC |
| `configs/three_agent_qasc/rendered_examples/rendered_inputs_from_runs.jsonl` | C inputs (skip and reference paths) and A/B user messages saved during the runs (2 per shard) | three-agent system on QASC |
| `configs/three_agent_qasc/prompts/B_prompt_v1.json` | pre-freeze B prompt (v1), superseded | three-agent system on QASC |
| `configs/three_agent_qasc/prompts/B_prompt_v2.json` | frozen prompt of agent B | three-agent system on QASC |
| `configs/three_agent_qasc/prompts/FROZEN_B_PROMPT.json` | freeze record of the B prompt | three-agent system on QASC |
| `splits/batched_serving/SPLITS.md` | populations used and where their IDs are | batched serving |
| `splits/batched_serving/batch_composition.csv` | IDs of every batch per pair, dataset and B | batched serving |
| `splits/batched_serving/replay_order.csv` | execution order of every arm and batch | batched serving |
| `splits/cpu_reanalysis/SPLITS.md` | populations used and where their IDs are | CPU re-analyses: gain retention on unseen questions, per-discipline MMLU-Pro certification, single-model baselines, medium/OBQA/Text lost-correction test |
| `splits/scale_32b/SPLITS.md` | populations used and where their IDs are | 32B helper and receiver pair |
| `splits/scale_32b/timing_panel_ids.json` | the 128-question development timing panel (byte copy of the version-1 file) | 32B helper and receiver pair |
| `splits/three_agent_qasc/SPLIT_MANIFEST.json` | split sizes, seed, hashes and de-duplication counts | three-agent system on QASC |
| `splits/three_agent_qasc/cal_ids.json` | question IDs of the split, in order | three-agent system on QASC |
| `splits/three_agent_qasc/dedup_map.json` | de-duplication rule, counts and the 9 dropped -> kept pairs | three-agent system on QASC |
| `splits/three_agent_qasc/dev_ids.json` | question IDs of the split, in order | three-agent system on QASC |
| `splits/three_agent_qasc/fit_ids.json` | question IDs of the split, in order | three-agent system on QASC |
| `splits/three_agent_qasc/panel128_dev_ids.json` | question IDs of the split, in order | three-agent system on QASC |
| `splits/three_agent_qasc/smoke_ids_fit.json` | pre-freeze smoke questions (fit subset) | three-agent system on QASC |
| `splits/three_agent_qasc/test_sealed_ids.json` | question IDs of the split, in order | three-agent system on QASC |
| `splits/three_agent_qasc/tune_ids_fit.json` | pre-freeze B-prompt tuning questions (fit subset) | three-agent system on QASC |
| `splits/three_agent_qasc/unused_ids.json` | question IDs of the split, in order | three-agent system on QASC |

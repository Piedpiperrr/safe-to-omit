# batched_serving configuration

research ID: E2a (version 2)

- `e2a_v2_config.json`: frozen version-2 configuration (batch sizes, arms, model pairs, frozen deployments and thresholds); `e2a_config.json`: the base configuration it extends.
- `model_identities.json` and `chat_templates/`: models at their revisions.
- Prompts, label token sets and the probe are those of the frozen main-study Text settings (version 1: `configs/dev_e2e_replay/protocol/` for large/Qwen3-8B and `configs/confidence_boundaries/`; the Llama-3.1-8B receiver setting is `results/x03_llama_receiver/`). Batching adds left padding only; per-question answers, routes, scores and helper messages are compared with each arm's own B = 1 run in `results/batched_serving/results/e2a_v2_consistency.csv`.

# cpu_reanalysis splits

research ID: R0. No new split. The populations are those of version 1:

- OBQA fit / calibration / development 2,100 / 1,366 / 742 and ARC fit / calibration / development 670 groups / 448 / 299:
  `splits/development_and_calibration/`.
- OBQA held-out 744 (not sealed; used for the unseen-question rows of the gain-retention analysis): `results/e09b_e09c/inputs/holdout_744_ids.json`.
- ARC test 1,172 (sealed ARC confirmation, later used for the Llama out-of-sample rows): `splits/sealed_arc_test/`.
- MMLU-Pro fit / calibration / development 3,000 / 6,000 / 2,641 groups, with disciplines: `results/mmlu_pro_breadth/splits/`.
- Llama-8B timing panels (128 development questions each): `splits/timing_panels/`.

# scale_32b splits

research ID: E2b. The 32B pair reuses the paper's existing OpenBookQA populations, released in version 1; no new split was made. IDs, order and options are those of the version-1 files below.

| population | questions | version-1 file | SHA-256 |
|---|---:|---|---|
| fit | 2,100 | `splits/development_and_calibration/obqa_fit_ids.json` | `1559bc9668d88a986df0967b96fdf14c995f02f622b126f396819b69b58c4028` |
| calibration | 1,366 | `splits/development_and_calibration/obqa_cal_ids.json` | `8b33a4d168131c5a0095c5eabcdea2f2b3bb272dfae8a41ab2a7e0e05d5ebaed` |
| development | 742 | `splits/development_and_calibration/obqa_dev_ids.json` | `80d1bbdd9148df3037401881533a2137b0d4ce9ee90a8ebb424fb413440bfbab` |
| held-out (not sealed; see below) | 744 | `results/e09b_e09c/inputs/holdout_744_ids.json` | `5129d91c60e6d9656520face24e09f470cb61271a996681338530c22d2147249` |
| timing panel (development, fixed order) | 128 | `splits/timing_panels/obqa_dev_panel_ids_e2e_replay.json` | `6ebbd7604ce342d43219288b4235053026449b1d9684cff73044be356b3263ac` |

`timing_panel_ids.json` here is a byte copy of the version-1 timing panel file (the 128 development IDs in replay order).

The 744 held-out questions were never used by this study before this experiment, but an earlier project stage scored small-pair outputs on them, so they are a held-out test for this model pair, not a sealed one. The protocol describes the whole experiment as a model-scale extension on previously exposed benchmarks.

Input repair: two option texts that the anonymised version-1 inputs had replaced by a placeholder were restored before any formal output (fit `14-1424`, option D "eagle"; development `12-577`, option C "Eagle"); see `results/scale_32b/DEVIATIONS.md`, section 2. IDs, order and gold labels are unchanged.

Shards: fit = the 300-question pilot plus 18 contiguous shards of the remaining 1,800; calibration 14 shards; development and held-out 9 shards each; timing 4 segments of 32. Per-shard metadata: `records/scale_32b/shard_metadata/`.

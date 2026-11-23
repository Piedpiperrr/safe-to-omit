# Records moved out of the PDF in arXiv version 2

arXiv version 2 prints shortened forms of the records below and points here for the rest; this file gives them in full, as printed in earlier versions of the paper. Appendix letters are those of arXiv version 2.

## Appendix J.1 (three-agent system): calibration tests for all 20 candidates

Calibration tests (N=3,000). n: skipped; m: answer changes among them; p_change=Pr[Bin(n,0.05)≤ m]; l: lost corrections; p_ε=Pr[Bin(N,ε)≤ l]. A test passes at p≤0.001. –: beyond the largest candidate passing the change-rate test, so not labeled. Bold: P1 (j=8), P2 (j=10), P0 (j=11).

| j | q | n | m | p_change | l | p_0.01 | p_0.02 |
|---|---|---|---|---|---|---|---|
| 1 | 0.05 | 141 | 1 | 6.09 × 10^−3 | 1 | 2.52 × 10^−12 | 2.97 × 10^−25 |
| 2 | 0.10 | 293 | 1 | 4.88 × 10^−6 | 1 | 2.52 × 10^−12 | 2.97 × 10^−25 |
| 3 | 0.15 | 464 | 1 | 1.17 × 10^−9 | 1 | 2.52 × 10^−12 | 2.97 × 10^−25 |
| 4 | 0.20 | 600 | 1 | 1.40 × 10^−12 | 1 | 2.52 × 10^−12 | 2.97 × 10^−25 |
| 5 | 0.25 | 744 | 2 | 2.15 × 10^−14 | 2 | 3.94 × 10^−11 | 9.23 × 10^−24 |
| 6 | 0.30 | 884 | 2 | 2.29 × 10^−17 | 2 | 3.94 × 10^−11 | 9.23 × 10^−24 |
| 7 | 0.35 | 1,053 | 5 | 1.65 × 10^−17 | 5 | 2.03 × 10^−8 | 3.70 × 10^−20 |
| **8** | 0.40 | 1,171 | 7 | 6.09 × 10^−18 | 7 | 4.78 × 10^−7 | 3.41 × 10^−18 |
| 9 | 0.45 | 1,330 | 18 | 7.17 × 10^−13 | 18 | 0.0126 | 1.45 × 10^−10 |
| **10** | 0.50 | 1,446 | 24 | 1.70 × 10^−11 | 23 | 0.113 | 3.38 × 10^−8 |
| **11** | 0.55 | 1,612 | 40 | 2.45 × 10^−7 | 39 | 0.955 | 0.00235 |
| 12 | 0.60 | 1,783 | 62 | 0.0012 | – | – | – |
| 13 | 0.65 | 1,964 | 96 | 0.436 | – | – | – |
| 14 | 0.70 | 2,131 | 137 | 0.998 | – | – | – |
| 15–20 | 0.75–1 | 2,257–3,000 | 169–488 | ≈1 | – | – | – |

## Appendix J.1 (three-agent system): the thirteen deviations and their timing

Deviations from the frozen protocol, with their timing relative to the first reading of calibration (Cal), development (Dev), sealed-test (Test), and timing outputs.

None changes a model input, the score, a primary endpoint, or a sample rule. The released dataset holds the same entries in `results/three_agent_qasc/DEVIATIONS.md` and `DEVIATIONS_TIMELINE.md`.

| # | Deviation | When |
|---|---|---|
| 1 | Leak shares quoted in the protocol were measured on 96 prompt-tuning questions; full-split values are reported (superseded by 9). | before all outputs |
| 2 | Generation ran on single GPUs (A, B, and C on one A100); the timing layout is unchanged. | before all outputs |
| 3 | Scheduling fix (GPU lock, chained submission) after two jobs collided before writing any record. | before all outputs |
| 4 | Fit-split controls (C alone, C reading the facts) added; descriptive. | before all outputs |
| 5 | The freeze record names its parent commit; the freeze commit is identified by a tag. | before all outputs |
| 6 | Remaining shards ran on a second cluster with the same GPU model and environment, behind an exact-match gate on four fit questions, with a per-shard lock; scheduling-only extensions followed. | before all outputs; extended after Cal and after Dev |
| 7 | The calibration analysis need not wait for the fit controls of entry 4, whose accuracies may be committed after calibration outputs are read; descriptive. | before all outputs |
| 8 | Calibration tests ran before the development outputs and fit controls were complete. | after Cal and Dev, before Test |
| 9 | Correction of the protocol's leak shares (pre-freeze subsample); full-split values are used. | after Cal and Dev, before Test |
| 10 | Clarification: C's reference history contains B's instruction without the facts and A's note, as in Text+fact; the code is unchanged since the freeze. | during test generation, before Test |
| 11 | Correction: the protocol's 60.7% zero share and 12 merged candidates came from the first 112 fit questions; the full fit split gives 57.15% and 11. | after Test, before timing |
| 12 | The timing panel ran on the second cluster (same GPU model, exclusive node), one run per policy enforced by a lock; failure handling fixed before timing. | before timing |
| 13 | Clarification: the 1,051 deduplicated questions outside the four splits were never processed by any model. | after all outputs |

## Appendix J.2 (32B pair): calibration tests for all 20 candidates

Calibration tests for the 32B pair (N=1,366); notation as in the J.1 table above. –: beyond the largest candidate passing the change-rate test, so not labeled. Bold: the primary policy (j=14), the ε=0.02 policy (j=16), and the label-free policy (j=17).

| j | q | n | m | p_change | l | p_0.01 | p_0.02 |
|---|---|---|---|---|---|---|---|
| 1 | 0.05 | 72 | 0 | 0.0249 | 0 | 1.09 × 10^−6 | 1.03 × 10^−12 |
| 2 | 0.10 | 139 | 0 | 8.01 × 10^−4 | 0 | 1.09 × 10^−6 | 1.03 × 10^−12 |
| 3–11 | 0.15–0.55 | 187–722 | 0 | ≤6.83 × 10^−5 | 0 | 1.09 × 10^−6 | 1.03 × 10^−12 |
| 12 | 0.60 | 789 | 1 | 1.13 × 10^−16 | 1 | 1.61 × 10^−5 | 2.99 × 10^−11 |
| 13 | 0.65 | 866 | 2 | 5.54 × 10^−17 | 2 | 1.20 × 10^−4 | 4.32 × 10^−10 |
| **14** | 0.70 | 924 | 5 | 6.51 × 10^−15 | 3 | 5.96 × 10^−4 | 4.16 × 10^−9 |
| 15 | 0.75 | 996 | 7 | 1.58 × 10^−14 | 5 | 0.00675 | 1.74 × 10^−7 |
| **16** | 0.80 | 1,079 | 12 | 2.57 × 10^−12 | 8 | 0.0722 | 1.27 × 10^−5 |
| **17** | 0.85 | 1,144 | 21 | 1.93 × 10^−8 | 13 | 0.501 | 0.00173 |
| 18 | 0.90 | 1,229 | 39 | 0.00117 | – | – | – |
| 19 | 0.95 | 1,298 | 53 | 0.070 | – | – | – |
| 20 | 1.00 | 1,366 | 85 | 0.981 | – | – | – |

## Appendix J.2 (32B pair): deviations

Every difference between the protocol and its execution was recorded as it occurred. None changes a model input, the score, a test, a policy, or a pass rule. (i) Before any formal output, two option strings that an anonymization step had altered in our copy of the inputs were restored to the original OBQA text. (ii) Generation ran as one-hour jobs on exclusively allocated nodes, in contiguous shards; every job first regenerated two pilot questions and required an exact match of the helper message, both answers, and the score. (iii) After the pilot, the shard sizes were reduced using its measured load time, and a rule that pauses submission on measured run time replaced one based on an extrapolation. (iv) One submission was refused before entering the queue for lack of disk space and was resubmitted unchanged. (v) The label log-probabilities behind the score and the component times of the replay were not saved; the score code, which was fixed before the run, and the route split above stand in for them.

The released dataset holds them in `results/scale_32b/DEVIATIONS.md`.

## Appendix B: extensions and their freeze order (full paragraph)

The medium pair, MMLU-Pro (large pair first; small and medium later as an 80-test family), cross-family pairs, Text+fact, stronger helper, GSM8K, SQuAD, and sampled answers were each added after earlier results (post-hoc motivated). Each used a protocol frozen before any of its outputs, committing to report every setting (fixing models, fusers, populations, paths, parser, score, grid, acceptance, and analysis). The medium pair's first execution stopped on an engineering guard before any action or probe and was rerun unchanged. For medium-pair and MMLU-Pro replays, thresholds and panel identities were frozen before any request. MMLU-Pro went first to the large pair because the compute budget allowed one pair with the full 6,000-question calibration split; its thresholds, calibration decisions, and deployments were written before gold labels were read for accuracy. Before collecting new outputs, later runs reproduced saved outputs bit for bit: cross-family code on 20 large and 16 small fit questions, small and medium MMLU-Pro and GSM8K on saved OBQA fit rows, sampling on saved greedy outputs, and SQuAD on 16 pilot questions. Smoke tests read no gold labels.

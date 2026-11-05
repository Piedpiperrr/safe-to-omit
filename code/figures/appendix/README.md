# Appendix figures (Figures 3–11)

```bash
bash code/figures/appendix/run_all.sh        # writes figures/figure_*.pdf, about 45 s on CPU
python3 code/figures/appendix/fig05_boundaries.py /tmp/fig3.pdf   # one figure, any output path
```

Tested with the versions pinned in [requirements.txt](../../../requirements.txt); the output is
identical to the figures in the paper. `make figures` at the repository root runs every script with
the settings in `code/figures/matplotlibrc` and checks that each PDF is byte-identical to the
committed one. The script
file names keep the numbering of an earlier draft; the "figure" column gives the paper's numbers.

| script | figure | reads |
|---|---|---|
| `fig05_boundaries.py` | Fig. 3, App. C | `results/boundaries_large_small/summary/calibration_all_160.csv`, `risk_coverage_curves.csv`; `configs/confidence_boundaries/deployments/all.json`; Tables 4–5 (`paper_tex/new_app_D.tex`) |
| `fig06_baselines.py` | Fig. 4, App. C | `record_tables/baselines.tex`; `results/boundaries_large_small/summary/calibration_all_160.csv`; Table 4; Table 1 |
| `fig11_binormal.py` | Fig. 5, App. D | `results/e17/results/E17_5c_grid.csv`, `E17_5c_contour.csv`, `E17_5b_binormal.csv`; Table 4 |
| `fig07_corrections.py` | Fig. 6, App. E | Table 10 (`paper_tex/new_app_F.tex`); `record_tables/omit_keep.tex`, `record_tables/patterns.tex`; Table 4 |
| `fig03_null_paired.py` | Fig. 7, App. F | Tables 13–14 (`paper_tex/new_app_C.tex`), Table 1 |
| `fig04_null_sources.py` | Fig. 8, App. F | `record_tables/patterns.tex`; `results/e09a/results/e9a_overlap.csv`, `e9a_receiver_confidence.csv`; Table 1 |
| `fig08_savings.py` | Fig. 9, App. H | Tables 18, 20, 21 (`paper_tex/new_app_G.tex`); `results/boundaries_large_small/summary/panel_cost.csv` |
| `fig09_oos.py` | Fig. 10, App. I | Tables 23–25 (`paper_tex/new_app_H.tex`), App. G.3, Table 2 |
| `fig10_zero_u.py` | Fig. 11, App. J | Tables 26 and 29 (`paper_tex/new_app_I.tex`) |

- `paper_tex/` holds the LaTeX of the paper's appendix sections and of main Tables 1–2. The
  scripts parse the numbers they plot from it, so a figure cannot drift from the printed tables.
- `record_tables/` holds three tables of the fuller record that the appendix condenses (all eight
  correctness patterns, the reference corrections each policy loses or keeps, and the same-target
  baselines with their component latencies). They use the earlier word "omit" for "skip".
- Every script recomputes intervals and p-values from counts, checks the numbers against the other
  tables and against the records where a released CSV exists, asserts the claims the appendix
  makes about the figure, and stops with an `AssertionError` if anything disagrees. Figures 4, 7,
  9, 10 and 11 also check their own layout (label clearances) and stop if a label would collide.

"""Run from any directory: python3 code/figures/appendix/fig11_binormal.py [output.pdf]

Figure 5 (Appendix D, fig:binormal): binormal certification probability over disagreement rate
and AUROC at the three fit/calibration sizes, with the 28 settings.

Regenerated from the released E17-5 records (results/e17/results/) at 8 pt;
replaces the earlier code/figures/figure_binormal_grid.py (P(certify) as a single-hue green ramp, the colour
of the certified region elsewhere).
Checks: the model's prediction agrees with every observed outcome, with the
development and the calibration AUROC; observed outcomes agree with Table 4
(tab:dev_full, new_app_D.tex); the ranges stated in Appendix D (AUROC needed 0.63 to 0.996;
certified probabilities 0.61 to 1; no fallback above 0.05) are asserted.
"""
import csv, re, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from pathlib import Path
HERE = Path(__file__).resolve().parent            # code/figures/appendix
ROOT = HERE.parents[2]                             # repository root
TEX = HERE / "paper_tex"                           # appendix LaTeX of the submitted paper
REC = HERE / "record_tables"                       # fuller tables that the appendix condenses
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "figure_binormal.pdf"
rd = lambda n: list(csv.DictReader(open(ROOT / "results/e17/results" / n)))
grid, cont, pts = rd("E17_5c_grid.csv"), rd("E17_5c_contour.csv"), rd("E17_5b_binormal.csv")
D = open(TEX / "new_app_D.tex").read()

assert len(pts) == 28
for r in pts:
    assert r["agree_devAUROC"] == "True" and r["agree_calAUROC"] == "True", r["setting"]
# observed outcomes against Table 7
t7 = D[D.index("\\label{tab:dev_full}"):D.index("\\end{table}", D.index("\\label{tab:dev_full}"))]
dev = {}
for l in t7.split("\\\\"):
    c = [x.strip() for x in l.split("&")]
    if len(c) == 10:
        c[0] = c[0].split()[-1] if c[0] else c[0]
        if c[0] in ("small", "medium", "large", "OLMo", "Llama", "Llama-8B"):
            dev[f"{c[0]}/{c[1]}/{re.sub(r'[$].*?[$]', '', c[2])}"] = c[6] != "fallback" and float(c[6]) > 0
alias = {"X1-Llama": "Llama", "X2-OLMo": "OLMo", "X3-Llama8B": "Llama-8B"}
seen = 0
for r in pts:
    pair, rest = r["setting"].split("/", 1)
    s = alias.get(pair, pair) + "/" + rest
    if s in dev:
        assert dev[s] == (r["observed"] == "certify"), s; seen += 1
assert seen == 23
cert = [r for r in pts if r["observed"] == "certify"]; fb = [r for r in pts if r["observed"] == "fallback"]
assert f"{min(float(r['P_certify_devAUROC']) for r in cert):.2f}" == "0.61" and max(float(r["P_certify_devAUROC"]) for r in cert) == 1.0
assert max(float(r["P_certify_devAUROC"]) for r in fb) <= 0.05
a50 = {r["setting"]: float(r["AUROC_at_P50_value"]) for r in pts}
assert min(a50, key=a50.get) == "large/ARC/Text" and f"{a50['large/ARC/Text']:.2f}" == "0.63"
assert max(a50, key=a50.get) == "X1-Llama/ARC/C2C" and f"{a50['X1-Llama/ARC/C2C']:.4f}".startswith("0.996")
print("checks passed")

FS, FS_H = 8.0, 8.5
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, AXIS = "#0b0b0b", "#52514e", "#c3c2b7"
cmap = LinearSegmentedColormap.from_list("pass", ["#ffffff", "#e3f4ea", "#a9d9bb", "#5fb383", "#2f7a4e"])
SIZES = [(670, 448, "ARC"), (2100, 1366, "OBQA"), (3000, 6000, "MMLU-Pro")]
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.5), sharey=True)
fig.subplots_adjust(left=0.085, right=0.905, bottom=0.28, top=0.845, wspace=0.12)
for ax, (nf, nc, name) in zip(axes, SIZES):
    g = [r for r in grid if (int(r["N_fit"]), int(r["N_cal"])) == (nf, nc)]
    P = sorted({float(r["prevalence"]) for r in g}); A = sorted({float(r["AUROC"]) for r in g})
    Z = np.full((len(A), len(P)), np.nan)
    for r in g:
        Z[A.index(float(r["AUROC"])), P.index(float(r["prevalence"]))] = float(r["P_certify"])
    assert not np.isnan(Z).any()
    im = ax.pcolormesh(P, A, Z, shading="nearest", cmap=cmap, vmin=0, vmax=1, edgecolors="face", linewidth=0.2)
    c = sorted((float(r["prevalence"]), float(r["AUROC_at_P50_value"])) for r in cont
               if (int(r["N_fit"]), int(r["N_cal"])) == (nf, nc) and r["AUROC_at_P50_value"] not in ("", "None"))
    ax.plot([x for x, _ in c], [y for _, y in c], color=INK, lw=1.0, solid_capstyle="butt", zorder=4)
    for r in pts:
        if (int(r["N_fit"]), int(r["N_cal"])) != (nf, nc):
            continue
        col = ORANGE if r["setting"].endswith("/C2C") else BLUE
        mk = "^" if r["setting"].endswith("/C2C") else "o"
        dep = r["observed"] == "certify"
        ax.plot(float(r["prevalence_cal"]), float(r["AUROC_dev"]), marker=mk, ms=4.6, mew=0.9, color=col,
                mfc=col if dep else "white", mec=col, ls="none", clip_on=False, zorder=5)
    ax.set_xlim(0, 0.63); ax.set_ylim(0.59, 1.00)   # room for the settings at 0.61 and 0.62
    ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_xticklabels(["0", "0.2", "0.4", "0.6"])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.tick_params(length=2.5, width=0.5, color=AXIS, labelcolor=INK2, labelsize=FS, pad=2)
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.set_title(f"{name} split sizes\nfit {nf:,}, calibration {nc:,}", fontsize=FS, color=INK, pad=3, linespacing=1.1)
axes[0].set_ylabel("AUROC (development)", fontsize=FS, color=INK2, labelpad=2)
axes[1].set_xlabel("Disagreement rate (calibration)", fontsize=FS, color=INK2, labelpad=2)
cax = fig.add_axes([0.918, 0.28, 0.013, 0.565])
cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1.0])
cb.ax.set_yticklabels(["0", "0.5", "1"]); cb.ax.tick_params(length=2, pad=1.5, width=0.5, labelsize=FS, labelcolor=INK2)
cb.solids.set_edgecolor("face"); cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(AXIS)
cb.set_label("P(certify)", fontsize=FS, color=INK2, labelpad=2)
H = [Line2D([], [], marker="o", ls="none", color=BLUE, ms=4.6, label="Text"),
     Line2D([], [], marker="^", ls="none", color=ORANGE, ms=4.6, label="C2C"),
     Line2D([], [], marker="o", ls="none", color=INK2, ms=4.6, label="filled: deployed"),
     Line2D([], [], marker="o", ls="none", color=INK2, mfc="white", ms=4.6, label="open: fallback"),
     Line2D([], [], color=INK, lw=1.0, label="P(certify) = 0.5")]
fig.legend(handles=H, loc="lower center", ncol=5, frameon=False, fontsize=FS, bbox_to_anchor=(0.5, 0.0),
           handletextpad=0.35, columnspacing=1.1, handlelength=1.5)
fig.savefig(OUT, metadata={"CreationDate": None})
print("wrote", OUT)

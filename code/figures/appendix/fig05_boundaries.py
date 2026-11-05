"""Run from any directory: python3 code/figures/appendix/fig05_boundaries.py [output.pdf]

Figure 3 (Appendix C, fig:boundaries): calibration and development change rates at the 20
predetermined quantiles, for the large and small pairs on OBQA and ARC.

Replaces the earlier code/figures/figure3_reference_boundaries.py, with every
text element at 8 pt. Inputs:
  results/boundaries_large_small/summary/calibration_all_160.csv
  results/boundaries_large_small/summary/risk_coverage_curves.csv   (split == dev)
  configs/confidence_boundaries/deployments/all.json                (deployed q; 0 = fallback)
Checks: every Clopper--Pearson bound and p-value is recomputed from its counts;
the accepted flag equals p <= 0.001; the deployed q is the largest accepted
grid point; and the deployed and fallback rows agree with Tables 4 and 5 of
the appendix (tab:dev_full and tab:calstats, new_app_D.tex).
"""
import csv, json, re, sys, os
from scipy.stats import beta, binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
HERE = Path(__file__).resolve().parent            # code/figures/appendix
ROOT = HERE.parents[2]                             # repository root
TEX = HERE / "paper_tex"                           # appendix LaTeX of the submitted paper
REC = HERE / "record_tables"                       # fuller tables that the appendix condenses
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "figure_boundaries.pdf"
D = open(TEX / "new_app_D.tex").read()
cal_all = list(csv.DictReader(open(ROOT / "results/boundaries_large_small/summary/calibration_all_160.csv")))
dev_all = [r for r in csv.DictReader(open(ROOT / "results/boundaries_large_small/summary/risk_coverage_curves.csv")) if r["split"] == "dev"]
dep = json.load(open(ROOT / "configs/confidence_boundaries/deployments/all.json"))
REF = {"T": "Text", "C": "C2C"}
DS = {"obqa": "OBQA", "arc": "ARC"}

def table_rows(label):
    t = D[D.index("\\label{%s}" % label):D.index("\\end{table}", D.index("\\label{%s}" % label))]
    return [[c.strip() for c in l.split("&")] for l in t.split("\\\\") if l.count("&") >= 6]

t7 = {(r[0].split()[-1], r[1], re.sub(r"\$.*?\$", "", r[2])): r for r in table_rows("tab:dev_full")}
t8 = {(r[0].split()[-1], r[1], re.sub(r"\$.*?\$", "", r[2])): r for r in table_rows("tab:calstats")}

curves = {}
for pair in ("large", "small"):
    for ds in ("obqa", "arc"):
        for b in ("T", "C"):
            cal = [r for r in cal_all if (r["pair"], r["dataset"], r["reference"]) == (pair, ds, b)]
            dev = [r for r in dev_all if (r["pair"], r["dataset"], r["reference"]) == (pair, ds, b)]
            assert len(cal) == 20 and len(dev) == 20, (pair, ds, b)
            q = dep[f"{pair}_{ds}_{b}"]["q"]
            for r in cal:
                n, k = int(r["n_R"]), int(r["changed"])
                p = binom.cdf(k, n, 0.05)
                ub = beta.ppf(0.999, k + 1, n - k) if k < n else 1.0
                assert abs(p - float(r["p_value"])) <= 1e-9 + 1e-6 * p, (pair, ds, b, r["q"])
                assert abs(ub - float(r["CP_upper_0_999"])) < 1e-9, (pair, ds, b, r["q"])
                assert (r["accepted"] == "True") == (p <= 0.001) == (ub < 0.05), (pair, ds, b, r["q"])
            acc = [float(r["q"]) for r in cal if r["accepted"] == "True"]
            assert q == (max(acc) if acc else 0.0), (pair, ds, b)
            # Table 4 (tab:dev_full): deployed q, coverage, changed/skipped, smallest calibration bound
            row = t7[(pair, DS[ds], REF[b])]
            assert (0.0 if row[6] == "fallback" else float(row[6])) == q, (pair, ds, b, row[6])
            assert f"{min(float(r['CP_upper_0_999']) for r in cal):.3f}" == row[5], (pair, ds, b, row[5])
            if q:
                d = next(r for r in dev if float(r["q"]) == q)
                assert row[8] == f"{d['changed']}/{d['n_R']}" and row[7] == f"{100 * float(d['coverage']):.1f}", row
                c = next(r for r in cal if float(r["q"]) == q)
            else:
                assert row[8] == "--"
                c = min(cal, key=lambda r: (float(r["p_value"]), float(r["q"])))
            # Table 5 (tab:calstats): calibration n and k at the deployed (or best fallback) grid point
            r8 = t8[(pair, DS[ds], REF[b])]
            assert (int(r8[5]), int(r8[6])) == (int(c["n_R"]), int(c["changed"])), (pair, ds, b, r8[5:7])
            curves[(pair, ds, b)] = (q, cal, dev)
# fixed R (q = 1) fails the test in all eight settings; the two large ARC rates are below 5%
fixedR = {k: next(r for r in v[1] if float(r["q"]) == 1.0) for k, v in curves.items()}
assert all(r["accepted"] == "False" for r in fixedR.values())
assert [(fixedR[("large", "arc", b)]["changed"], fixedR[("large", "arc", b)]["n_R"]) for b in "TC"] == [("12", "448"), ("21", "448")]
print("checks passed")

# ---------------- figure ----------------
from matplotlib.legend_handler import HandlerTuple
FS, FS_H = 8.0, 8.5
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
COL = {"T": "#2a78d6", "C": "#eb6834"}
MARK = {"T": "o", "C": "^"}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#ebeae5", "#c3c2b7"
PASS_BAND, PASS_LINE, PASS_INK = "#e3f4ea", "#7cc49a", "#3d8f5f"   # the certified region of Figure 2
YMAX = {"large": 15, "small": 70}
YT = {"large": [0, 5, 10, 15], "small": [0, 20, 40, 60]}

def uniq(xs, ys, flags):
    seen, out = set(), []
    for x, y, f in zip(xs, ys, flags):
        k = (round(x, 6), round(y, 6))
        if k not in seen:
            seen.add(k); out.append((x, y, f))
    return out

fig, axs = plt.subplots(1, 4, figsize=(5.5, 2.5))
fig.subplots_adjust(left=0.072, right=0.992, bottom=0.18, top=0.74, wspace=0.3)
for ax, (pair, ds) in zip(axs, [("large", "obqa"), ("large", "arc"), ("small", "obqa"), ("small", "arc")]):
    ax.axhspan(0, 5, color=PASS_BAND, lw=0, zorder=0)
    ax.axhline(5, color=PASS_LINE, lw=0.9, ls=(0, (3, 2)), zorder=1)
    for v in YT[pair][1:]:
        if v != 5:
            ax.axhline(v, color=GRID, lw=0.6, zorder=0)
    for b in ("T", "C"):
        q, cal, dev = curves[(pair, ds, b)]
        cx = [100 * float(r["coverage"]) for r in cal]; cy = [100 * float(r["CP_upper_0_999"]) for r in cal]
        dx = [100 * float(r["coverage"]) for r in dev]; dy = [100 * float(r["conditional_risk"]) for r in dev]
        assert max(cy + dy) <= YMAX[pair]
        ax.plot(dx, dy, color=COL[b], lw=0.9, ls=(0, (2.6, 1.6)), alpha=0.85, zorder=2, solid_capstyle="round")
        ax.plot(cx, cy, color=COL[b], lw=1.25, zorder=3, solid_joinstyle="round")
        for x, y, ok in uniq(cx, cy, [r["accepted"] == "True" for r in cal]):
            ax.plot([x], [y], marker=MARK[b], ms=3.4 if b == "T" else 3.8, mfc=COL[b] if ok else "white",
                    mec=COL[b], mew=0.8, ls="none", zorder=4)
        if q:
            ci = next(i for i, r in enumerate(cal) if float(r["q"]) == q)
            di = next(i for i, r in enumerate(dev) if float(r["q"]) == q)
            ax.plot([dx[di]], [dy[di]], marker="o", ms=6.0, mfc="white", mec=COL[b], mew=1.0, ls="none", zorder=5)
            ax.plot([cx[ci]], [cy[ci]], marker="*", ms=10.5, mfc=COL[b], mec="white", mew=0.7, ls="none", zorder=6)
    ax.set_xlim(0, 103); ax.set_ylim(0, YMAX[pair]); ax.set_yticks(YT[pair])
    ax.set_xticks([0, 50, 100])
    ax.tick_params(length=2.5, width=0.5, color=AXIS, labelcolor=INK2, labelsize=FS, pad=2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(AXIS)
    ax.set_title(f"{pair}/{DS[ds]}", fontsize=FS_H, color=INK, pad=4, loc="left")
    if pair == "small":
        ax.text(0.97, 0.15, "no threshold\npasses", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=FS, color=INK2, linespacing=1.05)
axs[0].text(2, 4.6, "5% target", ha="left", va="top", fontsize=FS, color=PASS_INK)
axs[0].set_ylabel("Change rate among skipped (%)", fontsize=FS, color=INK2, labelpad=2)
fig.text(0.532, 0.035, "Coverage: questions skipped (%)", ha="center", va="center", fontsize=FS, color=INK2)

GR = "#6d6c67"
hand = [Line2D([], [], color=COL["T"], lw=1.25, marker="o", ms=3.4, label="Text"),
        Line2D([], [], color=COL["C"], lw=1.25, marker="^", ms=3.8, label="C2C"),
        Line2D([], [], color=GR, lw=1.25, label="calibration: 0.999 upper bound"),
        Line2D([], [], color=GR, lw=0.9, ls=(0, (2.6, 1.6)), label="development: observed rate"),
        (Line2D([], [], color=GR, lw=0, marker="o", ms=3.8), Line2D([], [], color=GR, lw=0, marker="o", ms=3.8, mfc="white")),
        (Line2D([], [], color=GR, lw=0, marker="*", ms=9.5, mec="white", mew=0.6),
         Line2D([], [], color=GR, lw=0, marker="o", ms=5.6, mfc="white", mew=1.0))]
labels = ["Text", "C2C", "calibration: 0.999 upper bound", "development: observed rate",
          "filled / open: grid point passes / fails", "deployed threshold (calibration / development)"]
# The legend spans exactly the four panels: its first column starts at the left edge of the first
# panel, its last column ends at the right edge of the last panel, and the space left over is shared
# equally between the columns (mode="expand"); the rows sit where they did before.
x_left, x_right = axs[0].get_position().x0, axs[-1].get_position().x1
pad = (0.5 + 0.4) * FS / 72 / fig.get_figwidth()      # borderaxespad + borderpad, in figure width
fig.legend(hand, labels, loc="upper left", ncol=3, frameon=False, fontsize=FS, mode="expand",
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
           handlelength=2.0, columnspacing=1.3, handletextpad=0.5, labelspacing=0.4,
           bbox_to_anchor=(x_left - pad, 0.0, x_right - x_left + 2 * pad, 1.0))
fig.savefig(OUT, metadata={"CreationDate": None})
print("wrote", OUT)

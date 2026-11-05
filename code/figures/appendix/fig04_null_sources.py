"""Run from any directory: python3 code/figures/appendix/fig04_null_sources.py [output.pdf]

Figure 8 (Appendix F, fig:null_sources): where the oracle headroom comes from and how the two
references' changes relate.

(a) Questions only one of R, Text, C2C answers correctly (% of the population).
(b) Questions both references change, over what independent changes at the
    observed rates would give.
(c) Share of the answers each reference changes whose receiver uncertainty u is
    at or below the fit-split median, next to the share of all questions.

Counts come from the saved development outputs: (a) from the correctness-pattern
table in record_tables/patterns.tex, (b) and (c) from results/e09a/results/
(e9a_overlap.csv, e9a_receiver_confidence.csv); every count
is checked against Table 1 of the main text (correct answers, oracle headroom)
and against the numbers Appendix F states.
"""
import re, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import Patch

from pathlib import Path
HERE = Path(__file__).resolve().parent            # code/figures/appendix
ROOT = HERE.parents[2]                             # repository root
TEX = HERE / "paper_tex"                           # appendix LaTeX of the submitted paper
REC = HERE / "record_tables"                       # fuller tables that the appendix condenses
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "figure_null_sources.pdf"
T1 = open(TEX / "tab_main_complementarity.tex").read()
AC = open(TEX / "new_app_C.tex").read()
PAT = open(REC / "patterns.tex").read()
E9A = ROOT / "results" / "e09a" / "results"
POPS = ["small/OBQA", "small/ARC", "medium/OBQA", "medium/ARC", "large/OBQA", "large/ARC", "large/MMLU-Pro"]

pat = {}
for m in re.finditer(r"(small|medium|large)/(OBQA|ARC|MMLU-Pro) & (\d+) & " + " & ".join([r"(\d+)"] * 8), PAT):
    pop = m.group(1) + "/" + m.group(2); v = [int(x) for x in m.groups()[2:]]
    pat[pop] = dict(N=v[0], **dict(zip(["000", "100", "010", "001", "110", "101", "011", "111"], v[1:])))
assert set(pat) == set(POPS)
import csv
ov = {}
for r in csv.DictReader(open(E9A / "e9a_overlap.csv")):             # (b): overlap of the two references' changes
    a, b, c, N = (int(r[k]) for k in ("a_Text_only", "b_C2C_only", "c_both", "N"))
    ratio = c / ((a + c) * (b + c) / N)                                # recomputed from the counts
    assert abs(ratio - float(r["ratio_c_over_expected"])) < 1e-3, r["population"]
    ov[r["population"]] = dict(a=a, b=b, c=c, exp=float(r["expected_both_if_independent"]), ratio=round(ratio, 2))
conf = {}
for r in csv.DictReader(open(E9A / "e9a_receiver_confidence.csv")): # (c): u of the changed answers
    pop, ref = r["setting"].rsplit("/", 1)
    conf.setdefault(pop, {})[ref] = round(100 * float(r["share_u_le_fit_median_changed"]), 1)
    conf[pop]["all"] = round(100 * float(r["share_u_le_fit_median_all_dev"]), 1)
assert set(ov) == set(POPS) and set(conf) == set(POPS)

# ---- checks against Table 1 and Appendix F ----
t1key = {"small/OBQA": "small & OBQA", "small/ARC": " & ARC (299) & 110", "medium/OBQA": "medium & OBQA",
         "medium/ARC": " & ARC (299) & 217", "large/OBQA": "large & OBQA", "large/ARC": " & ARC (299) & 268",
         "large/MMLU-Pro": " & MMLU-Pro"}
for pop in POPS:
    p = pat[pop]; N = p["N"]
    assert sum(p[k] for k in p if k != "N") == N, pop
    R = p["100"] + p["110"] + p["101"] + p["111"]; T = p["010"] + p["110"] + p["011"] + p["111"]
    C = p["001"] + p["101"] + p["011"] + p["111"]; orc = N - p["000"]
    line = [l for l in T1.split("\\\\") if t1key[pop] in l][0]
    cells = [re.sub(r"\$.*?\$", "", c).strip() for c in line.split("&")]
    assert [int(cells[2]), int(cells[3]), int(cells[4])] == [R, T, C], (pop, cells[2:5], (R, T, C))
    assert abs(float(cells[5]) - round(100 * (orc - max(R, T, C)) / N, 2)) < 1e-9, pop
    o = ov[pop]; ta, ca = T - 0, 0
    assert abs(o["c"] / o["exp"] - o["ratio"]) / o["ratio"] < 0.01, pop   # expected is printed rounded
assert min(v["ratio"] for v in ov.values()) == 1.29 and max(v["ratio"] for v in ov.values()) == 6.64
assert "1.3 to 6.6 times" in AC
assert (conf["medium/OBQA"]["Text"], conf["medium/ARC"]["Text"], conf["medium/OBQA"]["C2C"], conf["medium/ARC"]["C2C"]) == (14.9, 26.9, 5.3, 2.1)
assert "14.9 and 26.9\\% of Text's changes" in AC and "5.3 and 2.1\\% of C2C's" in AC
print("checks passed")

# ---------------- figure ----------------
FS, FS_H = 8.0, 8.5
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.85), sharey=True, gridspec_kw=dict(width_ratios=[1.2, 0.8, 1.05]))
fig.subplots_adjust(left=0.165, right=0.975, top=0.86, bottom=0.235, wspace=0.16)
ys = list(range(len(POPS)))

ax = axes[0]
for i, pop in enumerate(POPS):
    p = pat[pop]; N = p["N"]; left = 0
    for key, col in (("100", GRAY), ("010", BLUE), ("001", ORANGE)):
        w = 100 * p[key] / N
        ax.barh(i, w, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.0, zorder=2)
        left += w
ax.set_xlim(0, 27); ax.set_xticks([0, 10, 20]); ax.set_xticklabels(["0%", "10%", "20%"])
ax.set_title("(a) Only one path right", fontsize=FS_H, pad=4, loc="left")

ax = axes[1]
for i, pop in enumerate(POPS):
    ax.barh(i, ov[pop]["ratio"], height=0.62, color=INK2, zorder=2)
    ax.text(ov[pop]["ratio"] + 0.12, i, f"{ov[pop]['ratio']:.1f}", va="center", fontsize=FS, color=INK2)
ax.axvline(1, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
ax.text(1.18, -0.78, "1 = independent", ha="left", va="center", fontsize=FS, color=INK)
ax.set_xlim(0, 8); ax.set_xticks([0, 1, 4, 8]); ax.set_xticklabels(["0", "1", "4", "8"])
ax.set_title("(b) Joint changes", fontsize=FS_H, pad=4, loc="left")

ax = axes[2]
for i, pop in enumerate(POPS):
    ax.plot([conf[pop]["all"]], [i], marker="|", ms=9, mew=1.6, color=INK, ls="none", zorder=2)
    ax.plot([conf[pop]["Text"]], [i - 0.12], marker="o", ms=5.0, color=BLUE, ls="none", zorder=3)
    ax.plot([conf[pop]["C2C"]], [i + 0.12], marker="^", ms=5.0, color=ORANGE, ls="none", zorder=3)
ax.set_xlim(-3, 80); ax.set_xticks([0, 25, 50, 75]); ax.set_xticklabels(["0%", "25%", "50%", "75%"])
ax.set_title("(c) Changed but confident", fontsize=FS_H, pad=4, loc="left")

for ax in axes:
    for v in ax.get_xticks():
        ax.axvline(v, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="x", length=2.5, width=0.5, color=AXIS, labelcolor=INK2, labelsize=FS)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
axes[0].set_yticks(ys); axes[0].set_yticklabels(POPS, fontsize=FS, color=INK)
axes[0].set_ylim(len(POPS) - 0.5, -1.05)
axes[0].set_xlabel("share of questions", fontsize=FS, color=INK2, labelpad=2)
axes[1].set_xlabel("observed / independent", fontsize=FS, color=INK2, labelpad=2)
axes[2].set_xlabel(r"share with $u\leq$ fit median", fontsize=FS, color=INK2, labelpad=2)

hand = [Patch(color=GRAY, label="receiver alone"), Patch(color=BLUE, label="Text"), Patch(color=ORANGE, label="C2C"),
        Line2D([], [], color=BLUE, marker="o", ms=5, lw=0, label="Text's changes"),
        Line2D([], [], color=ORANGE, marker="^", ms=5, lw=0, label="C2C's changes"),
        Line2D([], [], color=INK, marker="|", ms=9, mew=1.6, lw=0, label="all questions")]


class MarkerOnly(HandlerLine2D):
    """Legend handle exactly as wide as the marker, so each marker sits as close to its label (0.4 em)
    and as far from the previous label (1.0 em) as the colour patches do."""
    def __init__(self, width):
        super().__init__()
        self.w = width
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        handlebox.width = self.w
        return super().legend_artist(legend, orig_handle, fontsize, handlebox)


fig.legend(handles=hand, loc="lower center", ncol=6, frameon=False, fontsize=FS, handlelength=1.2,
           columnspacing=1.0, handletextpad=0.4, bbox_to_anchor=(0.55, 0.0),
           handler_map={hand[3]: MarkerOnly(5.0), hand[4]: MarkerOnly(5.0), hand[5]: MarkerOnly(1.6)})
fig.savefig(OUT)
print("wrote", OUT)

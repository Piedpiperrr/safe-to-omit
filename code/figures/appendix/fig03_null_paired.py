"""Run from any directory: python3 code/figures/appendix/fig03_null_paired.py [output.pdf]

Figure 7 (Appendix F, fig:null_paired): each real path against the content-free
null, matched one path at a time, under both matching designs.

Two stacked panels (top: matched in total; bottom: matched separately on
questions the receiver answers correctly and incorrectly). One slot per
population, grouped by pair; blue = Text, orange = C2C; bar = null/real gain
ratio (mean of 1,000 draws) drawn from 1 on a log axis; whisker = 95%
question-bootstrap interval; filled = interval excludes 1, outlined = it
includes 1. The stratified large/ARC Text ratio is a lower bound.

Numbers are parsed from Table 14 (tab:e22) in new_app_C.tex and checked against
Table 13(a) (tab:null_matched) and Table 1 of the main text.
"""
import re, sys, os
from decimal import Decimal, ROUND_HALF_UP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

from pathlib import Path
HERE = Path(__file__).resolve().parent            # code/figures/appendix
ROOT = HERE.parents[2]                             # repository root
TEX = HERE / "paper_tex"                           # appendix LaTeX of the submitted paper
REC = HERE / "record_tables"                       # fuller tables that the appendix condenses
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "figure_null_paired.pdf"
C = open(TEX / "new_app_C.tex").read()
T1 = open(TEX / "tab_main_complementarity.tex").read()

t5 = C[C.index("\\label{tab:e22}"):C.index("\\end{table}", C.index("\\label{tab:e22}"))]
num = r"([0-9.]+)(\$\^\{\\S\}\$)?\s*\[([0-9.]+),\s*([0-9.]+)\]"
rows, undet = {}, set()
for line in t5.split("\\\\"):
    m = re.search(r"(small|medium|large) & (OBQA|ARC|MMLU-Pro) & (Text|C2C) &.*?& " + num + r" & " + num, line)
    if m:
        pop = m.group(1) + "/" + m.group(2); g = m.groups()
        rows[(pop, g[2])] = {"strat": [float(g[3]), float(g[5]), float(g[6])],
                             "unstrat": [float(g[7]), float(g[9]), float(g[10])]}
        if g[4]:
            undet.add((pop, g[2]))
POPS = ["small/OBQA", "small/ARC", "medium/OBQA", "medium/ARC", "large/OBQA", "large/ARC", "large/MMLU-Pro"]
assert len(rows) == 14 and undet == {("large/ARC", "Text")}, (len(rows), undet)

t4 = C[C.index("\\textit{(a) Gain over"):C.index("\\textit{(b)")]
numb = r"([0-9.]+)\s*\[([0-9.]+),\s*([0-9.]+)\]"
seen = 0
for line in t4.split("\\\\"):
    m = re.search(r"(small|medium|large) & (OBQA|ARC|MMLU-Pro)[^&]*& \d+ & \d+ & " + numb + " & " + numb + " & " + numb, line)
    if m:
        pop = m.group(1) + "/" + m.group(2); g = [float(v) for v in m.groups()[2:]]
        assert rows[(pop, "Text")]["unstrat"] == g[3:6] and rows[(pop, "C2C")]["unstrat"] == g[6:9], pop
        seen += 1
assert seen == 7
def r2(x): return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
keys = {"small/OBQA": "small & OBQA", "small/ARC": " & ARC (299) & 110", "medium/OBQA": "medium & OBQA",
        "medium/ARC": " & ARC (299) & 217", "large/OBQA": "large & OBQA", "large/ARC": " & ARC (299) & 268",
        "large/MMLU-Pro": " & MMLU-Pro"}
for pop, key in keys.items():
    line = [l for l in T1.split("\\\\") if key in l][0]
    cells = [re.sub(r"\$.*?\$", "", c).strip() for c in line.split("&")]
    tu, cu, ts, cs = [float(x) for x in cells[-4:]]
    for ref, d, v in (("Text", "unstrat", tu), ("C2C", "unstrat", cu), ("Text", "strat", ts), ("C2C", "strat", cs)):
        assert r2(rows[(pop, ref)][d][0]) == v, (pop, ref, d, v)
for pop in POPS[2:]:
    for d in ("unstrat", "strat"):
        assert rows[(pop, "C2C")][d][2] > 1, (pop, d)
sig = {d: sum(rows[(p, "Text")][d][2] < 1 for p in POPS) for d in ("unstrat", "strat")}
assert sig == {"unstrat": 5, "strat": 2}, sig
print("checks passed; Text significant:", sig)

# ---------------- figure ----------------
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator


# ---------------- shared style ----------------
FS, FS_H, FS_S = 8.0, 8.5, 7.5
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e"
GRID, AXIS = "#ebeae5", "#c3c2b7"
COL = {"Text": BLUE, "C2C": ORANGE}
YLIM = (0.22, 3.9)
TICKS = [0.25, 0.5, 1.0, 2.0, 3.0]
BAR_W, OFF, GAPG = 0.34, 0.19, 0.55          # bar width, Text/C2C offset from the slot, extra gap between pairs
LW_BASE, LW_WHISK, CAPSIZE, LW_EDGE, LW_GRID = 0.8, 0.7, 3.0, 1.0, 0.55
DESIGNS = (("unstrat", "Matched in total"), ("strat", "Matched on right and wrong answers separately"))


def significant(lo, hi):
    return hi < 1 or lo > 1


# ---------------- layout (inches) ----------------
W, H = 5.5, 3.30
L_IN, R_IN = 0.44, 0.05        # left: y label + tick labels; right margin
TOP_IN = 0.40                  # legend row + top panel title
GAP_IN = 0.23                  # between the panels: bottom panel title
BOT_IN = 0.34                  # x tick labels + pair brackets
PW = W - L_IN - R_IN
PH = (H - TOP_IN - GAP_IN - BOT_IN) / 2
fig = plt.figure(figsize=(W, H), dpi=72)      # dpi 72: display units are points (for the clearance checks)
axT = fig.add_axes([L_IN / W, (BOT_IN + PH + GAP_IN) / H, PW / W, PH / H])
axB = fig.add_axes([L_IN / W, BOT_IN / H, PW / W, PH / H])
AXES = {"unstrat": axT, "strat": axB}

# x slots, grouped by pair (the scheme of fig06_baselines.py panel (a))
GROUPS = [("small", ["OBQA", "ARC"]), ("medium", ["OBQA", "ARC"]), ("large", ["OBQA", "ARC", "MMLU-Pro"])]
xs, x = {}, 0.0
for gi, (pair, bms) in enumerate(GROUPS):
    if gi:
        x += GAPG
    for bm in bms:
        xs[f"{pair}/{bm}"] = x; x += 1.0
assert list(xs) == POPS
XLIM = (-0.55, x - 0.45)
KX = PW * 72 / (XLIM[1] - XLIM[0])                 # points per x unit
KY = PH * 72 / np.log10(YLIM[1] / YLIM[0])         # points per decade of y
DX = {"Text": -OFF, "C2C": OFF}


def bar(ax, xc, p, col, filled):
    """Bar from 1 to p. Outlined bars get their stroke inset by half its width, so every bar's outer edge
    is exactly [xc - BAR_W/2, xc + BAR_W/2] x [min(p, 1), max(p, 1)], filled or not."""
    y0, y1 = min(p, 1.0), max(p, 1.0)
    if filled:
        ax.add_patch(Rectangle((xc - BAR_W / 2, y0), BAR_W, y1 - y0, facecolor=col, edgecolor="none", zorder=2))
        return
    h = LW_EDGE / 2
    dx, f = h / KX, 10 ** (h / KY)
    yy0, yy1 = y0 * f, y1 / f
    if yy1 < yy0:                                   # bar shorter than its stroke: a flat stroke at the midpoint
        yy0 = yy1 = np.sqrt(y0 * y1)
    ax.add_patch(Rectangle((xc - BAR_W / 2 + dx, yy0), BAR_W - 2 * dx, yy1 - yy0, facecolor="white",
                           edgecolor=col, lw=LW_EDGE, joinstyle="miter", zorder=2))


TITLES = {}
for d, title in DESIGNS:
    ax = AXES[d]
    ax.set_yscale("log"); ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    for v in TICKS:
        if v != 1:
            ax.axhline(v, color=GRID, lw=LW_GRID, zorder=0)
    ax.axhline(1, color=INK2, lw=LW_BASE, zorder=3)
    for pop in POPS:
        for ref in ("Text", "C2C"):
            p, lo, hi = rows[(pop, ref)][d]
            xc = xs[pop] + DX[ref]
            bar(ax, xc, p, COL[ref], significant(lo, hi))
            ax.errorbar([xc], [p], yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK2, elinewidth=LW_WHISK,
                        capsize=CAPSIZE, capthick=LW_WHISK, zorder=4)
    TITLES[d] = ax.set_title(title, fontsize=FS_H, color=INK, loc="left", pad=3)
    ax.yaxis.set_major_locator(FixedLocator(TICKS))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in TICKS]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(list(xs.values()))
    ax.tick_params(axis="y", length=0, labelcolor=INK2, labelsize=FS, pad=3)
    ax.tick_params(axis="x", length=0, labelcolor=INK, labelsize=FS, pad=2)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
axT.tick_params(axis="x", labelbottom=False)
axB.set_xticklabels([p.split("/")[1] for p in xs])

# pair names under their benchmarks, with a thin bracket (fig06_baselines.py panel (a); its offsets of
# -0.20 and -0.23 axes heights are rescaled so they are the same distance in points below this panel)
A_PH = 0.225 * 3.95                                  # height of fig04's panel (a), in
for pair, bms in GROUPS:
    x0, x1 = xs[f"{pair}/{bms[0]}"] - 0.42, xs[f"{pair}/{bms[-1]}"] + 0.42
    yb = -0.20 * A_PH / PH
    axB.plot([x0, x1], [yb, yb], color=AXIS, lw=0.6, transform=axB.get_xaxis_transform(), clip_on=False)
    axB.text((x0 + x1) / 2, yb - 0.03 * A_PH / PH, f"{pair} pair", ha="center", va="top", fontsize=FS,
             color=INK2, transform=axB.get_xaxis_transform())

# y label once, centred on the two panels
YLAB = fig.text(0.03 / W, (BOT_IN + PH + GAP_IN / 2) / H, "null gain over the receiver / real gain (log scale)",
         rotation=90, ha="left", va="center", fontsize=FS, color=INK2)

# the one stratified ratio that is only a lower bound: a label just above its upper whisker cap, centred on
# the whisker unless that would bring it within LB_GAP points of the C2C bar to its right
fig.canvas.draw()
REND = fig.canvas.get_renderer()
LB_GAP, LB_LIFT = 2.5, 1.0                           # points
LB = []
for pop, ref in sorted(undet):
    p, lo, hi = rows[(pop, ref)]["strat"]
    xw = xs[pop] + DX[ref]
    y_lab = hi * 10 ** ((LW_WHISK / 2 + LB_LIFT) / KY)
    t = axB.text(xw, y_lab, "lower\nbound", ha="center", va="bottom", fontsize=FS_S, color=INK2,
                 linespacing=1.0, zorder=5)
    w = t.get_window_extent(REND).width / KX          # label width, x units
    right_limit = xs[pop] + DX["C2C"] - BAR_W / 2 - LB_GAP / KX
    xl = min(xw, right_limit - w / 2)
    assert xl - w / 2 < xw - CAPSIZE / KX and xl + w / 2 > xw + CAPSIZE / KX, "label no longer covers its cap"
    t.set_x(xl); LB.append(t)


# ---------------- legend (one row, top) ----------------
class Glyph(HandlerBase):
    """Legend handle drawn by a function f(x0, yc) -> artists (in points), `width` points wide."""
    def __init__(self, width, f):
        super().__init__(); self.width, self.f = width, f

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        handlebox.width = self.width
        arts = self.f(handlebox.xdescent, handlebox.height / 2 - handlebox.ydescent)
        for a in arts:
            a.set_transform(handlebox.get_transform()); handlebox.add_artist(a)
        return arts[0]


SW, SH = 6.0, 8.0                                     # legend swatch: a small upright bar


def g_fill(col):
    return lambda x, y: [Rectangle((x, y - SH / 2), SW, SH, facecolor=col, edgecolor="none")]


def g_outline(x, y):
    h = LW_EDGE / 2
    return [Rectangle((x + h, y - SH / 2 + h), SW - 2 * h, SH - 2 * h, facecolor="white", edgecolor=GRAY,
                      lw=LW_EDGE, joinstyle="miter")]


def g_whisker(x, y):
    xm, hh = x + CAPSIZE, 4.5
    return [Line2D([xm, xm], [y - hh, y + hh], color=INK2, lw=LW_WHISK),
            Line2D([xm - CAPSIZE, xm + CAPSIZE], [y + hh, y + hh], color=INK2, lw=LW_WHISK),
            Line2D([xm - CAPSIZE, xm + CAPSIZE], [y - hh, y - hh], color=INK2, lw=LW_WHISK)]


hs = [object() for _ in range(5)]
LEG_LABELS = ["Text", "C2C", "whisker: 95% question-bootstrap interval", "filled: excludes 1", "outlined: includes 1"]
leg = fig.legend(handles=hs, labels=LEG_LABELS, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                 ncol=5, frameon=False, fontsize=FS, handletextpad=0.45, columnspacing=1.1, borderaxespad=0.2,
                 handler_map={hs[0]: Glyph(SW, g_fill(BLUE)), hs[1]: Glyph(SW, g_fill(ORANGE)),
                              hs[2]: Glyph(2 * CAPSIZE, g_whisker), hs[3]: Glyph(SW, g_fill(GRAY)),
                              hs[4]: Glyph(SW, g_outline)})
for t in leg.get_texts():
    t.set_color(INK)


# ---------------- clearance checks (points) ----------------
def gap(a, b):
    """Distance between two boxes (x0, y0, x1, y1); 0 when they touch or overlap."""
    dx = max(0.0, b[0] - a[2], a[0] - b[2]); dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return float(np.hypot(dx, dy))


def bb(artist):
    e = artist.get_window_extent(REND); return (e.x0, e.y0, e.x1, e.y1)


fig.canvas.draw()
report = []
for d, _ in DESIGNS:
    ax = AXES[d]; T = ax.transData.transform; ab = bb(ax)
    marks = []                                        # (name, box) for every drawn mark in this panel
    for pop in POPS:
        for ref in ("Text", "C2C"):
            p, lo, hi = rows[(pop, ref)][d]; xc = xs[pop] + DX[ref]
            (bx0, by0), (bx1, by1) = T([(xc - BAR_W / 2, min(p, 1)), (xc + BAR_W / 2, max(p, 1))])
            marks.append((f"{pop} {ref} bar", (bx0, by0, bx1, by1)))
            (wx, wy0), (_, wy1) = T([(xc, lo), (xc, hi)]); hw = LW_WHISK / 2
            marks.append((f"{pop} {ref} whisker", (wx - hw, wy0 - hw, wx + hw, wy1 + hw)))
            for yy in (wy0, wy1):
                marks.append((f"{pop} {ref} cap", (wx - CAPSIZE, yy - hw, wx + CAPSIZE, yy + hw)))
    bars = [(n, b) for n, b in marks if n.endswith("bar")]
    # bars of different slots, and the two bars of one slot
    bar_gaps = sorted((gap(b1, b2), n1, n2) for i, (n1, b1) in enumerate(bars) for n2, b2 in bars[i + 1:])
    report.append(f"[{d}] smallest bar-to-bar gaps (pt): " +
                  "; ".join(f"{v:.2f} ({a} | {b})" for v, a, b in bar_gaps[:3]))
    assert bar_gaps[0][0] > 1.0
    # everything inside the axes box
    for n, b in marks:
        assert ab[0] <= b[0] and b[2] <= ab[2] and ab[1] <= b[1] and b[3] <= ab[3], (d, n)
    if d == "strat":
        base = ("baseline y=1", (ab[0], T((0, 1))[1] - LW_BASE / 2, ab[2], T((0, 1))[1] + LW_BASE / 2))
        grid = [(f"grid {v:g}", (ab[0], T((0, v))[1] - LW_GRID / 2, ab[2], T((0, v))[1] + LW_GRID / 2))
                for v in TICKS if v != 1]
        for t in LB:
            lb = bb(t)
            near = sorted((gap(lb, b), n) for n, b in marks + [base] + grid)
            report.append("lower-bound label: nearest " + "; ".join(f"{v:.2f} ({n})" for v, n in near[:4]) +
                          f"; to panel top {ab[3] - lb[3]:.1f}")
            assert near[0][0] >= 1.0 and ab[3] - lb[3] > 1.0
# titles, legend, tick labels, pair labels
lg = bb(leg); tT, tB = bb(TITLES["unstrat"]), bb(TITLES["strat"])
report.append(f"legend {lg[0]:.1f}..{lg[2]:.1f} pt of {W * 72:.0f}; legend bottom to top title: {lg[1] - tT[3]:.1f} pt")
report.append(f"bottom title top to top panel bottom: {bb(axT)[1] - tB[3]:.1f} pt")
xt = [bb(t) for t in axB.get_xticklabels()]
pl = [bb(t) for t in axB.texts if t.get_text().endswith("pair")]
report.append(f"tick-label gaps: min {min(gap(a, b) for i, a in enumerate(xt) for b in xt[i + 1:]):.1f} pt; "
              f"tick labels to pair labels: min {min(gap(a, b) for a in xt for b in pl):.1f} pt; "
              f"pair labels bottom at {min(b[1] for b in pl):.1f} pt")
yl = bb(YLAB); yt = [bb(t) for ax in AXES.values() for t in ax.get_yticklabels()]
report.append(f"y label spans {yl[1]:.1f}..{yl[3]:.1f} pt (panels {bb(axB)[1]:.1f}..{bb(axT)[3]:.1f}); "
              f"y label to tick labels: min {min(gap(yl, b) for b in yt):.1f} pt")
texts = [leg, YLAB, *TITLES.values(), *LB, *axB.texts, *axB.get_xticklabels(), *[t for ax in AXES.values()
         for t in ax.get_yticklabels()]]
edge = min(min(b[0], b[1], W * 72 - b[2], H * 72 - b[3]) for b in map(bb, texts))
report.append(f"closest text to a figure edge: {edge:.1f} pt")
print("\n".join(report))
assert lg[0] >= 0 and lg[2] <= W * 72 and lg[1] - tT[3] > 2.0 and edge > 0.5
assert min(gap(yl, b) for b in yt) > 2.0 and bb(axB)[1] < yl[1] and yl[3] < bb(axT)[3]

fig.savefig(OUT, metadata={"CreationDate": None})
print("wrote", OUT, f"size {W} x {H} in; panels {PW:.2f} x {PH:.2f} in; {KX:.1f} pt per slot, {KY:.1f} pt per decade")


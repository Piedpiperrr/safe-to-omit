"""Run from any directory: python3 code/figures/appendix/fig10_zero_u.py [output.pdf]

Figure 11 (Appendix J, fig:zero_u): what the certificates rest on. One column
per setting in both panels, named once under (b).

(a) Change rate among skipped development questions with exactly zero receiver
    uncertainty (u = 0; light bars) and among the others (u > 0; dark bars),
    with two-sided Clopper--Pearson 95% intervals as whiskers, for every
    certified Qwen-receiver threshold and the q = 0.50 sensitivity threshold
    (Table 26, tab:zero_u); square-root axis.
(b) Share of the replayed saving on skipped questions that comes from u > 0
    questions, as 100% stacked bars (Table 29, tab:u0saving).

Checks: counts and intervals are recomputed from Table 26; shares are
recomputed from the u = 0 and u > 0 parts of Table 29; the ranges stated in
Appendix J (0 to 2.2%, 3.8 to 20.0%, 7.3 to 41.3%, lower bound above 5% for
large/ARC/Text and both MMLU-Pro policies) are asserted.
"""
import re, sys, os
from scipy.stats import beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pathlib import Path
HERE = Path(__file__).resolve().parent            # code/figures/appendix
ROOT = HERE.parents[2]                             # repository root
TEX = HERE / "paper_tex"                           # appendix LaTeX of the submitted paper
REC = HERE / "record_tables"                       # fuller tables that the appendix condenses
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "figure_zero_u.pdf"
I = open(TEX / "new_app_I.tex").read()
def tab(label):
    i = I.index("\\label{%s}" % label); return I[i:I.index("\\end{table}", i)]
def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return 100 * lo, 100 * hi

z = {}
for m in re.finditer(r"([\w-]+/[\w-]+/[\w+]+) & ([0-9.]+) & (\d+) & ([0-9.]+) & (\d+)/(\d+) & ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] & (\d+)/(\d+) & ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] & ([0-9.]+)", tab("tab:zero_u")):
    s = m.group(1) + ("" if m.group(2) != "0.50" else " ($q=0.50$)")
    k0, n0, k1, n1 = int(m.group(5)), int(m.group(6)), int(m.group(10)), int(m.group(11))
    assert n0 + n1 == int(m.group(3)), s
    for k, n, r, lo, hi in ((k0, n0, 7, 8, 9), (k1, n1, 12, 13, 14)):
        rr = [float(m.group(x)) for x in (r, lo, hi)]
        c = cp(k, n)
        assert abs(100 * k / n - rr[0]) < 0.006 and abs(c[0] - rr[1]) < 0.006 and abs(c[1] - rr[2]) < 0.006, (s, rr, c)
    z[s] = dict(k0=k0, n0=n0, k1=k1, n1=n1, r0=100 * k0 / n0, c0=cp(k0, n0), r1=100 * k1 / n1, c1=cp(k1, n1))
assert len(z) == 10, len(z)
nine = [s for s in z if "q=0.50" not in s]          # the nine certified thresholds (q=0.50 is a sensitivity row)
assert len(nine) == 9
assert (f"{min(z[s]['r0'] for s in nine):.1f}", f"{max(z[s]['r0'] for s in nine):.1f}") == ("0.0", "2.2")
assert (f"{min(z[s]['r1'] for s in nine):.1f}", f"{max(z[s]['r1'] for s in nine):.1f}") == ("3.8", "20.0")
lb = sorted(s for s in z if z[s]["c1"][0] > 5)
assert lb == ["large/ARC/Text", "large/MMLU-Pro/C2C", "large/MMLU-Pro/Text"], lb

sav = {}
for m in re.finditer(r"([\w-]+/[\w-]+/[\w+]+)(\$\^\\ast\$)? & (-?[0-9.]+) & (\d+) & (-?[0-9.]+) & (\d+) & \$?(-?[0-9.]+)\$? & (\d+) & [^&]+ & ([0-9.]+) &", tab("tab:u0saving")):
    s = m.group(1) + (" (all)" if m.group(2) else "")
    a0, a1 = float(m.group(3)), float(m.group(5))
    share = a1 / (a0 + a1)
    assert abs(share - float(m.group(9))) < 0.002, (s, share, m.group(9))   # parts are printed rounded
    sav[s] = share
assert len(sav) == 12, len(sav)
ex = [s for s in sav if s not in ("large/MMLU-Pro/C2C", "large/MMLU-Pro/C2C (all)") and not s.startswith("Llama")]
assert (f"{100 * min(sav[s] for s in ex):.1f}", f"{100 * max(sav[s] for s in ex):.1f}") == ("7.3", "41.3"), ex
print("checks passed")

# ---------------- figure ----------------
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, ScaledTranslation, blended_transform_factory


# ---------------- slots: one per setting, grouped by pair ----------------
GROUPS = [("large", ["large/OBQA/Text", "large/OBQA/C2C", "large/ARC/Text", "large/ARC/C2C",
                     "large/MMLU-Pro/Text", "large/MMLU-Pro/C2C", "large/OBQA/Text+fact"]),
          ("medium", ["medium/OBQA/C2C", "medium/OBQA/C2C ($q=0.50$)", "medium/ARC/C2C"])]
SLOTS = [s for _, ks in GROUPS for s in ks]
assert len(SLOTS) == len(set(SLOTS)) == 10 and set(SLOTS) == set(z), sorted(set(z) ^ set(SLOTS))
# (b) draws the settings of (a) that were replayed (present in sav); the q = 0.50 sensitivity row was not
REPLAYED = [s for s in SLOTS if s in sav]
NOTREP = [s for s in SLOTS if s not in sav]
assert len(REPLAYED) == 9
assert NOTREP == ["medium/OBQA/C2C ($q=0.50$)"], NOTREP
assert "large/MMLU-Pro/C2C (all)" in sav and "large/MMLU-Pro/C2C" in REPLAYED   # the certified row, not "(all)"


def split(s):
    """'medium/OBQA/C2C ($q=0.50$)' -> ('medium', 'OBQA', 'C2C', 'C2C, $q=0.50$')"""
    pair, bm, rest = s.split("/", 2)
    ref = rest.split(" ")[0]
    note = re.search(r"\((.*)\)", rest)
    return pair, bm, ref, ref + (", " + note.group(1) if note else "")


# ---------------- style ----------------
FS, FS_H, FS_T = 8.0, 8.5, 7.5
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#ebeae5", "#c3c2b7"
TINT_BLUE, TINT_ORANGE, BAND = "#a8c9ef", "#f5bfa6", "#f6f5f1"
PASS_BAND, PASS_LINE, PASS_INK = "#e3f4ea", "#7cc49a", "#3d8f5f"
COL = {"Text": BLUE, "Text+fact": BLUE, "C2C": ORANGE}
TINT = {"Text": TINT_BLUE, "Text+fact": TINT_BLUE, "C2C": TINT_ORANGE}
WLW, CAP = 0.7, 3.0                  # whisker line width, cap size (matplotlib capsize, pt)

# slot positions (data units): pitch 1, an extra gap G between the pairs
G = 0.45
xs, x = {}, 0.0
for gi, (_, ks) in enumerate(GROUPS):
    x += G if gi else 0.0
    for s in ks:
        xs[s] = x; x += 1.0
BW, DX, BWB = 0.34, 0.19, 0.60       # (a) bar width and offset from the slot; (b) bar width

# ---------------- layout (inches) ----------------
W_IN, H_IN = 5.5, 3.05
L_IN, R_IN = 0.30, 0.02
TOP_IN, GAP_IN, B_H, B_BOT = 0.21, 0.29, 0.76, 0.52
A_H = H_IN - TOP_IN - GAP_IN - B_H - B_BOT
AX_W = W_IN - L_IN - R_IN
fig = plt.figure(figsize=(W_IN, H_IN))
axA = fig.add_axes([L_IN / W_IN, (B_BOT + B_H + GAP_IN) / H_IN, AX_W / W_IN, A_H / H_IN])
axB = fig.add_axes([L_IN / W_IN, B_BOT / H_IN, AX_W / W_IN, B_H / H_IN], sharex=axA)
PT = fig.dpi / 72
rend = fig.canvas.get_renderer()


def text_w(s, fs):
    t = fig.text(0, 0, s, fontsize=fs); w = t.get_window_extent(rend).width / PT; t.remove(); return w


def lift(s, size):
    """pt by which the pdf backend draws a mathtext string above its baseline: matplotlib (3.10,
    _mathtext.Output.to_vector) rounds the box height up to a whole pt and hangs the glyphs from the top.
    Without a correction "C2C, $q=0.50$" sits 0.5 pt higher than the plain "C2C" beside it."""
    if "$" not in s:
        return 0.0
    try:
        from matplotlib import _mathtext
        from matplotlib.mathtext import MathTextParser
        from matplotlib.ft2font import LoadFlags
        from matplotlib.font_manager import FontProperties
        prop = FontProperties(size=size)
        fset = MathTextParser._font_type_mapping[prop.get_math_fontfamily()](prop, LoadFlags.NO_HINTING)
        h = _mathtext.ship(_mathtext.Parser().parse(s, fset, size, 72)).box.height
        return float(np.ceil(h) - h)
    except Exception as e:                        # private API: fall back to no correction
        print("mathtext lift not corrected:", e)
        return 0.0


def unlift(t):
    """move a text down by its mathtext lift, so that its baseline matches plain text"""
    d = lift(t.get_text(), t.get_fontsize())
    if d:
        t.set_transform(t.get_transform() + ScaledTranslation(0, -d / 72, fig.dpi_scale_trans))
    return d


TARGET_TXT = "5% target"
XMIN = xs[SLOTS[0]] - 0.45
if False:  # ("right" variant of the preview, not used)
    # room right of the last bar for the label: clearance + label + inset (pt), solved for the axis span
    a = xs[SLOTS[-1]] + DX + BW / 2 - XMIN
    b = 5.0 + text_w(TARGET_TXT, FS_T) + 1.0
    XMAX = XMIN + a / (1 - b / (AX_W * 72))
else:
    XMAX = xs[SLOTS[-1]] + 0.45
axA.set_xlim(XMIN, XMAX)
PITCH = AX_W * 72 / (XMAX - XMIN)                # pt per slot

# ---------------- (a) change rate, u = 0 (tint) and u > 0 (solid) ----------------
axA.set_yscale("function", functions=(np.sqrt, np.square))
axA.set_ylim(0, 55)
YT = [0, 1, 5, 10, 20, 50]           # 2% is not labelled: on this axis it would sit 6 pt above the 1% label
axA.set_yticks(YT); axA.set_yticklabels([f"{v}%" for v in YT])
axA.axhspan(0, 5, color=PASS_BAND, lw=0, zorder=0.4)
for v in YT:
    if v not in (0, 5):
        axA.axhline(v, color=GRID, lw=0.6, zorder=0.5)
axA.axhline(5, color=PASS_LINE, lw=0.9, ls=(0, (3, 2)), zorder=1)
OBST = []                            # drawn data marks, as display boxes, for the collision checks
ZERO = []
for s in SLOTS:
    d = z[s]; ref = split(s)[2]; xc = xs[s]
    for dx, r, (lo, hi), col, part in ((-DX, d["r0"], d["c0"], TINT[ref], "u=0"), (DX, d["r1"], d["c1"], COL[ref], "u>0")):
        if r > 0:
            axA.bar(xc + dx, r, width=BW, color=col, lw=0, zorder=2)
        else:
            ZERO.append((s, part))   # a rate of 0: no bar, only the whisker from 0 to the upper bound
        eb = axA.errorbar([xc + dx], [r], yerr=[[r - lo], [hi - r]], fmt="none", ecolor=INK2, elinewidth=WLW,
                          capsize=CAP, capthick=WLW, zorder=3)
        for art in eb.get_children():
            art.set_clip_on(False)
        OBST.append(("bar " + s + " " + part, axA, (xc + dx - BW / 2, 0, xc + dx + BW / 2, r), 0))
        OBST.append(("whisker " + s + " " + part, axA, (xc + dx, lo, xc + dx, hi), CAP))
print("rates of 0 (whisker only):", ZERO)

# "5% target", above the line
if False:  # ("right" variant of the preview, not used)
    tx, ha, dxp = XMAX, "right", -1.0
else:
    s6, s7 = GROUPS[0][1][-1], GROUPS[1][1][0]
    tx, ha, dxp = (xs[s6] + DX + BW / 2 + xs[s7] + DX - BW / 2) / 2, "center", 0.0
tgt = axA.text(tx, 5, TARGET_TXT, ha=ha, va="bottom", fontsize=FS_T, color=PASS_INK, zorder=4,
               transform=axA.transData + ScaledTranslation(dxp / 72, 1.6 / 72, fig.dpi_scale_trans))


# legend: four bar patches and a whisker
class Whisker:
    pass


class HandlerWhisker(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        cx, y0, y1 = -xdescent + width / 2, -ydescent - 1.0, -ydescent + height + 1.0
        arts = [Line2D([cx, cx], [y0, y1]), Line2D([cx - CAP, cx + CAP], [y0, y0]),
                Line2D([cx - CAP, cx + CAP], [y1, y1])]
        for art in arts:
            art.set_color(INK2); art.set_linewidth(WLW); art.set_solid_capstyle("butt"); art.set_transform(trans)
        return arts


hand = [Patch(facecolor=TINT_BLUE, edgecolor="none"), Patch(facecolor=TINT_ORANGE, edgecolor="none"),
        Patch(facecolor=BLUE, edgecolor="none"), Patch(facecolor=ORANGE, edgecolor="none"), Whisker()]
labs = ["Text, $u=0$", "C2C, $u=0$", "Text, $u>0$", "C2C, $u>0$", "95% Clopper–Pearson interval"]
y50 = np.sqrt(50) / np.sqrt(55)                  # the 50% grid line, in axes fraction
x_leg = (xs[SLOTS[0]] - DX - BW / 2 - XMIN) / (XMAX - XMIN)
leg = axA.legend(hand, labs, handler_map={Whisker: HandlerWhisker()}, ncol=3, loc="upper left",
                 bbox_to_anchor=(x_leg, y50 - 3.0 / (A_H * 72)), frameon=False, fontsize=FS,
                 handlelength=0.8, handleheight=0.95, handletextpad=0.45, columnspacing=1.3,
                 labelspacing=0.3, borderpad=0.0, borderaxespad=0.0)
for t in leg.get_texts():
    unlift(t)
axA.set_title("(a) Change rate among skipped development questions", fontsize=FS_H, color=INK, loc="left", pad=4)

# ---------------- (b) share of the replayed saving from u > 0 ----------------
axB.set_ylim(0, 100)
axB.set_yticks([0, 50, 100]); axB.set_yticklabels(["0%", "50%", "100%"])
SHARE_LAB = []
for s in SLOTS:
    xc = xs[s]; ref = split(s)[2]
    if s in sav:
        v = 100 * sav[s]
        axB.bar(xc, v, width=BWB, color=COL[ref], lw=0, zorder=2)                  # u > 0 part, bottom
        axB.bar(xc, 100 - v, bottom=v, width=BWB, color=TINT[ref], lw=0, zorder=2)  # u = 0 part, top
        SHARE_LAB.append((s, v, axB.annotate(f"{v:.0f}%", (xc, v), xytext=(0, 1.3), textcoords="offset points",
                                             ha="center", va="bottom", fontsize=FS_T, color=INK2, zorder=4)))
    else:
        nr = axB.text(xc, 50, "not replayed", rotation=90, ha="center", va="center", fontsize=FS_T, color=INK2)
axB.set_title("(b) Share of the replayed saving from $u>0$", fontsize=FS_H, color=INK, loc="left", pad=4)
unlift(axB.title)

# ---------------- axes styling ----------------
for ax in (axA, axB):
    ax.tick_params(length=2.5, width=0.5, color=AXIS, labelcolor=INK2, labelsize=FS, pad=2)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
axB.set_xticks([xs[s] for s in SLOTS]); axB.set_xticklabels([])
axA.tick_params(axis="x", labelbottom=False)

# ---------------- setting names, once, under (b) ----------------
TB = blended_transform_factory(axB.transData, axB.transAxes)
def at(dy):
    return TB + ScaledTranslation(0, dy / 72, fig.dpi_scale_trans)
Y1, Y2, YBR, Y3 = -9.2, -18.0, -22.8, -30.6     # baselines of the two name lines, bracket, pair-name baseline (pt)
NAME_TXT, BRACKETS = [], []
for pair, ks in GROUPS:
    runs = []                         # benchmark printed once per run of equal benchmarks within a pair
    for s in ks:
        bm = split(s)[1]
        if runs and runs[-1][0] == bm:
            runs[-1][1].append(s)
        else:
            runs.append((bm, [s]))
    for bm, ss in runs:
        NAME_TXT.append((1, axB.text((xs[ss[0]] + xs[ss[-1]]) / 2, 0, bm, transform=at(Y1), ha="center",
                                     va="baseline", fontsize=FS_T, color=INK)))
    for s in ks:
        NAME_TXT.append((2, axB.text(xs[s], 0, split(s)[3], transform=at(Y2), ha="center", va="baseline",
                                     fontsize=FS_T, color=INK)))
        unlift(NAME_TXT[-1][1])
    x0, x1 = xs[ks[0]] - 0.45, xs[ks[-1]] + 0.45
    BRACKETS.append(axB.plot([x0, x1], [0, 0], transform=at(YBR), color=AXIS, lw=0.6, clip_on=False)[0])
    NAME_TXT.append((3, axB.text((x0 + x1) / 2, 0, f"{pair} pair", transform=at(Y3), ha="center",
                                 va="baseline", fontsize=FS, color=INK2)))

# ---------------- collision checks ----------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()


def dbox(ax, x0, y0, x1, y1, pad_pt):
    (a0, b0), (a1, b1) = ax.transData.transform([(x0, y0), (x1, y1)])
    return Bbox.from_extents(a0 - pad_pt * PT, b0 - WLW / 2 * PT, a1 + pad_pt * PT, b1 + WLW / 2 * PT)


def clear(b1, b2, gap_pt):
    """True if the boxes are at least gap_pt apart (in x or in y)."""
    g_ = gap_pt * PT
    return b1.x1 + g_ <= b2.x0 or b2.x1 + g_ <= b1.x0 or b1.y1 + g_ <= b2.y0 or b2.y1 + g_ <= b1.y0


def glyphs(t):
    """box of the glyphs of a text drawn with va='baseline' (display px): x from the text layout,
    y from the glyph outlines, so that the padding of a mathtext box does not count"""
    from matplotlib.textpath import TextPath
    wb = t.get_window_extent(rend)
    e = TextPath((0, 0), t.get_text(), size=t.get_fontsize(), prop=t.get_fontproperties()).get_extents()
    yb = t.get_transform().transform(t.get_position())[1] + lift(t.get_text(), t.get_fontsize()) * PT
    return Bbox.from_extents(wb.x0, yb + e.y0 * PT, wb.x1, yb + e.y1 * PT)


marks = [(n, dbox(ax, *e, p)) for n, ax, e, p in OBST]
lb = leg.get_window_extent(rend); tb = tgt.get_window_extent(rend)
for n, b in marks:
    assert clear(lb, b, 2.0), ("legend", n)
    assert clear(tb, b, 2.0), ("5% target", n)
assert clear(lb, tb, 2.0)
tb_ax = axA.get_window_extent(rend)
assert tb.x1 <= tb_ax.x1 + 0.5 and lb.y1 <= tb_ax.y1, "label or legend outside panel (a)"
y5, y50_px = (axA.transData.transform((0, v))[1] for v in (5, 50))
assert tb.y0 >= y5 + 0.8 * PT, "5% label not above the line"
assert lb.y1 <= y50_px - 1.0 * PT, "legend crosses the 50% grid line"
names = [(t.get_text(), glyphs(t), ln) for ln, t in NAME_TXT]
for i in range(len(names)):
    for j in range(i + 1, len(names)):     # same line: 3 pt apart; different lines: 1 pt apart
        assert clear(names[i][1], names[j][1], 3.0 if names[i][2] == names[j][2] else 1.0), (names[i][0], names[j][0])
for br in BRACKETS:
    bb = br.get_window_extent(rend)
    for n, b, _ in names:
        assert clear(b, bb, 1.5), ("bracket", n)
for s, v, t in SHARE_LAB:              # the share label sits inside the tint part
    b = t.get_window_extent(rend); top = axB.transData.transform((0, 100))[1]
    assert b.y1 + 1.0 * PT <= top, s
nb = nr.get_window_extent(rend); bb_ = axB.get_window_extent(rend)
assert bb_.y0 < nb.y0 and nb.y1 < bb_.y1
tbt = axB.title.get_window_extent(rend)
assert tbt.y1 + 2 * PT <= axA.get_window_extent(rend).y0, "title (b) touches panel (a)"
fb = fig.bbox
for t in [axA.title, axB.title, leg, tgt] + [t for _, t in NAME_TXT] + axA.get_yticklabels() + axB.get_yticklabels():
    b = t.get_window_extent(rend)
    assert fb.x0 <= b.x0 and b.x1 <= fb.x1 and fb.y0 <= b.y0 and b.y1 <= fb.y1, ("clipped", t)
print("pitch %.1f pt/slot; panel heights (a) %.2f in, (b) %.2f in; figure %.2f x %.2f in"
      % (PITCH, A_H, B_H, W_IN, H_IN))
print("min horizontal gap between names on one line: %.1f pt" % min(
    max(names[j][1].x0 - names[i][1].x1, names[i][1].x0 - names[j][1].x1) / PT
    for i in range(len(names)) for j in range(i + 1, len(names)) if names[i][2] == names[j][2]))

fig.savefig(OUT, metadata={"CreationDate": None})
print("wrote", OUT)

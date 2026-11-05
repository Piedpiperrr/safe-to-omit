"""Run from any directory: python3 code/figures/appendix/fig09_oos.py [output.pdf]

Figure 10 (Appendix I, fig:oos): every frozen out-of-sample test as a funnel
plot: x = number of skipped questions n (log scale), y = change rate among
skipped answers with its two-sided 95% Clopper-Pearson interval, one panel per
group of tests. The pass region (rates whose upper bound is below 5% at that
n) and the delta = 0.001 boundary are computed from the counts; markers are
filled when the exact test rejects at 0.001. Labels are placed by a small
search that keeps them clear of markers, lines, and each other.

Every number is checked against the LaTeX source it comes from (new_app_H.tex,
new_app_E.tex, tab_main_economics.tex), and every interval and p-value is
recomputed from the counts. The script stops if anything disagrees.
"""
import re, sys, os
import numpy as np
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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "figure_oos.pdf"
H = open(TEX / "new_app_H.tex").read()
E = open(TEX / "new_app_E.tex").read()
T2 = open(TEX / "tab_main_economics.tex").read()

def need(text, snippet, where):
    if snippet not in text:
        sys.exit(f"MISSING in {where}: {snippet!r}")

def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return 100 * lo, 100 * hi

# (group, label, ref, k, n, reported rate, reported CI, p<=0.001?, source text, snippet)
TEXT, C2C, OFF = "Text", "C2C", "official"
rows = [
 ("Sealed ARC-Challenge test", "large/ARC/Text", TEXT, 20, 1100, 1.82, (1.11, 2.79), True, H, "20 of 1,100 and 11 of 1,047 skipped answers ([1.11\\%, 2.79\\%] and [0.53\\%, 1.87\\%]"),
 ("Sealed ARC-Challenge test", "large/ARC/C2C", C2C, 11, 1047, 1.05, (0.53, 1.87), True, H, "20 of 1,100 and 11 of 1,047"),
 ("Sealed ARC-Challenge test", "large/ARC/C2C, official extraction", OFF, 54, 1044, 5.17, (3.91, 6.70), False, H, "54 of 1,044 skipped C2C answers change (5.17\\% [3.91, 6.70]"),
 ("Held-out OBQA", "medium/OBQA/C2C", C2C, 8, 388, 2.06, (0.89, 4.02), False, H, "8/388 & 2.06 [0.89, 4.02] & $2.5\\times10^{-3}$ & 0.025"),
 ("Held-out OBQA", "medium/OBQA/C2C, $q=0.50$ (sensitivity)", C2C, 6, 346, 1.73, (0.64, 3.74), False, H, "6/346 & 1.73 [0.64, 3.74] & $1.4\\times10^{-3}$ & 0.025"),
 ("Held-out OBQA", "large/OBQA/Text ‡", TEXT, 12, 595, 2.02, (1.05, 3.50), True, H, "12/595 & 2.02 [1.05, 3.50] & $1.5\\times10^{-4}$ & 0.001"),
 ("Held-out OBQA", "large/OBQA/C2C ‡", C2C, 13, 595, 2.18, (1.17, 3.71), True, H, "13/595 & 2.18 [1.17, 3.71] & $3.7\\times10^{-4}$ & 0.001"),
 ("Held-out OBQA", "large/OBQA/Text+fact ‡", TEXT, 16, 558, 2.87, (1.65, 4.61), False, H, "16/558 & 2.87 [1.65, 4.61] & $9.2\\times10^{-3}$ & 0.025"),
 ("Later tests, pass rule prespecified", "Llama-8B/OBQA/Text", TEXT, 3, 440, 0.68, (0.14, 1.98), True, H, "3/440 & 0.68 [0.14, 1.98] & $3.7\\times10^{-7}$"),
 ("Later tests, pass rule prespecified", "Llama-8B/ARC/Text", TEXT, 15, 794, 1.89, (1.06, 3.10), True, H, "15/794 & 1.89 [1.06, 3.10] & $4.4\\times10^{-6}$"),
 ("Later tests, pass rule prespecified", "medium/ARC/C2C", C2C, 15, 733, 2.05, (1.15, 3.35), True, H, "15/733 & 2.05 [1.15, 3.35] & $3.1\\times10^{-5}$"),
 ("Later tests, pass rule prespecified", "Llama-8B/SQuAD/Text (sealed test)", TEXT, 9, 364, 2.47, (1.14, 4.64), False, E, "9 of 364 skipped answers change (2.47\\% [1.14, 4.64], $p=0.012$)"),
 ("Options rotated, no recalibration", "large/ARC/Text", TEXT, 27, 1100, 2.45, (1.62, 3.55), True, H, "& 27 & 2.45 [1.62, 3.55] & $1.5\\times10^{-5}$"),
 ("Options rotated, no recalibration", "large/ARC/C2C", C2C, 18, 1049, 1.72, (1.02, 2.70), True, H, "& 18 & 1.72 [1.02, 2.70] & $1.9\\times10^{-8}$"),
 ("Options rotated, no recalibration", "large/OBQA/Text", TEXT, 13, 582, 2.23, (1.19, 3.79), True, H, "& 13 & 2.23 [1.19, 3.79] & $5.4\\times10^{-4}$"),
 ("Options rotated, no recalibration", "large/OBQA/C2C", C2C, 12, 582, 2.06, (1.07, 3.57), True, H, "& 12 & 2.06 [1.07, 3.57] & $2.2\\times10^{-4}$"),
 ("Options rotated, no recalibration", "large/OBQA/Text+fact", TEXT, 14, 545, 2.57, (1.41, 4.27), False, H, "& 14 & 2.57 [1.41, 4.27] & $3.4\\times10^{-3}$"),
]
# Table 2 of the main text must carry the same sealed/held-out/later counts.
for s in ["sealed 20/1,100", "sealed 11/1,047", "held out 8/388", "held out 12/595", "held out 13/595",
          "held out 16/558", "held out 3/440", "held out 15/794", "held out 15/733", "sealed 9/364", "54/1,044"]:
    need(T2, s, "tab_main_economics.tex")

# ---- checks: source text, rate, interval, p-value class ----
n_tests = 0
for g, lab, ref, k, n, rate, ci, strict, text, snip in rows:
    need(text, snip, lab)
    r = 100 * k / n
    assert abs(r - rate) < 0.006, (lab, r, rate)
    lo, hi = cp(k, n)
    assert abs(lo - ci[0]) < 0.006 and abs(hi - ci[1]) < 0.006, (lab, lo, hi, ci)
    p = binom.cdf(k, n, 0.05)
    assert (p <= 0.001) == strict, (lab, p, strict)
    if ref != OFF:
        assert hi < 5.0, (lab, hi)            # every frozen test: upper bound below 5%
        if not g.startswith("Options"):
            n_tests += 1
assert n_tests == 11, n_tests                  # "eleven frozen out-of-sample tests"
print("all checks passed:", n_tests, "tests +", sum(r[0].startswith("Options") for r in rows), "rotated runs")

# ---- figure ----
import itertools
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
from scipy.stats import beta, binom


GROUPS = ["Sealed ARC-Challenge test", "Held-out OBQA",
          "Later tests, pass rule prespecified", "Options rotated, no recalibration"]
assert [r[0] for r in rows] == sorted([r[0] for r in rows], key=GROUPS.index) and len(rows) == 17

# ---------------- style ----------------
FS, FS_H, FS_L = 8.0, 8.5, 7.5
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e"
GRID, AXIS = "#ebeae5", "#c3c2b7"
TINT_BLUE, TINT_ORANGE, TINT_GRAY = "#a8c9ef", "#f5bfa6", "#d6d5cf"
PASS_BAND, PASS_LINE, PASS_INK = "#e3f4ea", "#7cc49a", "#3d8f5f"
LEAD = "#9a9993"
DASH_PASS, DOT_STRICT, DASH_TARGET = (0, (3, 2)), (0, (1, 1.5)), (0, (5, 2.5))
MS, MS_D = 6.5, 5.4                      # circle/triangle size, diamond size
STY = {TEXT: (BLUE, TINT_BLUE, "o", MS), C2C: (ORANGE, TINT_ORANGE, "^", MS), OFF: (GRAY, TINT_GRAY, "D", MS_D)}

# ---------------- data for plotting ----------------
SHORT = {"medium/OBQA/C2C, $q=0.50$ (sensitivity)": "medium/OBQA/C2C ($q=0.50$)",
         "Llama-8B/SQuAD/Text (sealed test)": "Llama-8B/SQuAD/Text (sealed)"}
# planned correction: the three rotated large/OBQA runs are nominal development certificates too
ROT_NOMINAL = {"large/OBQA/Text", "large/OBQA/C2C", "large/OBQA/Text+fact"}
DODGE = 1.06     # where a Text and a C2C test share n in a panel, the C2C one is drawn at 1.06 n (else the markers collide)
DODGE_REF = C2C

pts = []
for grp, lab, ref, k, n, rate, ci, strict, *_ in rows:
    lo, hi = cp(k, n)
    assert abs(100 * k / n - rate) < 0.006 and abs(lo - ci[0]) < 0.006 and abs(hi - ci[1]) < 0.006
    name = SHORT.get(lab, lab)
    if grp.startswith("Options") and lab in ROT_NOMINAL:
        name = lab + " ‡"
    pts.append(dict(group=grp, label=name, ref=ref, k=k, n=n, x=float(n), rate=100 * k / n,
                    lo=lo, hi=hi, strict=strict))
OTHER = {TEXT: C2C, C2C: TEXT}
for p in pts:
    if p["ref"] == DODGE_REF and any(q is not p and q["group"] == p["group"] and q["n"] == p["n"]
                                     and q["ref"] == OTHER[DODGE_REF] for q in pts):
        p["x"] = p["n"] * DODGE
        p["dodged"] = True

# ---------------- boundaries (k real, bisection on [0, n]) ----------------
def bisect(f, lo, hi, it=80):
    """f increasing in k; returns the k where f crosses 0."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

def k_pass(n):     # largest k with two-sided 95% CP upper bound hi(k, n) = beta.ppf(0.975, k+1, n-k) below 5%
    return bisect(lambda k: beta.ppf(0.975, k + 1, n - k) - 0.05, 0.0, n - 1e-9)

def k_strict(n):   # largest k with binom.cdf(k, n, 0.05) = beta.sf(0.05, k+1, n-k) <= 0.001
    return bisect(lambda k: beta.sf(0.05, k + 1, n - k) - 0.001, 0.0, n - 1e-9)

# the beta identity used for real k agrees with the binomial cdf at every loaded count
for p in pts:
    k, n = p["k"], p["n"]
    assert abs(binom.cdf(k, n, 0.05) - beta.sf(0.05, k + 1, n - k)) < 1e-12
    assert abs(100 * beta.ppf(0.975, k + 1, n - k) - p["hi"]) < 1e-9

XMIN, XMAX, YMAX = 300.0, 1300.0, 7.5
NG = np.geomspace(XMIN, XMAX, 240)
PASS = np.array([100 * k_pass(n) / n for n in NG])
STRICT = np.array([100 * k_strict(n) / n for n in NG])

# every marker is on the correct side of both curves at the n where it is drawn
for p in pts:
    rp = 100 * k_pass(p["x"]) / p["x"]; rs = 100 * k_strict(p["x"]) / p["x"]
    below_pass = p["rate"] < rp
    assert below_pass == (p["hi"] < 5.0), (p["label"], p["rate"], rp, p["hi"])
    assert (p["rate"] <= rs) == p["strict"], (p["label"], p["rate"], rs, p["strict"])
    p["below_pass"], p["below_strict"] = below_pass, p["rate"] <= rs
off = [p for p in pts if p["ref"] == OFF]
assert len(off) == 1 and off[0]["rate"] > 5.0 and not off[0]["below_pass"]

# ---------------- layout (inches) ----------------
W, H = 5.5, 4.5
L_IN, R_IN, B_IN, T_IN = 0.50, 0.07, 0.80, 0.19
WS_IN, HS_IN = 0.13, 0.27
PW = (W - L_IN - R_IN - WS_IN) / 2
PH = (H - B_IN - T_IN - HS_IN) / 2
fig = plt.figure(figsize=(W, H))
axes = []
for i, grp in enumerate(GROUPS):
    r, c = divmod(i, 2)
    x0 = L_IN + c * (PW + WS_IN); y0 = B_IN + (1 - r) * (PH + HS_IN)
    axes.append(fig.add_axes([x0 / W, y0 / H, PW / W, PH / H]))

XT = [300, 500, 700, 1000]
for i, (ax, grp) in enumerate(zip(axes, GROUPS)):
    r, c = divmod(i, 2)
    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(0, YMAX)
    ax.xaxis.set_major_locator(FixedLocator(XT)); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:,}" for t in XT] if r == 1 else [""] * len(XT)))
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"{t}%" for t in range(8)] if c == 0 else [""] * 8)
    ax.tick_params(length=2.5, width=0.5, color=AXIS, labelcolor=INK2, labelsize=FS, pad=2)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(AXIS)
    for t in XT[1:]: ax.axvline(t, color=GRID, lw=0.5, zorder=0.5)
    for t in range(1, 8): ax.axhline(t, color=GRID, lw=0.5, zorder=0.5)
    # pass region and boundaries
    ax.fill_between(NG, 0, PASS, color=PASS_BAND, lw=0, zorder=0.2)
    ax.plot(NG, PASS, color=PASS_LINE, lw=0.9, ls=DASH_PASS, zorder=1.2)
    ax.plot(NG, STRICT, color=PASS_INK, lw=0.8, ls=DOT_STRICT, zorder=1.2)
    ax.axhline(5.0, color=INK2, lw=0.6, ls=DASH_TARGET, zorder=1.1)
    ax.set_title(grp, loc="left", fontsize=FS_H, color=INK, pad=4)
    for p in pts:
        if p["group"] != grp: continue
        col, tint, m, ms = STY[p["ref"]]
        ax.plot([p["x"]] * 2, [p["lo"], p["hi"]], color=tint, lw=1.6, solid_capstyle="butt", zorder=2)
        if p["strict"]:
            ax.plot([p["x"]], [p["rate"]], marker=m, ms=ms, mfc=col, mec="white", mew=0.6, ls="none", zorder=3)
        else:
            ax.plot([p["x"]], [p["rate"]], marker=m, ms=ms, mfc="white", mec=col, mew=1.2, ls="none", zorder=3)
        p["ax"] = ax

fig.text((L_IN + PW + WS_IN / 2) / W, 0.47 / H, "skipped questions $n$ (log scale)",
         ha="center", va="center", color=INK2, fontsize=FS)
fig.text(0.075 / W, (B_IN + PH + HS_IN / 2) / H, "changed skipped answers (%)", rotation=90,
         ha="center", va="center", color=INK2, fontsize=FS)

# ---------------- legend (two rows at the bottom) ----------------
class PassHandler(HandlerBase):
    def create_artists(self, legend, orig, x0, y0, w, h, fontsize, trans):
        top = y0 + 0.62 * h
        band = Rectangle((x0, y0 - 0.05 * h), w, top - (y0 - 0.05 * h), facecolor=PASS_BAND, lw=0, transform=trans)
        ln = Line2D([x0, x0 + w], [top, top], color=PASS_LINE, lw=0.9, ls=DASH_PASS, transform=trans)
        return [band, ln]

class CIHandler(HandlerBase):
    """Marker on a short vertical interval line, as in the panels."""
    def __init__(self, col, tint, m, ms, filled):
        super().__init__(); self.a = (col, tint, m, ms, filled)
    def create_artists(self, legend, orig, x0, y0, w, h, fontsize, trans):
        col, tint, m, ms, filled = self.a
        xc, yc = x0 + w / 2, y0 + h / 2
        ci = Line2D([xc, xc], [y0 - 0.25 * h, y0 + 1.25 * h], color=tint, lw=1.6, solid_capstyle="butt", transform=trans)
        mk = Line2D([xc], [yc], marker=m, ms=ms, mfc=col if filled else "white", mec="white" if filled else col,
                    mew=0.6 if filled else 1.2, ls="none", transform=trans)
        return [ci, mk]

class Dummy:  # handle placeholder
    pass

h_text, h_c2c, h_off = Dummy(), Dummy(), Dummy()
h_fill = Line2D([], [], marker="o", ms=MS, mfc=INK2, mec="white", mew=0.6, ls="none")
h_open = Line2D([], [], marker="o", ms=MS, mfc="white", mec=INK2, mew=1.2, ls="none")
h_pass = Dummy()
h_strict = Line2D([], [], color=PASS_INK, lw=0.8, ls=DOT_STRICT)
h_target = Line2D([], [], color=INK2, lw=0.6, ls=DASH_TARGET)
hmap = {h_text: CIHandler(BLUE, TINT_BLUE, "o", MS, True), h_c2c: CIHandler(ORANGE, TINT_ORANGE, "^", MS, True),
        h_off: CIHandler(GRAY, TINT_GRAY, "D", MS_D, False), h_pass: PassHandler()}
row1 = [(h_text, "Text"), (h_c2c, "C2C"), (h_off, "official C2C extraction"),
        (h_fill, r"filled: $p\leq0.001$"), (h_open, "open: upper bound below 5% only")]
row2 = [(h_pass, "pass region (upper bound below 5%)"), (h_strict, r"$\delta=0.001$ boundary"),
        (h_target, "5% target")]
LEG_KW = dict(frameon=False, fontsize=FS, handlelength=1.3, handleheight=1.0, handletextpad=0.45,
              columnspacing=1.15, borderaxespad=0, borderpad=0, handler_map=hmap)
cx = 0.5
leg1 = fig.legend([h for h, _ in row1], [t for _, t in row1], loc="center", ncol=len(row1),
                  bbox_to_anchor=(cx, 0.27 / H), **LEG_KW)
leg2 = fig.legend([h for h, _ in row2], [t for _, t in row2], loc="center", ncol=len(row2),
                  bbox_to_anchor=(cx, 0.105 / H), **LEG_KW)

# ---------------- point labels: search for positions, then leaders ----------------
fig.canvas.draw()
REND = fig.canvas.get_renderer()
PT = fig.dpi / 72.0
def to_pt(ax, x, y): return ax.transData.transform((x, y)) / PT
def from_pt(ax, P): return ax.transData.inverted().transform(np.asarray(P, float) * PT)

def split2(lab):
    """two-line variant of a long label, broken at ' (' or ', '"""
    for sep, keep in ((" (", "("), (", ", ",")):
        if sep in lab:
            a_, b_ = lab.split(sep, 1)
            return (a_ + ("," if keep == "," else "")) + "\n" + (("(" if keep == "(" else "") + b_)
    return None

for p in pts:
    p["variants"] = []
    for txt, pen in ((p["label"], 0.0), (split2(p["label"]), 6.0)):
        if txt is None: continue
        t = fig.text(0, 0, txt, fontsize=FS_L, linespacing=1.0)
        bb = t.get_window_extent(REND); t.remove()
        p["variants"].append((txt, bb.width / PT, bb.height / PT, pen))
    p["P"] = to_pt(p["ax"], p["x"], p["rate"])
    col, tint, m, ms = STY[p["ref"]]
    p["half"] = (0.71 if m == "D" else 0.5) * ms + (0.3 if p["strict"] else 0.6)   # half size of the marker box

# ---- geometry helpers (all in points) ----
def box_overlap(a, b, pad=0.0):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])

def seg_box(p0, p1, bx, pad=0.0):
    """Does segment p0-p1 meet box bx (x0, y0, x1, y1) grown by pad? (Liang-Barsky)"""
    x0, y0, x1, y1 = bx[0] - pad, bx[1] - pad, bx[2] + pad, bx[3] + pad
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p0[0] - x0), (dx, x1 - p0[0]), (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if pp == 0:
            if qq < 0: return False
        else:
            t = qq / pp
            if pp < 0:
                if t > t1: return False
                t0 = max(t0, t)
            else:
                if t < t0: return False
                t1 = min(t1, t)
    return t0 <= t1

def seg_seg(a0, a1, b0, b1):
    def orient(p, q, r): return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return (orient(a0, a1, b0) != orient(a0, a1, b1)) and (orient(b0, b1, a0) != orient(b0, b1, a1))

def seg_pt(a0, a1, q):
    """distance from point q to segment a0-a1"""
    a0, a1, q = np.asarray(a0, float), np.asarray(a1, float), np.asarray(q, float)
    d = a1 - a0
    t = np.clip(np.dot(q - a0, d) / max(np.dot(d, d), 1e-12), 0, 1)
    return float(np.hypot(*(a0 + t * d - q)))

STOP = 1.2          # leader stops this far outside the marker's edge
START = 1.2         # leader starts this far outside the text box
ANGLES = np.deg2rad(np.arange(0, 360, 7.5))
LENS = [5, 7, 9, 11.5, 14, 17, 21, 25, 30, 36, 43, 51, 60]
FRACS = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]
PAD_BOX, PAD_LINE = 2.4, 2.0

def pts_in_box(Q, bx, pad):
    return bool(np.any((Q[:, 0] >= bx[0] - pad) & (Q[:, 0] <= bx[2] + pad) &
                       (Q[:, 1] >= bx[1] - pad) & (Q[:, 1] <= bx[3] + pad)))

def n_cross(Q, a0, a1):
    """number of crossings of polyline Q with segment a0-a1 (vectorized)"""
    P0, P1 = Q[:-1], Q[1:]
    def orient(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))
    A0, A1 = np.broadcast_to(a0, P0.shape), np.broadcast_to(a1, P0.shape)
    o1, o2 = orient(A0, A1, P0), orient(A0, A1, P1)
    o3, o4 = orient(P0, P1, A0), orient(P0, P1, A1)
    return int(np.sum((o1 != o2) & (o3 != o4)))

def densify(Q, step=0.4):
    out = [Q[0]]
    for a_, b_ in zip(Q[:-1], Q[1:]):
        m = max(1, int(np.ceil(np.hypot(*(b_ - a_)) / step)))
        out.extend(a_ + (b_ - a_) * t for t in np.arange(1, m + 1) / m)
    return np.array(out)

# cost of a label position: leader length (long ones cost more), steepness, and every line the leader crosses
# (a boundary curve or the 5% line: 14; another test's interval line: 60)
XCI_W, UY_W = 60.0, 4.0
PIN = {}   # "label|group" -> hand placement, see pinned(); empty: every label is placed by the search

def candidates(p, stat):
    key = p["label"] + "|" + p["group"]
    if key in PIN:
        return [pinned(p, stat, PIN[key])]
    out = []
    for txt, w_, h_, pen in p["variants"]:
        out += candidates1(p, stat, txt, w_ / 2, h_ / 2, pen)
    out.sort(key=lambda d: d["cost"])
    # thin out near-duplicates (same box and leader start within 2 pt), keep the cheapest
    kept, seen = [], set()
    for d in out:
        key = (round(d["box"][0] / 2), round(d["box"][1] / 2), round(d["s"][0] / 2), round(d["s"][1] / 2))
        if key not in seen:
            seen.add(key); kept.append(d)
    return kept

def static_check(p, stat, bx, S, f):
    """hard rules for one label box bx and its leader S->f; returns (list of broken rules, info)"""
    bad = []
    X0, Y0, X1, Y1 = stat["frame"]
    if bx[0] < X0 + 2.0 or bx[2] > X1 - 1.5 or bx[1] < Y0 + 2.5 or bx[3] > Y1 - 1.5:
        bad.append("frame")
    if any(box_overlap(bx, mb, PAD_BOX) for mb in stat["markers"]):
        bad.append("label-marker")
    if any(box_overlap(bx, cb, 2.4) for cb in stat["cis"]):
        bad.append("label-interval")
    if any(pts_in_box(Q, bx, PAD_LINE) for Q in stat["dense"]):
        bad.append("label-curve")
    Lv = float(np.hypot(*(S - f)))
    if Lv < 4.9:
        bad.append("leader too short")
    u = (S - f) / max(Lv, 1e-9)
    if abs(u[0]) < 0.42:
        bad.append("leader too steep")
    xci = 0
    for q, mb, cb in zip(stat["pts"], stat["markers"], stat["cis"]):
        if q is p: continue
        if seg_box(S, f, mb, 1.5):
            bad.append("leader-marker " + q["label"])
        xm = 0.5 * (cb[0] + cb[2])
        if seg_pt(S, f, (xm, cb[1])) < 3.5 or seg_pt(S, f, (xm, cb[3])) < 3.5:
            bad.append("leader near interval end " + q["label"])
        elif seg_box(S, f, cb, 0.6):
            yc = S[1] + (f[1] - S[1]) * (xm - S[0]) / (f[0] - S[0]) if f[0] != S[0] else S[1]
            if abs(yc - q["P"][1]) < 8.0:
                bad.append("leader crosses interval near marker " + q["label"])
            xci += 1
    cross = sum(n_cross(Q, S, f) for Q in stat["curves"][:2])
    x5 = n_cross(stat["curves"][2], S, f)
    return bad, dict(Lv=Lv, u=u, xci=xci, cross=cross, x5=x5)

def cost_of(info, fr, pen):
    Lv, u = info["Lv"], info["u"]
    return (Lv + 0.012 * Lv ** 2 + UY_W * abs(u[1]) + 1.5 * fr + 14.0 * info["cross"] + 14.0 * info["x5"]
            + XCI_W * info["xci"] + pen)

def candidates1(p, stat, txt, hw, hh, pen):
    out = []
    P = p["P"]
    stop = p["half"] + STOP
    for a in ANGLES:
        u = np.array([np.cos(a), np.sin(a)])
        if abs(u[0]) < 0.45:
            continue                       # nearly vertical leaders would run along the interval line
        for Lv in LENS:
            S = P + u * (stop + Lv)        # leader start (label side)
            f = P + u * stop               # leader end (marker side)
            anchors = []                   # box centres such that S lies START outside the box edge
            if u[0] > 0.2: anchors.append((S[0] + START + hw, S[1], 0.0))
            if u[0] < -0.2: anchors.append((S[0] - START - hw, S[1], 0.0))
            for fr in FRACS:
                if u[1] > 0.2: anchors.append((S[0] - fr * hw, S[1] + START + hh, abs(fr)))
                if u[1] < -0.2: anchors.append((S[0] - fr * hw, S[1] - START - hh, abs(fr)))
            for cx_, cy_, fr in anchors:
                bx = (cx_ - hw, cy_ - hh, cx_ + hw, cy_ + hh)
                bad, info = static_check(p, stat, bx, S, f)
                if bad: continue
                off_ = (S[0] - cx_) / hw
                ma = "left" if off_ < -0.3 else ("right" if off_ > 0.3 else "center")
                out.append(dict(c=np.array([cx_, cy_]), box=bx, s=S, f=f, cost=cost_of(info, fr, pen), txt=txt, ma=ma,
                                cross=info["cross"], x5=info["x5"], xci=info["xci"]))
    return out

def pinned(p, stat, spec):
    """a hand-placed label: spec = (x0, y0, variant, (ax_, ay_)) in points from the panel's lower-left corner;
    (ax_, ay_) is where on the box (fractions of width and height) the leader leaves it."""
    x0, y0, vi, (fx, fy) = spec
    txt, w_, h_, pen = p["variants"][vi]
    X0, Y0 = stat["frame"][:2]
    bx = (X0 + x0, Y0 + y0, X0 + x0 + w_, Y0 + y0 + h_)
    A_ = np.array([bx[0] + fx * w_, bx[1] + fy * h_])
    P = p["P"]
    d = P - A_; L = np.hypot(*d); u = d / L
    # leave the box grown by START along the ray from the attach point towards the marker
    ts = []
    for k_, lo_, hi_ in ((0, bx[0] - START, bx[2] + START), (1, bx[1] - START, bx[3] + START)):
        if u[k_] > 1e-9: ts.append((hi_ - A_[k_]) / u[k_])
        elif u[k_] < -1e-9: ts.append((lo_ - A_[k_]) / u[k_])
    S = A_ + u * min(ts)
    f = P - u * (p["half"] + STOP)
    bad, info = static_check(p, stat, bx, S, f)
    if bad:
        print("  PIN PROBLEM", p["label"], bad)
    ma = "left" if fx < 0.3 else ("right" if fx > 0.7 else "center")
    return dict(c=np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2]), box=bx, s=S, f=f, cost=cost_of(info, 0, pen),
                txt=txt, ma=ma, cross=info["cross"], x5=info["x5"], xci=info["xci"])

def compatible(ca, cb):
    if box_overlap(ca["box"], cb["box"], 3.0): return False
    if seg_box(ca["s"], ca["f"], cb["box"], 1.2) or seg_box(cb["s"], cb["f"], ca["box"], 1.2): return False
    if seg_seg(ca["s"], ca["f"], cb["s"], cb["f"]): return False
    return True

STATS = {}
for ax, grp in zip(axes, GROUPS):
    P_ = [p for p in pts if p["group"] == grp]
    bb = ax.get_window_extent(REND)
    frame = (bb.x0 / PT, bb.y0 / PT, bb.x1 / PT, bb.y1 / PT)
    curves = [np.array([to_pt(ax, x, y) for x, y in zip(NG, PASS)]),
              np.array([to_pt(ax, x, y) for x, y in zip(NG, STRICT)]),
              np.array([to_pt(ax, XMIN, 5.0), to_pt(ax, XMAX, 5.0)])]
    markers = [(q["P"][0] - q["half"], q["P"][1] - q["half"], q["P"][0] + q["half"], q["P"][1] + q["half"]) for q in P_]
    cis = []
    for q in P_:
        a0 = to_pt(ax, q["x"], q["lo"]); a1 = to_pt(ax, q["x"], q["hi"])
        cis.append((a0[0] - 0.8, a0[1], a0[0] + 0.8, a1[1]))
    stat = dict(frame=frame, curves=curves, dense=[densify(Q) for Q in curves], markers=markers, cis=cis, pts=P_)
    STATS[grp] = stat
    C = [candidates(q, stat) for q in P_]
    for q, cq in zip(P_, C):
        assert cq, ("no free position for", q["label"])
    order = sorted(range(len(P_)), key=lambda i: len(C[i]))
    best = [np.inf, None]
    def dfs(j, chosen, cost):
        if cost >= best[0]: return
        if j == len(order):
            best[0], best[1] = cost, dict(chosen); return
        i = order[j]
        for cand in C[i]:
            if cost + cand["cost"] >= best[0]: break
            if all(compatible(cand, chosen[i2]) for i2 in chosen):
                chosen[i] = cand; dfs(j + 1, chosen, cost + cand["cost"]); del chosen[i]
    dfs(0, {}, 0.0)
    assert best[1] is not None, ("no joint placement in", grp)
    for i, q in enumerate(P_):
        q["place"] = best[1][i]
    print(f"{grp}: cost {best[0]:.1f}", [len(c) for c in C])
    for i, q in enumerate(P_):
        d = best[1][i]
        print(f'    {q["label"]:30s} leader {np.hypot(*(d["s"] - d["f"])):5.1f} pt, crosses curves {d["cross"]}, 5% line {d["x5"]}, other intervals {d["xci"]}')

for p in pts:
    ax, pl = p["ax"], p["place"]
    cx_, cy_ = from_pt(ax, pl["c"])
    p["text"] = ax.text(cx_, cy_, pl["txt"], fontsize=FS_L, color=INK, ha="center", va="center", zorder=5,
                        linespacing=1.0, multialignment=pl["ma"])
    (x0, y0), (x1, y1) = from_pt(ax, [pl["s"], pl["f"]])
    ax.plot([x0, x1], [y0, y1], color=LEAD, lw=0.5, zorder=2.5, solid_capstyle="butt")

# ---- final check: every rule again, with the text boxes as rendered ----
fig.canvas.draw()
problems = []
for ax, grp in zip(axes, GROUPS):
    P_ = [p for p in pts if p["group"] == grp]
    stat = STATS[grp]
    final = []
    for p in P_:
        b = p["text"].get_window_extent(REND)
        bx = (b.x0 / PT, b.y0 / PT, b.x1 / PT, b.y1 / PT)
        bad, info = static_check(p, stat, bx, p["place"]["s"], p["place"]["f"])
        problems += [(p["label"], x) for x in bad]
        final.append(dict(box=bx, s=p["place"]["s"], f=p["place"]["f"]))
    for (i, a_), (j, b_) in itertools.combinations(enumerate(final), 2):
        if not compatible(a_, b_):
            problems.append((P_[i]["label"], "clashes with " + P_[j]["label"]))
    # panel title must stay clear of the plot area and the labels
    tb = ax.title.get_window_extent(REND)
    assert tb.y0 / PT >= stat["frame"][3], grp
print("label problems:", problems or "none")
assert not problems
for lg in (leg1, leg2):
    b = lg.get_window_extent(REND)
    print("legend row: x %.1f..%.1f pt of %.1f, y %.1f..%.1f pt" % (b.x0 / PT, b.x1 / PT, W * 72, b.y0 / PT, b.y1 / PT))
    assert b.x0 / PT >= 2 and b.x1 / PT <= W * 72 - 2 and b.y0 / PT >= 1

fig.savefig(OUT, metadata={"CreationDate": None})
print("wrote", OUT, "height %.2f in" % H)
for p in pts:
    print(f'{p["group"][:14]:14s} {p["label"]:30s} n={p["n"]:5d} x={p["x"]:7.1f} rate={p["rate"]:.2f} '
          f'CI=[{p["lo"]:.2f},{p["hi"]:.2f}] strict={p["strict"]} below_pass={p["below_pass"]} '
          f'below_strict={p["below_strict"]}')

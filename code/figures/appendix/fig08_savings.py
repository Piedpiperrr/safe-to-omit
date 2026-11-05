"""Run from any directory: python3 code/figures/appendix/fig08_savings.py [output.pdf]

Figure 9 (Appendix H, fig:savings): mean end-to-end saving of every replayed
policy across runs and implementations, as grouped bars.

Two panels with separate scales (Text-reference and C2C-reference policies).
For each policy: solid bar = the original replay (first configuration;
Table 18, tab:e2e_full) with its paired bootstrap 95% whisker and its mean
printed above; markers on it = the three repeats (second configuration;
Table 20, tab:repeats); outlined bar = the single later run of Text+fact or a
Llama-8B policy (second configuration; Appendix H text, Table 29); hatched
bar = answering skipped questions from the probe's argmax (second
configuration; Table 21, tab:impl); black tick = the component estimate fixed
before the replays (large pair: results/boundaries_large_small/summary/
panel_cost.csv; medium pair: the CPU recombination stated in Appendix H).

Checks: every number is parsed from the appendix source; the original savings
agree across Tables 18 and 20 and the main text's Table 2 is not touched; the
component estimates agree with the stated differences (-1.76 to +20.50 ms);
every saving class is recomputed from the intervals; and the figure's claims
(argmax raises every saving; classes are unchanged) are asserted.
"""
import csv, re, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
HERE = Path(__file__).resolve().parent            # code/figures/appendix
ROOT = HERE.parents[2]                             # repository root
TEX = HERE / "paper_tex"                           # appendix LaTeX of the submitted paper
REC = HERE / "record_tables"                       # fuller tables that the appendix condenses
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "figure_savings.pdf"
G = open(TEX / "new_app_G.tex").read()
NUM = r"(\$?[+-]?[0-9.]+\$?) \[(\$?[+-]?[0-9.]+\$?), (\$?[+-]?[0-9.]+\$?)\]"
def f(x): return float(x.replace("$", ""))
def tab(label):
    i = G.index("\\label{%s}" % label); return G[i:G.index("\\end{table}", i)]

orig = {}
for m in re.finditer(r"(medium|large)/(OBQA|ARC|MMLU-Pro) & (Text|C2C) & \d+/128 & [0-9.]+ & [0-9.]+ & " + NUM, tab("tab:e2e_full")):
    orig[f"{m.group(1)}/{m.group(2)}/{m.group(3)}"] = [f(m.group(i)) for i in (4, 5, 6)]
assert len(orig) == 8
t19 = tab("tab:repeats")
t19a = t19[:t19.index("(b)")] if "(b)" in t19 else t19
rep = {}
for m in re.finditer(r"(medium|large) & (OBQA|ARC|MMLU-Pro) & (Text|C2C) & " + " & ".join([NUM] * 4), t19a):
    s = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"; v = [f(x) for x in m.groups()[3:]]
    assert v[0:3] == orig[s], s                                  # Table 20 original = Table 18
    rep[s] = [v[3:6], v[6:9], v[9:12]]
assert len(rep) == 8
impl = {}
for m in re.finditer(r"(medium|large) & (OBQA|ARC|MMLU-Pro) & (Text|C2C) & " + " & ".join([NUM] * 3), tab("tab:impl")):
    s = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"; v = [f(x) for x in m.groups()[3:]]
    impl[s] = dict(orig=v[0:3], reuse=v[3:6], argmax=v[6:9])
assert len(impl) == 8
for s, d in impl.items():
    assert d["argmax"][0] > d["orig"][0], s                        # argmax raises every saving
    assert abs(d["reuse"][0] - d["orig"][0]) <= 5.0, s             # reuse within 5 ms
# single later runs on the second configuration (Appendix H text)
later = {}
for s, pat in (("Llama-8B/OBQA/Text", r"Llama-8B savings of ([0-9.]+)~ms \[([0-9.]+), ([0-9.]+)\]"),
               ("Llama-8B/ARC/Text", r"Llama-8B savings of [0-9.]+~ms \[[0-9.]+, [0-9.]+\] and ([0-9.]+)~ms \[([0-9.]+), ([0-9.]+)\]"),
               ("large/OBQA/Text+fact", r"Text\+fact policy skips .*?a saving of ([0-9.]+)~ms \[([0-9.]+), ([0-9.]+)\]")):
    m = re.search(pat, G); assert m, s
    later[s] = [float(x) for x in m.groups()]
# component estimates fixed before the replays
comp = {}
for r in csv.DictReader(open(ROOT / "results/boundaries_large_small/summary/panel_cost.csv")):
    if r["pair"] == "large" and r["policy"] == "ProbeMax":
        s = f"large/{ {'obqa': 'OBQA', 'arc': 'ARC'}[r['dataset']]}/{ {'T': 'Text', 'C': 'C2C'}[r['reference']]}"
        comp[s] = float(r["net_saving_ms"])
diff = {"large/OBQA/Text": -1.76, "large/OBQA/C2C": 8.40, "large/ARC/Text": 20.50, "large/ARC/C2C": 9.74}
assert "saved 1.76~ms less than this estimate" in G and "8.40, 20.50, and 9.74~ms more" in G
for s, dv in diff.items():
    assert abs((orig[s][0] - comp[s]) - dv) < 0.06, (s, orig[s][0] - comp[s], dv)
m = re.search(r"had projected ([0-9.]+) and ([0-9.]+)~ms for the medium pair", G); assert m
comp["medium/OBQA/C2C"], comp["medium/ARC/C2C"] = float(m.group(1)), float(m.group(2))
# saving classes: interval above zero except large/MMLU-Pro/C2C
def cls(ci): return "up" if ci[1] > 0 else ("down" if ci[2] < 0 else "zero")
for s in orig:
    want = "zero" if s == "large/MMLU-Pro/C2C" else "up"
    assert cls(orig[s]) == want and all(cls(r) == want for r in rep[s]), s
print("checks passed")

# ---------------- figure ----------------

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple, HandlerBase
from matplotlib.transforms import blended_transform_factory

FS, FS_H, FS_S = 8.0, 8.5, 7.5
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "hatch.linewidth": 0.5})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#ebeae5", "#c3c2b7"
MK = {"Text": "o", "C2C": "^"}

W_BAR, D_ARG = 0.32, 0.34            # bar width; offset of the argmax bar from the original bar
JIT = (-0.06, 0.0, 0.06)             # horizontal jitter of the three repeats
# group centres (x ticks): a pair (original bar at centre - 0.17, argmax bar at centre + 0.17) or a single
# later-run bar centred on its slot. The Text+fact slot is set 1.30 after MMLU-Pro and 1.05 before
# Llama-8B/OBQA so that its two-line label ("OBQA" over "(Text+fact)") keeps >= 4 pt clear of both neighbours.
PANELS = [
    dict(ref="Text", col=BLUE,
         rows=["large/OBQA/Text", "large/ARC/Text", "large/MMLU-Pro/Text", "large/OBQA/Text+fact",
               "Llama-8B/OBQA/Text", "Llama-8B/ARC/Text"],
         cx=[0.17, 1.17, 2.17, 3.47, 4.52, 5.52], xlim=(-0.30, 5.86),
         groups=[("large", 0, 3), ("Llama-8B", 4, 5)], yt=[0, 200, 400, 600, 800]),
    dict(ref="C2C", col=ORANGE,
         rows=["medium/OBQA/C2C", "medium/ARC/C2C", "large/OBQA/C2C", "large/ARC/C2C", "large/MMLU-Pro/C2C"],
         cx=[0.17, 1.17, 2.17, 3.17, 4.17], xlim=(-0.30, 4.57),
         groups=[("medium", 0, 1), ("large", 2, 4)], yt=[0, 100, 200, 300]),
]
# y limits from the data: Text starts at 0; C2C from -30 (its intervals go below zero)
for P in PANELS:
    his = []
    for s in P["rows"]:
        if s in orig:
            his += [orig[s][2], impl[s]["argmax"][2]] + [r[0] for r in rep[s]] + ([comp[s]] if s in comp else [])
        else:
            his += [later[s][2]]
    lo = 0.0 if P["ref"] == "Text" else -30.0
    P["ylim"] = (lo, max(his) + 0.08 * (max(his) - lo))
    assert P["ylim"][1] > P["yt"][-1]
    if P["ref"] == "C2C":
        assert min(orig[s][1] for s in P["rows"]) > lo      # every drawn interval stays inside the panel

def dataset_label(s):
    parts = s.split("/")
    return parts[1] + ("\n(Text+fact)" if parts[2] == "Text+fact" else "")

# ---------------- layout (inches) ----------------
FW, FH = 5.5, 3.0
L, GAP, R = 0.40, 0.30, 0.08          # left margin (y label + ticks), gap between panels, right margin
B, T = 0.68, 0.20                      # bottom (x labels, group row, legend), top (panel titles)
BRACKET_PT = 24.0                      # depth of the pair bracket below the axes (points)
units = [P["xlim"][1] - P["xlim"][0] for P in PANELS]
aw = FW - L - GAP - R
wid = [aw * u / sum(units) for u in units]
fig = plt.figure(figsize=(FW, FH))
axes = [fig.add_axes([L / FW, B / FH, wid[0] / FW, (FH - B - T) / FH]),
        fig.add_axes([(L + wid[0] + GAP) / FW, B / FH, wid[1] / FW, (FH - B - T) / FH])]
PT_PER_UNIT = wid[0] * 72 / units[0]

labels_num = []                        # (text artist, argmax-obstacle artists) for the nudge step
for ax, P in zip(axes, PANELS):
    c = P["col"]; mk = MK[P["ref"]]
    for s, xc in zip(P["rows"], P["cx"]):
        if s in orig:
            x0, xa = xc - D_ARG / 2, xc + D_ARG / 2
            m, lo, hi = orig[s]
            ax.bar(x0, m, W_BAR, color=c, lw=0, zorder=2)
            ax.errorbar(x0, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=0.7,
                        capsize=1.6, capthick=0.7, zorder=3)
            for dx, r in zip(JIT, rep[s]):
                ax.plot([x0 + dx], [r[0]], marker=mk, ms=4, mfc=c, mec="white", mew=0.6, ls="none", zorder=4)
            am, alo, ahi = impl[s]["argmax"]
            ab = ax.bar(xa, am, W_BAR, color="white", edgecolor=c, lw=0.6, hatch="////", zorder=2)
            ae = ax.errorbar(xa, am, yerr=[[am - alo], [ahi - am]], fmt="none", ecolor=INK2, elinewidth=0.7,
                             capsize=1.6, capthick=0.7, zorder=3)
            if s in comp:
                ax.plot([x0 - W_BAR / 2, x0 + W_BAR / 2], [comp[s]] * 2, color=INK, lw=1.4,
                        solid_capstyle="butt", zorder=5)
            xl, obst = x0, [ab.patches[0]] + list(ae.lines[1]) + list(ae.lines[2])
        else:
            m, lo, hi = later[s]
            ax.bar(xc, m, W_BAR, color="white", edgecolor=c, lw=1.0, zorder=2)
            ax.errorbar(xc, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=0.7,
                        capsize=1.6, capthick=0.7, zorder=3)
            xl, obst = xc, []
        # mean of the original / later run above the upper whisker end
        t = ax.annotate(f"{m:.0f}", (xl, hi), xytext=(0, 2.2), textcoords="offset points", ha="center",
                        va="bottom", fontsize=FS_S, color=INK2, zorder=6)
        labels_num.append((ax, t, obst))
    # axes, grid, zero line
    ax.set_xlim(*P["xlim"])
    ax.set_ylim(*P["ylim"]); ax.set_yticks(P["yt"])
    for v in P["yt"][1:]:
        ax.axhline(v, color=GRID, lw=0.55, zorder=0)
    ax.axhline(0, color=INK2, lw=0.7, zorder=2.5)
    ax.set_xticks(P["cx"]); ax.set_xticklabels([dataset_label(s) for s in P["rows"]], fontsize=FS_S, color=INK,
                                               linespacing=1.1)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=2.5, width=0.5, color=AXIS, labelcolor=INK2, labelsize=FS, pad=2)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.set_title(f"{P['ref']} reference", fontsize=FS_H, color=INK, loc="left", pad=5)
    # second label row: the model pair, under a thin bracket spanning its policies
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ah_pt = (FH - B - T) * 72
    yb, up = -BRACKET_PT / ah_pt, 2.0 / ah_pt           # bracket depth and end-tick height (axes fraction)
    for name, i0, i1 in P["groups"]:
        xa_, xb_ = P["cx"][i0] - 0.34, P["cx"][i1] + 0.34
        ax.plot([xa_, xa_, xb_, xb_], [yb + up, yb, yb, yb + up], transform=tr, color=AXIS, lw=0.6,
                clip_on=False, solid_joinstyle="miter")
        ax.annotate(name, ((P["cx"][i0] + P["cx"][i1]) / 2, yb), xycoords=tr, xytext=(0, -1.6),
                    textcoords="offset points", ha="center", va="top", fontsize=FS_S, color=INK)
axes[0].set_ylabel("mean saving per query (ms)", fontsize=FS, color=INK2, labelpad=3)

# ---------------- legend (one row) ----------------
class HandlerWhisker(HandlerBase):
    def create_artists(self, legend, orig_handle, xd, yd, w, h, fontsize, trans):
        xm, cap = xd + w / 2, 1.6
        y0, y1 = yd - 0.25 * h, yd + 1.25 * h
        arts = [Line2D([xm, xm], [y0, y1]), Line2D([xm - cap, xm + cap], [y0, y0]), Line2D([xm - cap, xm + cap], [y1, y1])]
        for a in arts:
            a.set_color(INK2); a.set_linewidth(0.7); a.set_transform(trans)
        return arts
class HandlerTick(HandlerBase):
    def create_artists(self, legend, orig_handle, xd, yd, w, h, fontsize, trans):
        a = Line2D([xd + w * 0.15, xd + w * 0.85], [yd + h / 2] * 2, color=INK, lw=1.4, solid_capstyle="butt")
        a.set_transform(trans); return [a]
class Whisker: pass
class Tick: pass
def rect(**kw): return Rectangle((0, 0), 1, 1, **kw)
hand = [(rect(color=BLUE, lw=0), rect(color=ORANGE, lw=0)),
        (Line2D([], [], marker="o", ms=4, mfc=BLUE, mec="white", mew=0.6, ls="none"),
         Line2D([], [], marker="^", ms=4, mfc=ORANGE, mec="white", mew=0.6, ls="none")),
        (rect(facecolor="white", edgecolor=BLUE, lw=0.6, hatch="////"),
         rect(facecolor="white", edgecolor=ORANGE, lw=0.6, hatch="////")),
        rect(facecolor="white", edgecolor=BLUE, lw=1.0),
        Tick(), Whisker()]
labs = ["original run", "repeats", "argmax answering", "later run", "component estimate", "95% interval"]
leg = fig.legend(hand, labs, loc="lower center", ncol=len(labs), frameon=False, fontsize=FS,
                 handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35), Whisker: HandlerWhisker(),
                              Tick: HandlerTick()},
                 handlelength=1.5, handleheight=0.75, handletextpad=0.4, columnspacing=0.85, borderaxespad=0.15,
                 bbox_to_anchor=(0.5, 0.0))
for t in leg.get_texts():
    t.set_color(INK)

# ---------------- nudge the number labels clear of the argmax bar and whisker ----------------
from matplotlib.collections import LineCollection
from matplotlib.transforms import Bbox
fig.canvas.draw()
rnd = fig.canvas.get_renderer()
PX = fig.dpi / 72.0                                    # display pixels per point
CLEAR = 1.5 * PX
def ext(a):
    if isinstance(a, LineCollection):                  # errorbar stems: extent of the segments (+ line width)
        pts = [a.get_transform().transform(sg) for sg in a.get_segments()]
        e = Bbox.union([Bbox.from_extents(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in pts])
        return e.padded(0.5 * max(a.get_linewidths()) * PX)
    if isinstance(a, Line2D) and a.get_marker() == "_":   # errorbar caps: the drawn segment, not the marker box
        x, y = a.get_transform().transform(a.get_xydata())[0]
        hw, ht = a.get_markersize() / 2 * PX, a.get_markeredgewidth() / 2 * PX
        return Bbox.from_extents(x - hw, y - ht, x + hw, y + ht)
    return a.get_window_extent(rnd)
for ax, t, obst in labels_num:
    for _ in range(40):
        bb = ext(t)
        if not any(bb.padded(CLEAR).overlaps(ext(o)) for o in obst):
            break
        x, y = t.xyann; t.xyann = (x - 0.5, y)
    else:
        raise SystemExit("could not clear label " + t.get_text())
    if t.xyann[0] != 0:
        print(f"nudged label {t.get_text()} left by {-t.xyann[0]:.1f} pt")

# ---------------- collision checks ----------------
fig.canvas.draw()
texts = [t for _, t, _ in labels_num]
for ax in axes:
    texts += [tl for tl in ax.get_xticklabels() + ax.get_yticklabels() if tl.get_visible() and tl.get_text()]
    texts += [a for a in ax.texts if a.get_text()] + [ax.title]
texts = list(dict.fromkeys(texts)) + list(leg.get_texts()) + [axes[0].yaxis.label]
bad = []
for i, a in enumerate(texts):
    for b in texts[i + 1:]:
        if ext(a).padded(1.0 * PX).overlaps(ext(b)):
            bad.append((a.get_text(), b.get_text()))
# number labels vs every data artist of their panel (bars, whiskers, caps, markers, component ticks)
for ax, t, _ in labels_num:
    tb = ext(t).padded(1.0 * PX)
    arts = list(ax.patches) + [l for l in ax.lines if len(l.get_xdata()) and l.get_xdata()[0] != ax.get_xlim()[0]]
    for c in ax.containers:
        if hasattr(c, "lines"):
            arts += list(c.lines[1]) + list(c.lines[2])
    arts = [a for a in arts if ext(a).width < 0.9 * ax.bbox.width]   # skip grid and zero lines
    for a in arts:
        if tb.overlaps(ext(a)):
            bad.append((t.get_text(), type(a).__name__))
# number labels vs the left spine of their panel
for ax, t, _ in labels_num:
    if ext(t).x0 < ax.bbox.x0 + 1.0 * PX:
        bad.append((t.get_text(), "left spine"))
# pair-row labels vs the legend
top_leg = ext(leg).y1
for ax in axes:
    for a in ax.texts:
        if ext(a).y0 < top_leg + 1.0 * PX:
            bad.append((a.get_text(), "legend"))
# text inside the figure
fb = fig.bbox
for a in texts:
    e = ext(a)
    if e.x0 < fb.x0 or e.x1 > fb.x1 or e.y0 < fb.y0 or e.y1 > fb.y1:
        bad.append((a.get_text(), "clipped"))
print("collisions:", bad if bad else "none"); assert not bad
# gaps between neighbouring x tick labels (points)
for ax in axes:
    tl = [t for t in ax.get_xticklabels() if t.get_text()]
    gaps = [(ext(b).x0 - ext(a).x1) / PX for a, b in zip(tl, tl[1:])]
    print("x-label gaps (pt):", [round(v, 1) for v in gaps])
print("points per x unit: %.1f; figure %.2f x %.2f in" % (PT_PER_UNIT, FW, FH))
fig.savefig(OUT, metadata={"CreationDate": None})
print("wrote", OUT)

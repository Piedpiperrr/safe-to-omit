"""Run from any directory: python3 code/figures/appendix/fig07_corrections.py [output.pdf]

Figure 6 (Appendix E, fig:corrections): how much of a helpful reference's gain skipping keeps.

(a) Share of the reference's gain over the receiver that the policy keeps, at
    each tolerance alpha (Table 10, tab:retention); missing points fall back.
(b) Reference corrections (questions the reference answers correctly and the
    receiver alone does not) that each deployed Qwen-pair policy loses by
    skipping or keeps by running the reference, with the net change in correct
    answers (development; descriptive and gold-dependent).

Data: Table 10 (new_app_F.tex) and the released correction record
(record_tables/omit_keep.tex). Checks: skipped counts and the net change
equal Table 4 (tab:dev_full; skipped questions; policy minus reference correct answers);
lost + kept equals the reference corrections counted from the correctness
patterns (record_tables/patterns.tex); gained - lost = net; kappa*alpha in
Table 10 is coverage x alpha; the retained share at alpha = 0.05 is recomputed
from Table 4's correct counts; and the claims made in Appendix E are asserted.
"""
import re, sys, os
from decimal import Decimal, ROUND_HALF_UP
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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "figure_corrections.pdf"
F = open(TEX / "new_app_F.tex").read()
D = open(TEX / "new_app_D.tex").read()
OK = open(REC / "omit_keep.tex").read()
PAT = open(REC / "patterns.tex").read()
def r2(x): return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ---- Table 10 (tab:retention) ----
t = F[F.index("\\label{tab:retention}"):F.index("\\end{table}", F.index("\\label{tab:retention}"))]
blocks = t.split("\\midrule")
names = [("large/OBQA/Text", "large/ARC/Text"), ("large/MMLU-Pro/Text", "large/OBQA/Text+fact")]
ret = {}
for blk, (left, right) in zip(blocks[1:3], names):
    for line in blk.split("\\\\"):
        c = [x.strip() for x in line.split("&")]
        c[0] = c[0].split()[-1] if c[0].split() else ""
        if len(c) == 12 and re.fullmatch(r"0\.0[1-5]", c[0]):
            a = float(c[0])
            for nm, cols in ((left, c[1:6]), (right, c[7:12])):
                q, cov, chg, ka, kept = cols
                if q in ("fb", "fallback"):
                    ret.setdefault(nm, {})[a] = None
                else:
                    assert abs(float(cov) * a - float(ka)) <= 0.0051, (nm, a, cov, ka)   # coverage is printed rounded
                    ret.setdefault(nm, {})[a] = dict(q=float(q), cov=float(cov), kept=float(kept))
assert all(len(v) == 5 for v in ret.values()) and len(ret) == 4
# Table 4 correct counts reproduce the retained share at alpha = 0.05
t7 = D[D.index("\\label{tab:dev_full}"):D.index("\\end{table}", D.index("\\label{tab:dev_full}"))]
dev = {}
for l in t7.split("\\\\"):
    c = [x.strip() for x in l.split("&")]
    if len(c) == 10:
        c[0] = c[0].split()[-1]
        if c[0] in ("small", "medium", "large"):
            s = f"{c[0]}/{c[1]}/{re.sub(r'[$].*?[$]', '', c[2])}"
            p, b = [int(x) for x in c[9].split("/")]
            dev[s] = dict(pol=p, ref=b, chg=c[8], q=0.0 if c[6] == "fallback" else float(c[6]))
Rc = {"large/OBQA": 617, "large/ARC": 268, "large/MMLU-Pro": 1282}   # receiver alone, Table 1
for nm in ("large/OBQA/Text", "large/ARC/Text", "large/MMLU-Pro/Text"):
    d = dev[nm]; R = Rc[nm.rsplit("/", 1)[0]]
    assert f"{(d['pol'] - R) / (d['ref'] - R):.3f}" == f"{ret[nm][0.05]['kept']:.3f}", nm
# claims in Appendix E about Table 10
assert all(ret[n][0.01] is None for n in ret) and ret["large/MMLU-Pro/Text"][0.02] is None
for n in ("large/OBQA/Text", "large/ARC/Text", "large/OBQA/Text+fact"):
    assert ret[n][0.03]["kept"] > ret[n][0.05]["kept"], n
assert ret["large/MMLU-Pro/Text"][0.03]["kept"] < ret["large/MMLU-Pro/Text"][0.05]["kept"]
for n in ret:   # lowering alpha never raises coverage
    covs = [ret[n][a]["cov"] if ret[n][a] else 0.0 for a in (0.01, 0.02, 0.03, 0.04, 0.05)]
    assert covs == sorted(covs), n

# ---- reference corrections ----
corr = {}
for m in re.finditer(r"(\w+/[\w-]+/\w+) & (\d+)/(\d+) & (\d+)/(\d+) & (\d+)/(\d+) & (\d+) & \$([+-]\d+)\$ & (\d+) & (\d+)/(\d+)\\\\", OK):
    g = [int(x) for x in m.groups()[1:]]
    s = m.group(1)
    n, N, lost, tot, kept, tot2, gained, net, other, un_o, un_k = g
    assert tot == tot2 and lost + kept == tot and gained - lost == net, s
    d = dev[s]
    assert d["chg"].split("/")[1] == str(n), (s, d["chg"], n)          # skipped count, Table 4
    assert d["pol"] - d["ref"] == net, (s, d["pol"] - d["ref"], net)     # identity: policy - reference
    corr[s] = dict(n=n, N=N, lost=lost, tot=tot, kept=kept, gained=gained, net=net, un_o=un_o, un_k=un_k)
assert len(corr) == 8
# reference corrections from the correctness patterns (R, Text, C2C)
pat = {}
for m in re.finditer(r"(small|medium|large)/(OBQA|ARC|MMLU-Pro) & (\d+) & " + " & ".join([r"(\d+)"] * 8), PAT):
    v = [int(x) for x in m.groups()[2:]]
    pat[m.group(1) + "/" + m.group(2)] = dict(zip(["N", "000", "100", "010", "001", "110", "101", "011", "111"], v))
for s, c in corr.items():
    pop, ref = s.rsplit("/", 1); p = pat[pop]
    tot = p["010"] + p["011"] if ref == "Text" else p["001"] + p["011"]
    assert tot == c["tot"], (s, tot, c["tot"])
c2c = [c for s, c in corr.items() if s.endswith("C2C")]
assert max(c["lost"] for c in c2c) == 3 and [c["net"] for c in c2c] == [6, 1, 7, 1, 24]
assert (corr["large/OBQA/Text"]["lost"], corr["large/OBQA/Text"]["tot"]) == (13, 46)
assert (corr["large/ARC/Text"]["lost"], corr["large/ARC/Text"]["tot"]) == (6, 10)
assert sum(c["un_k"] > c["un_o"] for c in corr.values()) == 7
print("checks passed")

# ---------------- figure ----------------
from matplotlib.legend_handler import HandlerTuple
FS, FS_H = 8.0, 8.5
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
PASTEL = {"Text": "#c7dbf5", "C2C": "#f9d3c3"}
COL = {"Text": BLUE, "C2C": ORANGE}
LOST, KEPT = "#6f52a3", "#e4ddf0"   # (a): one colour per outcome (violet, not a reference colour), for both references
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#ebeae5", "#c3c2b7"

fig = plt.figure(figsize=(5.5, 2.4))
axA = fig.add_axes([0.625, 0.235, 0.33, 0.605])    # (b): the axis runs on under the end labels
axB = fig.add_axes([0.2, 0.235, 0.26, 0.605])

# (a) share of the gain kept
AL = [0.01, 0.02, 0.03, 0.04, 0.05]
STY = {"large/OBQA/Text": ("#2a78d6", "o", "OBQA"), "large/ARC/Text": ("#8db6ea", "s", "ARC"),
       "large/MMLU-Pro/Text": ("#17467f", "D", "MMLU-Pro"), "large/OBQA/Text+fact": ("#52514e", "^", "Text+fact")}
axA.axhline(1.0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
for v in (0.25, 0.5, 0.75):
    axA.axhline(v, color=GRID, lw=0.6, zorder=0)
LBL_DY = {"large/OBQA/Text": -0.07, "large/OBQA/Text+fact": 0.04, "large/MMLU-Pro/Text": 0.0, "large/ARC/Text": 0.0}
for n, (c, mk, lab) in STY.items():
    xs = [a for a in AL if ret[n][a]]; ys = [ret[n][a]["kept"] for a in xs]
    axA.plot(xs, ys, color=c, lw=1.3, marker=mk, ms=4.2, mec="white", mew=0.5, zorder=3)
    axA.text(0.0515, ys[-1] + LBL_DY[n], lab, ha="left", va="center", fontsize=FS, color=INK)
axA.text(0.0105, 0.07, "all fall back", ha="left", va="bottom", fontsize=FS, color=INK2)
axA.set_xlim(0.0085, 0.0655); axA.set_ylim(0, 1.13)
axA.set_xticks(AL); axA.set_xticklabels(["0.01", "0.02", "0.03", "0.04", "0.05"])
axA.set_yticks([0, 0.5, 1.0]); axA.set_yticklabels(["0", "0.5", "1"])
axA.set_xlabel(r"tolerance $\alpha$", fontsize=FS, color=INK2, labelpad=2)
axA.set_title("(b) Share of Text's gain kept", fontsize=FS_H, color=INK, loc="left", pad=4)

# (b) reference corrections lost and kept
ROWS = [["medium/OBQA/C2C", "medium/ARC/C2C"],
        ["large/OBQA/Text", "large/OBQA/C2C", "large/ARC/Text", "large/ARC/C2C"],
        ["large/MMLU-Pro/Text", "large/MMLU-Pro/C2C"]]
ypos, y = {}, 0.0
for gi, grp in enumerate(ROWS):
    if gi:
        y += 0.5
    for s in grp:
        ypos[s] = y; y += 1.0
H = 0.62
for s, yy in ypos.items():
    c = corr[s]; ref = s.split("/")[-1]
    fl = 100 * c["lost"] / c["tot"]
    axB.barh(yy, fl, height=H, color=LOST, edgecolor="white", lw=1.0, zorder=2)
    axB.barh(yy, 100 - fl, left=fl, height=H, color=KEPT, edgecolor="white", lw=1.0, zorder=2)
    axB.text(103, yy, f"{c['lost']}/{c['tot']}", ha="left", va="center", fontsize=FS, color=INK2)
    axB.text(142, yy, f"{c['net']:+d}".replace("-", "−"), ha="right", va="center", fontsize=FS, color=INK2)
axB.text(103, -0.95, "lost", ha="left", va="center", fontsize=FS, color=INK2)
axB.text(142, -0.95, "net", ha="right", va="center", fontsize=FS, color=INK2)
axB.set_yticks(list(ypos.values())); axB.set_yticklabels(list(ypos.keys()), fontsize=FS, color=INK)
axB.set_ylim(y - 0.5, -1.4)
axB.set_xlim(0, 100); axB.set_xticks([0, 50, 100]); axB.set_xticklabels(["0%", "50%", "100%"])
axB.set_xlabel("share of the reference's corrections", fontsize=FS, color=INK2, labelpad=2)
axB.set_title("(a) Reference corrections lost by skipping", fontsize=FS_H, color=INK, loc="left", pad=4, x=-0.715)

for ax in (axA, axB):
    ax.tick_params(length=2.5, width=0.5, color=AXIS, labelcolor=INK2, labelsize=FS, pad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
axB.tick_params(axis="y", length=0, labelcolor=INK)
axB.spines["left"].set_visible(False)

hand = [Patch(facecolor=LOST), Patch(facecolor=KEPT)]
fig.legend(hand, ["lost: message skipped", "kept: reference run"], loc="lower left", ncol=2, frameon=False, fontsize=FS,
           handlelength=1.2, columnspacing=1.2,
           handletextpad=0.5, bbox_to_anchor=(0.1, 0.0))
fig.savefig(OUT, metadata={"CreationDate": None})
print("wrote", OUT)

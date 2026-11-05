"""Run from any directory: python3 code/figures/appendix/fig06_baselines.py [output.pdf]

Figure 4 (Appendix C, fig:baselines): same-target baselines (post hoc, descriptive).

(a) Fixed receiver-only execution (q = 1): calibration change rate in all 14
    main Qwen-pair settings, against the 5% target; none passes the test.
    Bars grouped by pair; one annotation for the two large-pair ARC settings
    that lie below 5% but fail the exact test.
(b) One small panel per deployed setting (eight), in the accuracy-latency
    plane relative to the fixed reference: x = component-estimated mean
    latency as a share of the reference's, y = development correct answers
    minus the reference in percentage points. Filled marker = certified
    policy, open diamond = receiver alone, dotted line = random skipping at
    any coverage (it lies on the line from receiver alone to the reference),
    ring = random skipping at the policy's coverage (expected count), black
    square = the historical learned predictor (large/OBQA/Text; its latency
    from that stage's accounting, stated in Appendix C).

Data: the same-target baseline table (record_tables/baselines.tex; records in
results/e00_baselines/) and the learned-predictor sentences of Appendix C.
Checks: p-values recomputed from the counts; policy and reference correct
counts, deployed q, and coverage agree with Table 4 (tab:dev_full,
new_app_D.tex); receiver-alone correct counts agree with Table 1 of the main
text; the random expected counts are recomputed from Table 4's coverage; the
claims stated in Appendix C are asserted; and the per-grid-point probability
that a random subset passes the test is recomputed for the large and small
pairs from the released calibration records.
"""
import csv, re, sys, os
from scipy.stats import binom, hypergeom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

from pathlib import Path
HERE = Path(__file__).resolve().parent            # code/figures/appendix
ROOT = HERE.parents[2]                             # repository root
TEX = HERE / "paper_tex"                           # appendix LaTeX of the submitted paper
REC = HERE / "record_tables"                       # fuller tables that the appendix condenses
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "figure_baselines.pdf"
TB = open(REC / "baselines.tex").read()
D = open(TEX / "new_app_D.tex").read()
T1 = open(TEX / "tab_main_complementarity.tex").read()

part1 = TB[TB.index("(i) Fixed"):TB.index("(ii) Component")]
part2 = TB[TB.index("(ii) Component"):]
base = {}
for m in re.finditer(r"(\w+/[\w-]+/\w+) & (\d+)/(\d+) & (\$>0\.99\$|[0-9.]+) & (\d+) & ([0-9.]+)(\$\^\\dagger\$)? & (\d+) & (--|[0-9.]+)\\\\", part1):
    s = m.group(1)
    base[s] = dict(k=int(m.group(2)), N=int(m.group(3)), p=m.group(4), R=int(m.group(5)), pol=float(m.group(6)),
                   fb=bool(m.group(7)), ref=int(m.group(8)), rnd=None if m.group(9) == "--" else float(m.group(9)))
for m in re.finditer(r"(\w+/[\w-]+/\w+) & ([0-9.]+) & ([0-9.]+) & (--|[0-9.]+) & (--|[0-9.]+)\\\\", part2):
    b = base[m.group(1)]
    b.update(lat_ref=float(m.group(2)), lat_R=float(m.group(3)),
             lat_pol=None if m.group(4) == "--" else float(m.group(4)),
             lat_rnd=None if m.group(5) == "--" else float(m.group(5)))
assert len(base) == 14 and all("lat_ref" in b for b in base.values())

# ---- Table 4 (tab:dev_full, development record) ----
t7 = D[D.index("\\label{tab:dev_full}"):D.index("\\end{table}", D.index("\\label{tab:dev_full}"))]
dev = {}
for l in t7.split("\\\\"):
    c = [x.strip() for x in l.split("&")]
    c[0] = c[0].split()[-1] if c[0] else c[0]
    if len(c) == 10 and c[0] in ("small", "medium", "large"):
        s = f"{c[0]}/{c[1]}/{re.sub(r'[$].*?[$]', '', c[2])}"
        pol, ref = [int(x) for x in c[9].split("/")]
        dev[s] = dict(q=0.0 if c[6] == "fallback" else float(c[6]), cov=float(c[7]), chg=c[8], pol=pol, ref=ref)
assert set(dev) == set(base), set(base) ^ set(dev)

# ---- Table 1 of the main text: receiver-alone correct answers ----
t1key = {"small/OBQA": "small & OBQA", "small/ARC": " & ARC (299) & 110", "medium/OBQA": "medium & OBQA",
         "medium/ARC": " & ARC (299) & 217", "large/OBQA": "large & OBQA", "large/ARC": " & ARC (299) & 268",
         "large/MMLU-Pro": " & MMLU-Pro"}
Rt1 = {}
for pop, key in t1key.items():
    line = [l for l in T1.split("\\\\") if key in l][0]
    Rt1[pop] = int([re.sub(r"\$.*?\$", "", x).strip() for x in line.split("&")][2])

DEPLOYED = []
for s, b in base.items():
    d = dev[s]; pop = s.rsplit("/", 1)[0]
    p = binom.cdf(b["k"], b["N"], 0.05)
    assert (b["p"] == "$>0.99$" and p > 0.99) or abs(p - float(b["p"])) < 5e-4, (s, p, b["p"])
    assert p > 0.001, s                                             # fixed R fails everywhere
    assert (b["pol"], b["ref"]) == (d["pol"], d["ref"]), (s, b, d)  # Table 4 Correct P/B
    assert b["R"] == Rt1[pop], (s, b["R"], Rt1[pop])                # Table 1 receiver alone
    assert b["fb"] == (d["q"] == 0) == (b["rnd"] is None) == (b["lat_pol"] is None), s
    if d["q"]:
        k, n = [int(x) for x in d["chg"].split("/")]
        N = 742 if "OBQA" in s else 299 if "ARC" in s else 2641
        kap = n / N
        assert abs(round((1 - kap) * b["ref"] + kap * b["R"], 1) - b["rnd"]) < 0.051, (s, b["rnd"])
        assert abs(100 * kap - d["cov"]) < 0.051, s
        DEPLOYED.append(s)
assert len(DEPLOYED) == 8
rate = {s: 100 * b["k"] / b["N"] for s, b in base.items()}
assert f"{min(rate.values()):.1f}" == "2.7" and f"{max(rate.values()):.1f}" == "56.7"
assert [base[f"large/ARC/{r}"]["p"] for r in ("Text", "C2C")] == ["0.011", "0.435"]
assert (f"{min(rate[s] for s in DEPLOYED):.1f}", f"{max(rate[s] for s in DEPLOYED):.1f}") == ("2.7", "29.1")
above = sorted(s for s in DEPLOYED if base[s]["rnd"] > base[s]["pol"])
assert above == ["large/ARC/C2C", "large/OBQA/C2C", "medium/ARC/C2C"], above
gap = [base[s]["lat_pol"] - base[s]["lat_rnd"] for s in DEPLOYED]
assert (round(min(gap)), round(max(gap))) == (18, 67), gap
# historical learned predictor (large/OBQA/Text): 230 skipped, 6 changed, 643 correct
for snip in ("It skipped 230 of 742 development questions with 6 changes and 643 correct answers",
             "against 572 skipped, 19 changes, 638 correct, and 460.9~ms for ProbeMax"):
    assert snip in D, snip
LEARNED = 643

# ---- probability that a random subset passes, large and small pairs (released records) ----
cal = list(csv.DictReader(open(ROOT / "results/boundaries_large_small/summary/calibration_all_160.csv")))
worst = {}
for key in {(r["pair"], r["dataset"], r["reference"]) for r in cal}:
    rows = [r for r in cal if (r["pair"], r["dataset"], r["reference"]) == key]
    Ncal = int(rows[0]["N"]); K = int(next(r for r in rows if float(r["q"]) == 1.0)["changed"])
    best = (0.0, -1.0)
    for r in rows:
        n = int(r["n_R"])
        if n == Ncal:
            continue
        kmax = max([k for k in range(n + 1) if binom.cdf(k, n, 0.05) <= 0.001], default=-1)
        pr = hypergeom.cdf(kmax, Ncal, K, n) if kmax >= 0 else 0.0
        best = max(best, (pr, float(r["q"])))
    worst[key] = best
lt = worst[("large", "arc", "T")]
assert round(lt[0], 3) == 0.017 and lt[1] == 0.8, lt
assert all(v[0] < 1e-4 for k, v in worst.items() if k != ("large", "arc", "T")), worst
print("checks passed; random-subset pass probability, large/ARC/Text: %.4f at q=%.2f" % lt)

# ---------------- figure ----------------

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch


# the learned predictor's component latency is only stated in Appendix C, in the earlier stage's accounting
m = re.search(r"It skipped 230 of 742 development questions with 6 changes and 643 correct answers "
              r"\(component latency ([0-9.]+)~ms in that stage's accounting\)", D)
assert m, "learned-predictor latency sentence not found"
LAT_LEARNED = float(m.group(1))

# ---------------- style ----------------
FS, FS_H = 8.0, 8.5
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
COL = {"Text": BLUE, "C2C": ORANGE}; MK = {"Text": "o", "C2C": "^"}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#ebeae5", "#c3c2b7"
PASS_BAND, PASS_LINE, PASS_INK = "#e3f4ea", "#7cc49a", "#3d8f5f"   # the certified region of Figure 2
RANGE, LEAD = "#cfcec7", "#9a9993"
PATH = "#cfcec7"
REFLS = (0, (3, 2))

# ---------------- layout (inches, converted to figure fractions) ----------------
W_IN, H_IN = 5.5, 4.3
fig = plt.figure(figsize=(W_IN, H_IN))
A_HIN = 0.74                                  # panel (a) height (in)
A_TOP = 0.18                                  # whitespace above panel (a)'s title (in)
axA = fig.add_axes([0.07, 1 - (A_TOP + A_HIN) / H_IN, 0.92, A_HIN / H_IN])

L_IN, R_IN, GAPX = 0.42, 0.04, 0.16           # left margin (y label + ticks), right margin, column gap
ROW1_TOP, ROW3_BOT, GAPY = 2.74, 0.38, 0.19   # top of row 1, bottom of row 3, row gap (room for titles)
NC, NR = 3, 3
PW = (W_IN - L_IN - R_IN - (NC - 1) * GAPX) / NC
PH = (ROW1_TOP - ROW3_BOT - (NR - 1) * GAPY) / NR
ORDER = ["medium/OBQA/C2C", "medium/ARC/C2C", "large/OBQA/Text", "large/OBQA/C2C",
         "large/ARC/Text", "large/ARC/C2C", "large/MMLU-Pro/Text", "large/MMLU-Pro/C2C"]
assert sorted(ORDER) == sorted(DEPLOYED)
axes = {}
def cell(i):
    r, c = divmod(i, NC)
    x0 = L_IN + c * (PW + GAPX)
    y0 = ROW3_BOT + (NR - 1 - r) * (PH + GAPY)
    return [x0 / W_IN, y0 / H_IN, PW / W_IN, PH / H_IN]
for i, s_ in enumerate(ORDER):
    axes[s_] = fig.add_axes(cell(i))

# ---------------- (a): receiver alone, calibration change rate ----------------
# (a) receiver alone, calibration change rate: bars grouped by pair, benchmarks under each bar
GROUPS = [("small", ["OBQA", "ARC"]), ("medium", ["OBQA", "ARC"]), ("large", ["OBQA", "ARC", "MMLU-Pro"])]
W, GAPG = 0.36, 0.55
xs, x = {}, 0.0
for gi, (pair, bms) in enumerate(GROUPS):
    if gi:
        x += GAPG
    for bm in bms:
        xs[f"{pair}/{bm}"] = x; x += 1.0
axA.axhspan(0, 5, color=PASS_BAND, lw=0, zorder=0)
axA.axhline(5, color=PASS_LINE, lw=0.9, ls=(0, (3, 2)), zorder=3)
for v in (20, 40, 60):
    axA.axhline(v, color=GRID, lw=0.6, zorder=0)
for pop, xc in xs.items():
    for ref, dx in (("Text", -W / 2 - 0.01), ("C2C", W / 2 + 0.01)):
        r = rate[f"{pop}/{ref}"]
        axA.bar(xc + dx, r, width=W, color=COL[ref], edgecolor="none", zorder=2)
        axA.text(xc + dx, r + 1.2, f"{r:.1f}", ha="center", va="bottom", fontsize=FS, color=INK2, zorder=4)
# the two large/ARC rates are below 5% but fail the test: one ink label with a leader to the pair
xa = xs["large/ARC"]
lab = axA.text(xa - 0.62, 33.0, "below 5%, yet not certified:\n$p=0.011$ (Text), $0.435$ (C2C)",
               ha="center", va="center", fontsize=FS, color=INK, linespacing=1.15, zorder=5)
axA.annotate("", xy=(xa, 11.0), xytext=(xa - 0.35, 25.0),
             arrowprops=dict(arrowstyle="-", color=LEAD, lw=0.5, shrinkA=0, shrinkB=0), zorder=4)
axA.set_xticks(list(xs.values()))
axA.set_xticklabels([p.split("/")[1] for p in xs], fontsize=FS, color=INK)
xmin, xmax = -0.55, x - 0.45
axA.set_xlim(xmin, xmax); axA.set_ylim(0, 66)
# pair names under their benchmarks, with a thin bracket
for pair, bms in GROUPS:
    x0, x1 = xs[f"{pair}/{bms[0]}"] - 0.42, xs[f"{pair}/{bms[-1]}"] + 0.42
    yb = -0.20
    axA.plot([x0, x1], [yb, yb], color=AXIS, lw=0.6, transform=axA.get_xaxis_transform(), clip_on=False)
    axA.text((x0 + x1) / 2, yb - 0.03, f"{pair} pair", ha="center", va="top", fontsize=FS, color=INK2,
             transform=axA.get_xaxis_transform())
axA.set_yticks([0, 20, 40, 60]); axA.set_yticklabels(["0%", "20%", "40%", "60%"])
axA.set_title("(a) Receiver alone: calibration change rate; no setting passes the test", fontsize=FS_H,
              color=INK, loc="left", pad=4)
target = (Patch(facecolor=PASS_BAND, edgecolor="none"), Line2D([], [], color=PASS_LINE, lw=0.9, ls=(0, (3, 2))))
axA.legend(handles=[Patch(facecolor=COL["Text"], label="Text"), Patch(facecolor=COL["C2C"], label="C2C"), target],
           labels=["Text", "C2C", "5% target"], loc="upper right", ncol=3, frameon=False, fontsize=FS,
           handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)}, handlelength=1.1, handletextpad=0.4,
           columnspacing=1.0, borderaxespad=0.0)
# (a) axis styling, as in the tkw4 script
axA.tick_params(length=2.5, width=0.5, color=AXIS, labelcolor=INK2, labelsize=FS, pad=2)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    axA.spines[sp].set_color(AXIS)
axA.tick_params(axis="x", length=0, labelcolor=INK)
axA.spines["left"].set_visible(False); axA.tick_params(axis="y", length=0)

# ---------------- (b): accuracy-latency plane, one small panel per deployed setting ----------------
NDEV = {"OBQA": 742, "ARC": 299, "MMLU-Pro": 2641}
XLIM, YLIM = (16, 105), (-4.6, 5.8)
MS_POL, MS_R, MS_RND = 6.4, 5.0, 3.3
RNDLS = (0, (1.2, 1.6))
allx, ally = [], []
for i, s_ in enumerate(ORDER):
    ax = axes[s_]; r, c = divmod(i, NC)
    b = base[s_]; ref = s_.split("/")[-1]; N = NDEV[s_.split("/")[1]]
    pts = {"R": 100 * (b["R"] - b["ref"]) / N, "pol": 100 * (b["pol"] - b["ref"]) / N,
           "rnd": 100 * (b["rnd"] - b["ref"]) / N}
    lat = {"R": 100 * b["lat_R"] / b["lat_ref"], "pol": 100 * b["lat_pol"] / b["lat_ref"],
           "rnd": 100 * b["lat_rnd"] / b["lat_ref"]}
    allx += list(lat.values()); ally += list(pts.values())
    # grid, then the reference crosshair
    for xx in (25, 50, 75):
        ax.axvline(xx, color=GRID, lw=0.6, zorder=0)
    for yy in (-4, -2, 2, 4):
        ax.axhline(yy, color=GRID, lw=0.6, zorder=0)
    ax.axhline(0, color=INK2, lw=0.6, ls=REFLS, zorder=1.5)
    ax.axvline(100, color=INK2, lw=0.6, ls=REFLS, zorder=1.5)
    # random skipping at any coverage lies on the straight line from receiver alone to the reference:
    # dotted line, with a small bead at the policy's coverage
    ax.plot([lat["R"], 100], [pts["R"], 0], color=GRAY, lw=0.8, ls=RNDLS, zorder=1.2,
            solid_capstyle="round", dash_capstyle="round")
    ax.plot([lat["R"]], [pts["R"]], marker="D", ms=MS_R, mfc="white", mec=GRAY, mew=1.0, ls="none", zorder=3)
    ax.plot([lat["pol"]], [pts["pol"]], marker=MK[ref], ms=MS_POL, mfc=COL[ref], mec="white", mew=0.6,
            ls="none", zorder=4)
    ax.plot([lat["rnd"]], [pts["rnd"]], marker="o", ms=MS_RND, mfc="white", mec=INK2, mew=0.9, ls="none",
            zorder=6)
    ax.plot([100], [0], marker="+", ms=6.0, mec=INK, mew=1.0, ls="none", zorder=4.5)
    if s_ == "large/OBQA/Text":
        xl, yl = 100 * LAT_LEARNED / b["lat_ref"], 100 * (LEARNED - b["ref"]) / N
        allx.append(xl); ally.append(yl)
        ax.plot([xl], [yl], marker="s", ms=4.6, mfc=INK, mec="white", mew=0.4, ls="none", zorder=5)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks([25, 50, 75, 100]); ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_xticklabels(["25%", "50%", "75%", "100%"]); ax.set_yticklabels(["$-4$", "$-2$", "0", "+2", "+4"])
    ax.tick_params(length=2.5, width=0.5, color=AXIS, labelcolor=INK2, labelsize=FS, pad=2,
                   labelbottom=(r == NR - 1 or i == 7), labelleft=(c == 0))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS); ax.spines[sp].set_linewidth(0.6)
    ax.set_title(s_, fontsize=FS, color=INK, loc="left", pad=2.5)
# every drawn point lies inside the shared window
assert XLIM[0] < min(allx) and max(allx) < XLIM[1] and YLIM[0] < min(ally) and max(ally) < YLIM[1]

# heading for (b), axis labels once
fig.text(0.07, (ROW1_TOP + 0.30) / H_IN, "(b) Deployed settings: development accuracy against latency, "
         "each relative to the reference", fontsize=FS_H, color=INK, ha="left", va="top")
gx0, gx1 = L_IN / W_IN, (W_IN - R_IN) / W_IN
fig.text((gx0 + gx1) / 2, 0.06 / H_IN, "latency, share of the reference (component estimate)",
         fontsize=FS, color=INK2, ha="center", va="bottom")
fig.text(0.04 / W_IN, (ROW3_BOT + ROW1_TOP) / 2 / H_IN, "correct answers minus reference (pp)",
         fontsize=FS, color=INK2, ha="left", va="center", rotation=90)

# legend in the empty ninth cell
axL = fig.add_axes(cell(8)); axL.axis("off")
pol = (Line2D([], [], color=BLUE, marker="o", ms=MS_POL, mec="white", mew=0.6, lw=0),
       Line2D([], [], color=ORANGE, marker="^", ms=MS_POL, mec="white", mew=0.6, lw=0))
hand = [pol,
        Line2D([], [], color=GRAY, marker="D", ms=MS_R, mfc="white", mew=1.0, lw=0),
        Line2D([], [], color=GRAY, lw=0.8, ls=RNDLS, marker="o", ms=MS_RND, mfc="white", mec=INK2, mew=0.9),
        Line2D([], [], color=INK, marker="s", ms=4.6, mec="white", mew=0.4, lw=0),
        Line2D([], [], color=INK2, lw=0.6, ls=REFLS, marker="+", ms=6.0, mec=INK, mew=1.0)]
labels = ["certified policy", "receiver alone",
          "random skipping (dotted line:\nany coverage; ring: the policy's)",
          "learned predictor", "reference"]
axL.legend(hand, labels, loc="center left", ncol=1, frameon=False, fontsize=FS,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
           handlelength=1.5, handletextpad=0.5, labelspacing=0.6, bbox_to_anchor=(-0.06, 0.5),
           borderaxespad=0.0)
fig.savefig(OUT, metadata={"CreationDate": None})
print("wrote", OUT, "panel size %.2f x %.2f in" % (PW, PH))

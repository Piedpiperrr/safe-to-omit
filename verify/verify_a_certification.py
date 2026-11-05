"""Verification (a): re-run the certification test from the released calibration
ledgers and check that it reproduces the deployed q of three deployments.

Rule (configs/confidence_boundaries/frozen_config.json): a 20-point fit-quantile
grid; at each q an exact one-sided binomial test on the questions routed to the
reference; q is accepted when p <= p_cutoff; the deployed q is the largest
accepted q, else the fixed reference (q = 0).

Run from the repository root:  python3 verify/verify_a_certification.py
"""
import csv, json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p), computed with an iterative pmf."""
    if n == 0:
        return 1.0
    pmf = (1.0 - p) ** n
    total = pmf
    for i in range(1, k + 1):
        pmf *= (n - i + 1) / i * p / (1.0 - p)
        total += pmf
    return min(total, 1.0)

def col(row, *names):
    for n in names:
        if n in row:
            return row[n]
    raise KeyError(names)

def certify(rows, alpha, p_cutoff):
    table, accepted = [], []
    for r in rows:
        q   = float(col(r, "q"))
        nR  = int(col(r, "n_R", "routed"))
        chg = int(col(r, "changed"))
        p   = binom_cdf(chg, nR, alpha)
        ok  = (p <= p_cutoff)
        stored = col(r, "accepted") if "accepted" in r else None
        table.append((q, nR, chg, p, ok, stored))
        if ok:
            accepted.append(q)
    return (max(accepted) if accepted else 0.0), table

cfg   = json.load(open(os.path.join(HERE, "configs/confidence_boundaries/frozen_config.json")))
ALPHA = cfg["alpha"]; PCUT = cfg["p_cutoff"]
MED   = "results/boundaries_medium/execution_retry1_20260915T164957Z"

CASES = [
    ("large/OBQA/Text",
     "configs/confidence_boundaries/deployments/large_obqa_T_tests.csv",
     ("json", "configs/confidence_boundaries/deployments/large_obqa_T.json")),
    ("large/ARC/Text",
     "configs/confidence_boundaries/deployments/large_arc_T_tests.csv",
     ("json", "configs/confidence_boundaries/deployments/large_arc_T.json")),
    ("medium/OBQA/C2C",
     MED + "/calibration/obqa_C.csv",
     ("freeze", MED + "/ALL_DEPLOYMENTS_FREEZE.json", "obqa_C")),
]

def deployed_q(spec):
    if spec[0] == "json":
        return json.load(open(os.path.join(HERE, spec[1])))["q"]
    d = json.load(open(os.path.join(HERE, spec[1])))["deployments"][spec[2]]
    return d["q"]

print("Verification (a) - certification test from released calibration records")
print("alpha = %s   p_cutoff = %s   grid = %d points\n" % (ALPHA, PCUT, len(cfg["q_grid"])))
allok = True
for name, ledger, spec in CASES:
    lp = os.path.join(HERE, ledger)
    if not os.path.exists(lp):
        print("%-17s MISSING %s" % (name, ledger)); allok = False; continue
    rows = list(csv.DictReader(open(lp)))
    recomputed, table = certify(rows, ALPHA, PCUT)
    stored = deployed_q(spec)
    ok = abs(recomputed - float(stored)) < 1e-12
    allok &= ok
    dis = sum(1 for q, n, c, p, a, s in table
              if s is not None and a != (str(s).strip().lower() == "true"))
    print("%-17s grid=%2d  accepted=%2d  recomputed q=%-5s deployed q=%-5s  %s"
          % (name, len(rows), sum(1 for t in table if t[4]), recomputed, stored,
             "PASS" if ok else "FAIL"))
    print("%-17s per-test accept flags disagreeing with recomputation: %d/%d"
          % ("", dis, len(table)))
print("\nRESULT: %s" % ("PASS - all three deployed q reproduced from the released ledgers"
                        if allok else "FAIL"))

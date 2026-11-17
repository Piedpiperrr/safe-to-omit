"""Verification (d): re-derive the prospective three-agent test (QASC) from its
released per-question exports: the three calibrated policies and the sealed-test
row of the primary policy (Table 3).

Procedure (frozen protocol, results/three_agent_qasc/PROTOCOL.md):
- candidate j = 1..19 is the ceil(j * N_fit / 20)-th smallest fit-split score u;
  j = 20 skips every question with a finite score; a question is skipped when u <= tau_j;
- change-rate test on the calibration split: pass when P[Bin(n, 0.05) <= m] <= 0.001
  (n skipped, m answer changes among them; n = 0 fails);
- lost-correction test: pass when P[Bin(N_cal, eps) <= l] <= 0.001 (l: skipped questions
  the reference answers correctly and the cheap path does not). Gold labels are used only
  where the label rule queried them: candidates are scanned in increasing tau, eps = 0.02
  first, each eps stops at its first failing candidate, and no label is used beyond the
  largest candidate that passes the change-rate test;
- label-free policy: largest candidate passing the change-rate test; joint policies:
  largest candidate passing both tests at eps = 0.01 (primary) and eps = 0.02.

Inputs (also in the dataset under results/three_agent_qasc/recheck/):
  results/three_agent_qasc/recheck/fit.csv      fit-split scores
  results/three_agent_qasc/recheck/cal.csv      calibration scores, answers, queried labels
  results/three_agent_qasc/recheck/sealed.csv   sealed-test scores, answers and gold labels

Run from the repository root:  python3 verify/verify_d_three_agent.py
"""
import csv, math, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA, DELTA, NCAND = 0.05, 0.001, 20
EXP_J = {"label-free": 11, "joint, eps=0.01": 8, "joint, eps=0.02": 10}
EXP_LABELS = 40
EXP_SEALED = dict(N=2000, n=798, m=5, l=5, acc_s=0.6380, acc_b=0.7660, acc_pi=0.7635)


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


def find(*c):
    for x in c:
        p = os.path.join(HERE, x)
        if os.path.exists(p):
            return p
    raise SystemExit("missing input: " + c[0])


def rows(name):
    p = find("results/three_agent_qasc/recheck/" + name,
             "data/results/three_agent_qasc/recheck/" + name)
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def score(r):
    x = float(r["u"])
    return x if math.isfinite(x) else None


fit, cal, sealed = rows("fit.csv"), rows("cal.csv"), rows("sealed.csv")

# candidate thresholds from the fit split
u_fit = sorted(score(r) for r in fit if score(r) is not None)
tau = {j: u_fit[math.ceil(j * len(u_fit) / NCAND) - 1] for j in range(1, NCAND)}
tau[NCAND] = math.inf


def skipped(r, j):
    u = score(r)
    return u is not None and u <= tau[j]


# calibration: change-rate test at every candidate
N = len(cal)
change_pass = {}
for j in range(1, NCAND + 1):
    sk = [r for r in cal if skipped(r, j)]
    n, m = len(sk), sum(r["ans_s"] != r["ans_b"] for r in sk)
    change_pass[j] = n > 0 and binom_cdf(m, n, ALPHA) <= DELTA
label_free = max([j for j in change_pass if change_pass[j]], default=None)

# lost-correction test with the label rule
labelled = set()
joint = {}
for eps in (0.02, 0.01):
    best = None
    for j in range(1, (label_free or 0) + 1):
        need = [r for r in cal if skipped(r, j) and r["ans_s"] != r["ans_b"]]
        missing = [r["id"] for r in need if r["gold"] == ""]
        if missing:
            raise SystemExit("label rule needs %d labels missing from cal.csv" % len(missing))
        labelled.update(r["id"] for r in need)
        l = sum(r["ans_b"] == r["gold"] and r["ans_s"] != r["gold"] for r in need)
        if binom_cdf(l, N, eps) > DELTA:
            break
        if change_pass[j]:
            best = j
    joint[eps] = best

policies = {"label-free": label_free, "joint, eps=0.01": joint[0.01], "joint, eps=0.02": joint[0.02]}

print("Verification (d) - prospective three-agent test (QASC)\n")
print("  calibration questions: %d   fit questions: %d" % (N, len(fit)))
ok = True
for name, j in policies.items():
    good = j == EXP_J[name]
    ok &= good
    print("  %-17s j=%-3s (expected %d)   calibration skip %.1f%%   %s"
          % (name, j, EXP_J[name], 100 * sum(skipped(r, j) for r in cal) / N if j else 0.0,
             "PASS" if good else "FAIL"))
print("  gold labels used by the rule: %d   (expected %d)" % (len(labelled), EXP_LABELS))
ok &= len(labelled) == EXP_LABELS

# sealed test, primary policy
j = policies["joint, eps=0.01"]
Ns = len(sealed)
cs = [r["ans_s"] == r["gold"] for r in sealed]
cb = [r["ans_b"] == r["gold"] for r in sealed]
sk = [skipped(r, j) for r in sealed]
n = sum(sk)
m = sum(s and r["ans_s"] != r["ans_b"] for s, r in zip(sk, sealed))
l = sum(s and b and not a for s, a, b in zip(sk, cs, cb))
acc_s, acc_b = sum(cs) / Ns, sum(cb) / Ns
acc_pi = sum(a if s else b for s, a, b in zip(sk, cs, cb)) / Ns
print("\n  sealed test, primary policy (joint, eps = 0.01):")
print("  questions            : %d   (expected %d)" % (Ns, EXP_SEALED["N"]))
print("  skipped              : %d = %.1f%%   (expected %d)" % (n, 100 * n / Ns, EXP_SEALED["n"]))
print("  changed among skipped: %d/%d   (expected %d/%d)" % (m, n, EXP_SEALED["m"], EXP_SEALED["n"]))
print("  lost corrections     : %d   (expected %d)" % (l, EXP_SEALED["l"]))
print("  accuracy A+C / A+B+C / policy: %.2f / %.2f / %.2f%%"
      % (100 * acc_s, 100 * acc_b, 100 * acc_pi))
print("  gain of B: %.2f points   gain kept by the policy: %.2f points"
      % (100 * (acc_b - acc_s), 100 * (acc_pi - acc_s)))
ok &= (Ns, n, m, l) == (EXP_SEALED["N"], EXP_SEALED["n"], EXP_SEALED["m"], EXP_SEALED["l"])
ok &= all(abs(x - EXP_SEALED[k]) < 5e-5 for k, x in
          (("acc_s", acc_s), ("acc_b", acc_b), ("acc_pi", acc_pi)))
print("\nRESULT: %s" % ("PASS - three policies, 40 labels and 798 / 5 / 5 / 12.55 reproduced"
                        if ok else "FAIL"))

"""Verification (c): rebuild one Table 2 row - large/ARC/Text - from the released
per-question development records, and check it gives 95.0% skipped and 9/284 changed.

Routing rule (configs/confidence_boundaries/frozen_config.json, mode "selective"):
a question is answered by the receiver alone (the reference call is skipped) when
its ProbeMax is at or below the frozen deployed threshold; otherwise the reference
is called. "changed" counts the skipped questions whose stored reference answer
differs from the stored receiver-only answer - the conditional risk of skipping.

Inputs (records/ comes from the dataset: run `make data` first):
  configs/confidence_boundaries/deployments/large_arc_T.json        q and threshold
  splits/development_and_calibration/arc_dev_representatives.json   dev ids
  records/boundaries_large_small/records/large_arc_dev_probes.jsonl.gz   ProbeMax
  records/answer_parser_v2/labels/full_development_P2_SCORING_V2.jsonl.gz  o_R, o_T

Run from the repository root:  python3 verify/verify_c_table2_row.py
"""
import json, gzip, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR, DS, INV = "large", "arc", "INVALID"
EXP_COVERAGE, EXP_CHANGED, EXP_NR, EXP_N = 0.9498327759197325, 9, 284, 299

def jl(p):
    op = gzip.open(p, "rt") if p.endswith(".gz") else open(p)
    with op as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)

def find(*c):
    for x in c:
        p = os.path.join(HERE, x)
        if os.path.exists(p):
            return p
    raise SystemExit("missing input: " + c[0])

dep = json.load(open(find("configs/confidence_boundaries/deployments/large_arc_T.json")))
q, thr, mode = dep["q"], dep["threshold"], dep["mode"]

ids = json.load(open(find("splits/development_and_calibration/arc_dev_representatives.json")))

probe = {}
for r in jl(find("records/boundaries_large_small/records/large_arc_dev_probes.jsonl.gz",
                 "records/boundaries_large_small/records/large_arc_dev_probes.jsonl")):
    if r.get("pair") == PAIR and r.get("dataset") == DS and r.get("split") == "dev":
        probe[r["id"]] = r["ProbeMax"]

lab = {}
for r in jl(find("records/answer_parser_v2/labels/full_development_P2_SCORING_V2.jsonl.gz",
                 "records/answer_parser_v2/labels/full_development_P2_SCORING_V2.jsonl")):
    if r.get("pair") == PAIR and r.get("dataset") == DS:
        lab[r["id"]] = r

miss = [i for i in ids if i not in probe or i not in lab]
if miss:
    raise SystemExit("missing probe/label rows for %d dev ids" % len(miss))

n_R = changed = 0
for i in ids:
    skipped = (probe[i] <= thr)          # confident -> receiver alone, no reference call
    if not skipped:
        continue
    n_R += 1
    oR = lab[i]["o_R"] or INV
    oT = lab[i]["o_T"] or INV
    if oR != oT:
        changed += 1

N = len(ids)
cov = n_R / N
print("Verification (c) - Table 2 row, large/ARC/Text (mode=%s, q=%s)\n" % (mode, q))
print("  threshold (frozen)   : %r" % thr)
print("  N dev questions      : %d   (expected %d)" % (N, EXP_N))
print("  skipped (receiver only): %d   (expected %d)" % (n_R, EXP_NR))
print("  coverage / %% skipped  : %.4f  = %.1f%%   (expected %.1f%%)"
      % (cov, 100 * cov, 100 * EXP_COVERAGE))
print("  changed among skipped: %d/%d   (expected %d/%d)" % (changed, n_R, EXP_CHANGED, EXP_NR))
print("  conditional risk     : %.6f" % (changed / n_R if n_R else float("nan")))
ok = (N == EXP_N and n_R == EXP_NR and changed == EXP_CHANGED
      and abs(cov - EXP_COVERAGE) < 1e-12)
print("\nRESULT: %s" % ("PASS - 95.0% skipped and 9/284 changed reproduced"
                        if ok else "FAIL"))

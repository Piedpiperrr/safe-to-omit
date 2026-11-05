"""Verification (b): recompute the E22-1 / E9-a per-reference unstratified ratio
for large/OBQA/Text from the released per-question records and check it is 0.547.

The null model and the draw are NOT reimplemented: this script imports E9-a's own
`Inst` class from code/analysis/e09a/scripts/e9a_common.py and feeds it arrays
built from the released records, exactly as E9-a's Pop.__init__ does.

Inputs (records/ comes from the dataset: run `make data` first):
  splits/development_and_calibration/obqa_dev_representatives.json   dev ids, in order
  records/answer_parser_v2/labels/full_development_P2_SCORING_V2.jsonl.gz
                                                 gold, o_R, o_T, o_C (parser V2)
  records/e01_e02_e04/results/e4/large_obqa__V0-V1.jsonl.gz , __V2.jsonl.gz
                                                 V1 / V2 null-control answers

Run from the repository root:  python3 verify/verify_b_e22_ratio.py
"""
import json, gzip, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "code/analysis/e09a/scripts"))
sys.path.insert(0, os.path.join(HERE, "code/analysis/e05_change_decomposition/scripts"))
sys.path.insert(0, os.path.join(HERE, "code/analysis/r1_cpu_prep/src"))
sys.dont_write_bytecode = True
from e9a_common import Inst, NREP           # E9-a's own null model, unchanged
INV = "INVALID"
PAIR, DS, EXPECTED = "large", "obqa", 0.547

def jl(path):
    op = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    with op as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)

def find(*cands):
    for c in cands:
        p = os.path.join(HERE, c)
        if os.path.exists(p):
            return p
    raise SystemExit("missing input: " + cands[0])

ids = json.load(open(find("splits/development_and_calibration/obqa_dev_representatives.json")))

lab = {}
for r in jl(find("records/answer_parser_v2/labels/full_development_P2_SCORING_V2.jsonl.gz",
                 "records/answer_parser_v2/labels/full_development_P2_SCORING_V2.jsonl")):
    if r.get("pair") == PAIR and r.get("dataset") == DS:
        lab[r["id"]] = r

NULL = {"V1": {}, "V2": {}}
for fn in ("large_obqa__V0-V1.jsonl.gz", "large_obqa__V2.jsonl.gz",
           "large_obqa__V0-V1.jsonl", "large_obqa__V2.jsonl"):
    p = os.path.join(HERE, "records/e01_e02_e04/results/e4", fn)
    if not os.path.exists(p):
        continue
    for r in jl(p):
        v = r.get("variant")
        if v in NULL:
            NULL[v][r["id"]] = r.get("parsed_original") or INV

missing = [i for i in ids if i not in lab]
if missing:
    raise SystemExit("labels missing for %d dev ids" % len(missing))

g  = [lab[i]["gold"] for i in ids]
oR = [lab[i]["o_R"] or INV for i in ids]
oT = [lab[i]["o_T"] or INV for i in ids]
oC = [lab[i]["o_C"] or INV for i in ids]
v1 = [NULL["V1"].get(i, INV) for i in ids]
v2 = [NULL["V2"].get(i, INV) for i in ids]

n     = len(ids)
yR    = np.array([a == b for a, b in zip(oR, g)], bool)
yT    = np.array([a == b for a, b in zip(oT, g)], bool)
yC    = np.array([a == b for a, b in zip(oC, g)], bool)
chT   = np.array([a != b for a, b in zip(oT, oR)], bool)
chC   = np.array([a != b for a, b in zip(oC, oR)], bool)
sameT = np.array([a == b for a, b in zip(oT, oC)], bool)
psize = np.zeros(n, int); pcor = np.zeros((n, 2), bool)
for j in range(n):
    pool = sorted({v1[j], v2[j]} - {oR[j]})
    psize[j] = len(pool)
    for k, a in enumerate(pool):
        pcor[j, k] = (a == g[j])

v = Inst(yR, yT, yC, chT, chC, sameT, psize, pcor)
rng = np.random.default_rng(0)           # E9-a: one fresh stream per population, Text drawn first
gT  = v.per_reference(rng, "T", NREP)
ratio = gT.mean() / v.real_T

print("Verification (b) - E22-1 per-reference unstratified ratio, large/OBQA/Text\n")
print("  N dev questions        : %d" % v.M)
print("  eligible (|P(x)|>0)  E : %d" % v.E)
print("  m_Text (answers changed): %d" % v.mT)
print("  real gain over R       : %d" % v.real_T)
print("  null gain mean (%d draws): %.2f" % (NREP, gT.mean()))
print("  ratio null/real        : %.3f   expected %.3f" % (ratio, EXPECTED))
ok = abs(round(ratio, 3) - EXPECTED) < 1e-9
print("\nRESULT: %s" % ("PASS - 0.547 reproduced from released records"
                        if ok else "FAIL - got %.3f" % ratio))

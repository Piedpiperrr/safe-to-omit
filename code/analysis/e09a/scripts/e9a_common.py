"""E9-a: matched-null designs on the 7 development populations (CPU re-analysis of saved outputs only).

Reuses the E5-c definitions verbatim: the pool P(x) = distinct answers of V1 and V2 at x that differ from o_R(x),
the oracle (a question counts if any member of the set is correct), and the gain (oracle count - |R correct|).
INVALID is a symbol and counts as incorrect. Nothing outside this stage folder is written.
"""
import sys, collections
from pathlib import Path
import numpy as np

STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parent
# E5-c scripts (r2_common) and the R1 src they import
sys.path.insert(0, str(ROOT / 'P2_R2_CPU_20260919T220412Z' / 'scripts'))
sys.dont_write_bytecode = True
from r2_common import jl, csvout, load_main, INV  # noqa: E402

RES = STAGE / 'results'
LABEL = 'POST-HOC re-analysis (reviewer R3/E9-a); does not change any primary decision'

EXP = ROOT / 'P2_R1_EXP_20260919T050555Z'
P210 = ROOT / 'P2_10_20260911T122423Z'
MMLU_PARQUET = ROOT / 'P2_MMLU_PRO_BREADTH_PROTOCOL_AND_COST_REVIEW_20260915T224801Z/dataset/test-00000-of-00001.parquet'
POPS = [('small', 'obqa'), ('small', 'arc'), ('medium', 'obqa'), ('medium', 'arc'),
        ('large', 'obqa'), ('large', 'arc'), ('large', 'mmlu_pro')]
TASKNAME = {'obqa': 'OBQA', 'arc': 'ARC', 'mmlu_pro': 'MMLU-Pro'}
POPNAME = [f'{p}/{TASKNAME[d]}' for p, d in POPS]
NREP = 1000
NBOOT = 2000
NBOOT_NULL = 100


def gold_map():
    """Gold answers for the three benchmarks (identical sources to e5c.py)."""
    G = {}
    for ds, fn in [('obqa', 'obqa_dev.jsonl'), ('arc', 'arc_validation.jsonl')]:
        for r in jl(P210 / 'data' / fn):
            G[ds, r['id']] = r['gold_answer']
    import pyarrow.parquet as pq
    for r in pq.read_table(MMLU_PARQUET, columns=['question_id', 'answer']).to_pylist():
        G['mmlu_pro', 'test:' + str(r['question_id'])] = r['answer']
    return G


def null_answers():
    """V1 / V2 null-control receiver answers, parsed back to the original option order (identical source to e5c.py)."""
    N = collections.defaultdict(dict)
    for f in sorted((EXP / 'results/e4').glob('*.jsonl')):
        for r in jl(f):
            N[r['pair'], r['dataset'], r['variant']][r['id']] = r.get('parsed_original') or INV
    return N


class Pop:
    """One development population, reduced to the arrays every design needs.

    yR/yT/yC  correctness of o_R / Text / C2C (INVALID never equals gold -> counts as incorrect)
    chT/chC   whether the reference changed the receiver's answer
    psize     |P(x)| in {0,1,2};  pcor  (N,2) correctness of the pool members (padded)
    """

    def __init__(self, pair, ds, main, GOLD, NULL):
        D = main[pair, TASKNAME[ds]]['dev']
        self.pair, self.ds, self.name = pair, ds, f'{pair}/{TASKNAME[ds]}'
        self.ids = list(D['ids'])
        n = self.N = len(self.ids)
        g = [GOLD[ds, i] for i in self.ids]
        oR = list(D['ans']['R']); oT = list(D['ans']['T']); oC = list(D['ans']['C'])
        v1 = [NULL[pair, ds, 'V1'].get(i, INV) for i in self.ids]
        v2 = [NULL[pair, ds, 'V2'].get(i, INV) for i in self.ids]
        self.missing = {k: sum(1 for i in self.ids if i not in NULL[pair, ds, k]) for k in ('V1', 'V2')}
        self.yR = np.array([a == b for a, b in zip(oR, g)], bool)
        self.yT = np.array([a == b for a, b in zip(oT, g)], bool)
        self.yC = np.array([a == b for a, b in zip(oC, g)], bool)
        self.chT = np.array([a != b for a, b in zip(oT, oR)], bool)
        self.chC = np.array([a != b for a, b in zip(oC, oR)], bool)
        self.same_TC = np.array([a == b for a, b in zip(oT, oC)], bool)
        psize = np.zeros(n, int); pcor = np.zeros((n, 2), bool)
        for j in range(n):
            pool = sorted({v1[j], v2[j]} - {oR[j]})
            psize[j] = len(pool)
            for k, a in enumerate(pool):
                pcor[j, k] = (a == g[j])
        self.psize, self.pcor = psize, pcor

    def view(self, idx=None):
        """The full population (idx=None) or a bootstrap resample, as a plain array bundle."""
        if idx is None:
            idx = np.arange(self.N)
        return Inst(self.yR[idx], self.yT[idx], self.yC[idx], self.chT[idx], self.chC[idx],
                    self.same_TC[idx], self.psize[idx], self.pcor[idx])


class Inst:
    """A set of question slots (the population itself, or one bootstrap resample of it)."""

    def __init__(self, yR, yT, yC, chT, chC, same_TC, psize, pcor):
        self.yR, self.yT, self.yC = yR, yT, yC
        self.chT, self.chC, self.same_TC = chT, chC, same_TC
        self.psize, self.pcor = psize, pcor
        self.M = len(yR)
        self.need = ~yR                      # only these questions can add to the oracle
        self.elig = np.flatnonzero(psize > 0)  # nonempty P(x)
        self.E = len(self.elig)
        self.mT = int(chT.sum()); self.mC = int(chC.sum())
        self.a = int((chT & ~chC).sum()); self.b = int((chC & ~chT).sum())
        self.c = int((chT & chC).sum()); self.c_same = int((chT & chC & same_TC).sum())
        # real gains
        self.real_T = int((self.need & yT).sum())
        self.real_C = int((self.need & yC).sum())
        self.real_TC = int((self.need & (yT | yC)).sum())

    # ---- shared primitive: D independent uniform random subsets of `elig`, in uniform random order
    def _pick(self, rng, k, D, ordered=True):
        """`ordered=False` returns the same uniform random k-subset (the k smallest of a uniform row) but
        does not sort within the row; designs that only sum over the subset are unaffected by the order."""
        if k <= 0 or self.E == 0:
            return np.empty((D, 0), int)
        k = min(k, self.E)
        r = rng.random((D, self.E))
        if ordered or k == self.E:
            order = r.argsort(axis=1)[:, :k]
        else:
            order = np.argpartition(r, k - 1, axis=1)[:, :k]
        return self.elig[order]

    def _one_draw(self, rng, jj):
        """Uniform draw from P(x) at each selected slot -> (correct?, index drawn)."""
        sz = self.psize[jj]
        i1 = (rng.random(jj.shape) * sz).astype(int)
        return self.pcor[jj, i1], i1, sz

    # ---- design 1: per-reference matched null
    def per_reference(self, rng, which, D):
        """V_ref* changes exactly as many receiver answers as the real reference; oracle over {R, V_ref*}."""
        m = self.mT if which == 'T' else self.mC
        jj = self._pick(rng, m, D)
        if jj.shape[1] == 0:
            return np.zeros(D, int)
        cor, _, _ = self._one_draw(rng, jj)
        return (cor & self.need[jj]).sum(axis=1)

    # ---- design 2: independent null (the E5-c design), oracle over {R, V_T*, V_C*}
    def independent(self, rng, D):
        g = np.zeros(D, int)
        add = np.zeros((D, self.M), bool)
        for m in (self.mT, self.mC):
            jj = self._pick(rng, m, D)
            if jj.shape[1] == 0:
                continue
            cor, _, _ = self._one_draw(rng, jj)
            np.logical_or.at(add, (np.arange(D)[:, None], jj), cor)
        return (add & self.need).sum(axis=1)

    # ---- design 3: joint-matched null, oracle over {R, V_T*, V_C*}
    def joint(self, rng, D):
        """a Text-only + b C2C-only + c both slots drawn without replacement from the eligible set, roles at random.
        Returns (gains, forced_same_per_draw, K_used, a_used, b_used, c_used, cs_used)."""
        a, b, c, cs = self.a, self.b, self.c, self.c_same
        K = a + b + c
        if K > self.E:  # under-matched: keep the a:b:c shape as closely as possible (largest remainder)
            K2 = self.E
            raw = [a * K2 / K, b * K2 / K, c * K2 / K]
            fl = [int(np.floor(x)) for x in raw]
            for i in np.argsort([-(raw[i] - fl[i]) for i in range(3)])[:K2 - sum(fl)]:
                fl[i] += 1
            a, b, c = fl
            cs = min(cs, c) if c else 0
            K = K2
        jj = self._pick(rng, K, D)
        if K == 0:
            return np.zeros(D, int), np.zeros(D, int), 0, a, b, c, cs
        cor1, i1, sz = self._one_draw(rng, jj)
        flag = cor1.copy()
        lo = a + b + cs                       # start of the "both, different answers" block
        if K > lo:
            s = sz[:, lo:]
            i2 = (i1[:, lo:] + 1 + (rng.random((jj.shape[0], K - lo)) * np.maximum(s - 1, 0)).astype(int)) % np.maximum(s, 1)
            flag[:, lo:] |= self.pcor[jj[:, lo:], i2]
            forced = (s == 1).sum(axis=1)     # |P(x)| = 1 -> the two draws are forced to be the same answer
        else:
            forced = np.zeros(jj.shape[0], int)
        return (flag & self.need[jj]).sum(axis=1), forced, K, a, b, c, cs


def load_pops():
    GOLD, NULL, main = gold_map(), null_answers(), load_main()
    return [Pop(p, d, main, GOLD, NULL) for p, d in POPS]


def fmt(x, n=3):
    return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'

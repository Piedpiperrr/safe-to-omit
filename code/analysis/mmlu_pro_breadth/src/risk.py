"""Frozen 20-point risk calibration; no entry point for correctness labels."""
import math
from scipy.stats import binom,beta
GRID=[j/20 for j in range(1,21)]
def thresholds(scores):
    vals=sorted(scores);n=len(vals);assert n and all(math.isfinite(v) for v in vals)
    return [vals[(j*n+19)//20-1] if j<20 else 'Infinity' for j in range(1,21)]
def mask(scores,q,threshold):
    if q==0:return [False]*len(scores)
    if q==1:return [True]*len(scores)
    return [u<=threshold for u in scores]
def calibrate(scores,disagreements,cutoffs):
    assert len(scores)==len(disagreements)
    ledger=[]
    for q,t in zip(GRID,cutoffs):
        m=mask(scores,q,t);n=sum(m);k=sum(d for d,on in zip(disagreements,m) if on)
        p=float(binom.cdf(k,n,.05)) if n else 1.
        upper=float(beta.ppf(.999,k+1,n-k)) if n and k<n else 1.
        ledger.append({'q':q,'threshold':t,'N_cal':len(scores),'routed':n,'changed':k,
            'coverage':n/len(scores),'conditional_disagreement':k/n if n else None,
            'p_value':p,'CP_upper_0_999':upper,'accepted':p<=.001})
    accepted=[r for r in ledger if r['accepted']]
    chosen=accepted[-1] if accepted else None
    deployment={'q':chosen['q'] if chosen else 0.,'threshold':chosen['threshold'] if chosen else None,
        'mode':('fixed_R' if chosen['q']==1 else 'selective') if chosen else 'fixed_reference',
        'accepted_q':[r['q'] for r in accepted],'chosen_calibration':chosen,
        'accuracy_used_for_selection':False}
    return ledger,deployment

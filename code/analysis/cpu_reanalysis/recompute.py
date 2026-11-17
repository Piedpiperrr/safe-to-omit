"""P2 rebuttal R0: CPU-only stored-record analysis. Writes only beside this file.

Run with the existing c2c_official Python; no model/tokenizer imports or downloads.
The original risk.py supplies thresholds/masks/tests. E17-6 and E21 supply the
subgroup and lost-correction definitions. Bootstrap: paired questions, 2000, seed 0.
"""
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import beta, binom
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent
# As run, this file lived in <release repository>/rebuttal/r0_cpu/; ROOT is that repository (code/, records/, results/,
# splits/ in the dataset layout). Set P2_RELEASE_ROOT to such a tree to run it elsewhere.
ROOT = Path(os.environ.get('P2_RELEASE_ROOT', OUT.parents[1]))
DATA_ROOT = Path(os.environ.get('DATA_DIR', '$DATA_DIR'))  # shared-filesystem data root (a private absolute path in the run)
GOLD_OBQA = DATA_ROOT / 'c2c_reproduction_assets/datasets/allenai--openbookqa/main/train-00000-of-00001.parquet'
MED = 'boundaries_medium/execution_retry1_20260915T164957Z'
PARSER = ROOT / 'code/pipeline/answer_scoring_v2/scoring_v2.py'
RISK = ROOT / 'code/analysis/mmlu_pro_breadth/src/risk.py'
READ = set()


def resolved(p):
    p = Path(p)
    if not p.is_absolute(): p = ROOT / p
    if not p.exists() and Path(str(p) + '.gz').exists(): p = Path(str(p) + '.gz')
    READ.add(str(p))
    return p


def op(p):
    p = resolved(p)
    return gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()


def jl(p):
    with op(p) as f: return [json.loads(s) for s in f if s.strip()]


def js(p):
    with op(p) as f: return json.load(f)


def cr(p):
    with op(p) as f: return list(csv.DictReader(f))


def write(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def source(*ps): return '; '.join(str(resolved(p)) for p in ps)


def module(p, name):
    spec = importlib.util.spec_from_file_location(name, resolved(p))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def cp(k, n):
    if not n: return None, None
    return (0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k)))


def unique(rr, key=lambda r:r['id']):
    d = {key(r): r for r in rr}; assert len(d) == len(rr); return d


def bootstrap_columns(columns):
    a = np.column_stack(columns).astype(float)
    idx = np.random.default_rng(0).integers(0, len(a), (2000, len(a)))
    # Keep memory below 100 MiB by processing columns separately.
    return [np.quantile(a[idx, j].mean(1), [.025, .975]) for j in range(a.shape[1])]


def evidence(ids, r, t, y, mask):
    r, t, y, mask = map(np.asarray, (r, t, y, mask))
    assert len(set(ids)) == len(ids) == len(r) == len(t) == len(y) == len(mask)
    p = np.where(mask, r, t)
    ar, at, ap = [(a == y).astype(int) for a in (r, t, p)]
    values = [ar, at, ap, at-ar, ap-ar, at-ap]
    names = ['R_acc_pct', 'Text_acc_pct', 'policy_acc_pct', 'G_pp', 'remaining_gain_pp', 'loss_vs_Text_pp']
    out = dict(N=len(ids), R_correct=int(ar.sum()), Text_correct=int(at.sum()), policy_correct=int(ap.sum()),
               skipped=int(mask.sum()), changed=int(((r != t) & mask).sum()), kappa=float(mask.mean()),
               kappa_alpha_pp=100*.05*float(mask.mean()), alpha=.05,
               lost_corrections=int(((at == 1) & (ar == 0) & mask).sum()),
               gained_answers=int(((at == 0) & (ar == 1) & mask).sum()))
    for n, v, ci in zip(names, values, bootstrap_columns(values)):
        out[n] = 100*float(v.mean()); out[n+'_lo95'], out[n+'_hi95'] = 100*ci
    out['retention_descriptive'] = float((ap.sum()-ar.sum())/(at.sum()-ar.sum())) if at.sum()!=ar.sum() else None
    out['retention_unstable'] = out['G_pp_lo95'] <= 0 <= out['G_pp_hi95'] or abs(out['G_pp']) < 1
    out['bootstrap'] = 'paired question percentile 95%; 2000 resamples; numpy default_rng(0); conditional on frozen policy'
    out['change_CP95_lo'], out['change_CP95_hi'] = cp(out['changed'], out['skipped'])
    return out


def disciplines():
    f = 'records/mmlu_pro_breadth/summary/merged_numeric_rows.jsonl'
    raw = unique(jl(f)); data = {}
    for sp in ['fit', 'cal', 'dev']:
        groups = js(f'results/mmlu_pro_breadth/splits/{sp}_groups.json')
        assert len({g['representative_id'] for g in groups}) == len(groups)
        data[sp] = [(raw[g['representative_id']], g['category']) for g in groups]
        assert all(r['split'] == sp for r, _ in data[sp])
    assert [len(data[s]) for s in ['fit','cal','dev']] == [3000,6000,2641]
    cats = sorted({c for _, c in data['fit']}); assert len(cats) == 14
    stored = {(r['path'],r['category']):r for r in cr('results/e17/results/E17_6_mmlu_categories.csv')}
    rows = []
    for ref, act in [('Text','T'),('C2C','C')]:
        def arrays(sp, c=None):
            a = [r for r, cat in data[sp] if c is None or c == cat]
            return np.array([r['u'] for r in a]), np.array([norm(r['R']) != norm(r[act]) for r in a])
        fit, _ = arrays('fit'); cal, dis = arrays('cal')
        cuts = risk.thresholds(fit); led, dep = risk.calibrate(cal, dis, cuts)
        assert dep['q'] == .4 and dep['threshold'] == js('results/mmlu_pro_breadth/deployments/deployment_configs.json')['deployments'][act]['threshold']
        for strategy, delta in [('pooled',.001),('per_discipline',.001),('per_discipline_div14',.001/14)]:
            group_rows = []
            for c in cats:
                fu, _ = arrays('fit',c); cu, cd = arrays('cal',c); du, dd = arrays('dev',c)
                if strategy == 'pooled':
                    q, tau = dep['q'], dep['threshold']; chosen = dep['chosen_calibration']
                    passed = None; certification_scope = 'all disciplines pooled; no discipline-specific certificate'
                else:
                    cled, _ = risk.calibrate(cu,cd,risk.thresholds(fu))
                    accepted = [r for r in cled if r['p_value'] <= delta]
                    chosen = accepted[-1] if accepted else None
                    q, tau = (chosen['q'],chosen['threshold']) if chosen else (0.,None)
                    passed = bool(chosen); certification_scope = c
                    old = stored[ref,c]; suffix = '' if strategy == 'per_discipline' else '_delta_over_14'
                    assert q == (0. if old['q'+suffix]=='fallback' else float(old['q'+suffix]))
                m = np.asarray(risk.mask(du,q,tau),bool); cm = np.asarray(risk.mask(cu,q,tau),bool)
                n,k = int(m.sum()),int(dd[m].sum()); lo,hi = cp(k,n)
                if strategy != 'pooled' and q:
                    assert n == int(old['dev_n'+suffix]) and k == int(old['dev_k'+suffix])
                row = dict(reference=ref,strategy=strategy,discipline=c,N_fit=len(fu),N_cal=len(cu),N_dev=len(du),
                           certificate_passed=passed,certificate_scope=certification_scope,selected_q=q,selected_threshold=tau,
                           calibration_skipped=int(cm.sum()),calibration_changed=int(cd[cm].sum()),
                           certificate_n=chosen['routed'] if chosen else None,certificate_k=chosen['changed'] if chosen else None,
                           certificate_p=chosen['p_value'] if chosen else None,alpha=.05,delta_per_candidate=delta,
                           dev_skipped=n,dev_total=len(du),dev_changed=k,dev_change_rate=k/n if n else None,
                           dev_change_CP95_lo=lo,dev_change_CP95_hi=hi,coverage=n/len(du),
                           mode='full_reference' if not q else 'selective',analysis_status='retrospective exposed development',
                           source=source(f,'results/mmlu_pro_breadth/splits/fit_groups.json',
                                         'results/mmlu_pro_breadth/splits/cal_groups.json','results/mmlu_pro_breadth/splits/dev_groups.json',RISK,
                                         'code/analysis/e17/scripts/e17_6.py'))
                group_rows.append(row)
            rows.extend(group_rows)
            total = {k:sum(r[k] for r in group_rows) for k in ['N_fit','N_cal','N_dev','dev_skipped','dev_total','dev_changed']}
            total.update(reference=ref,strategy=strategy,discipline='ALL',coverage=total['dev_skipped']/total['dev_total'],
                         dev_change_rate=total['dev_changed']/total['dev_skipped'] if total['dev_skipped'] else None,
                         analysis_status='retrospective exposed development; micro aggregation',source=group_rows[0]['source'])
            total['dev_change_CP95_lo'],total['dev_change_CP95_hi'] = cp(total['dev_changed'],total['dev_skipped'])
            rows.append(total)
    write('discipline_certification.csv',rows)
    return [r for r in rows if r['discipline']=='ALL']


def medium(ds,sp):
    base = 'records/boundaries_medium/execution_retry1' if ds=='obqa' and sp in ['fit','cal'] else f'records/{MED}'
    f = f'{base}/actions/{ds}_{sp}.jsonl'; actions = jl(f)
    a = unique(actions,lambda r:(r['id'],r['action']))
    ids = js(f'results/boundaries_medium/splits/{ds}_{sp}_representatives.json')
    ans = {}
    for act in ['R','T','C']:
        ans[act] = []
        for i in ids:
            r = a[i,act]
            parsed = parser.parse_answer(r['output']['raw_answer'],r['query']['choice_labels'])
            answer = norm(parsed['answer'])
            assert answer == norm(r['answer']), (i,act)
            ans[act].append(answer)
    return ids,ans,f


def joint_and_baselines():
    table = pq.read_table(resolved(GOLD_OBQA),columns=['id','answerKey'],use_threads=False)
    ho_gold = dict(zip(table['id'].to_pylist(),table['answerKey'].to_pylist()))
    arc_gold = {r['id']:r['gold'] for r in jl('records/sealed_arc_test/inputs/evaluation_gold_after_prediction_freeze.jsonl')}
    labels_path = 'records/answer_parser_v2/labels/full_development_P2_SCORING_V2.jsonl'
    labels = {(r['dataset'],r['id']):r for r in jl(labels_path) if r['pair']=='large'}
    assert all(ho_gold[i]==r['gold'] for (ds,i),r in labels.items() if ds=='obqa')
    rows=[]
    for ds in ['obqa','arc']:
        f = f'records/e16/results/analysis_oos/routes_llama_{ds}.jsonl'; rr=jl(f)
        gold=ho_gold if ds=='obqa' else arc_gold; ids=[r['id'] for r in rr]
        ev=evidence(ids,[r['o_R'] for r in rr],[r['o_ref'] for r in rr],[gold[i] for i in ids],[r['omitted'] for r in rr])
        ev.update(setting=f'Llama-8B/{ds.upper()}/Text',population='held-out OBQA 744' if ds=='obqa' else 'ARC test 1172 (later OOS)',
                  source=source(f,GOLD_OBQA if ds=='obqa' else 'records/sealed_arc_test/inputs/evaluation_gold_after_prediction_freeze.jsonl'))
        rows.append(ev)
    f='records/e09b_e09c/analysis/SEALED_ROUTES.jsonl'; rr=jl(f)
    order=[r['id'] for r in jl('records/e09b_e09c/inputs/holdout_744_queries.jsonl')]
    for name,pol in [('Text','large_obqa_T_q80'),('Text+fact','large_obqa_TF_q75')]:
        d=unique([r for r in rr if r['policy']==pol]); a=[d[i] for i in order]
        ev=evidence(order,[r['R_answer'] for r in a],[r['reference_answer'] for r in a],[ho_gold[i] for i in order],[r['omitted'] for r in a])
        assert all(r['policy_answer']==(r['R_answer'] if r['omitted'] else r['reference_answer']) for r in a)
        ev.update(setting=f'large/OBQA/{name}',population='held-out OBQA 744',source=source(f,GOLD_OBQA))
        rows.append(ev)
    assert [(r['N'],r['R_correct'],r['Text_correct'],r['policy_correct'],r['skipped'],r['changed']) for r in rows] == [
        (744,598,618,615,440,3),(1172,953,993,991,794,15),(744,618,647,640,595,12),(744,618,678,667,558,16)]
    # Separate full-development accuracy and 128-question development latency panel.
    x3=[]
    for p in sorted((ROOT/'records/x03_llama_receiver/results/runs').glob('chain_*.jsonl.gz')): x3+=jl(p)
    x3=unique(x3,lambda r:(r['dataset'],r['id']))
    timing=jl('records/e16/replay_llama/records/e2e_requests.jsonl')
    for ev,ds,q,tau in zip(rows[:2],['obqa','arc'],[.6,.7],[.009371757507324219,.014996349811553955]):
        ids=[r['id'] for r in jl(f'records/x03_llama_receiver/records/populations/{ds}_dev.jsonl') if r['representative']]
        a=[x3[ds,i] for i in ids]; u=np.array([r['P']['ProbeMax'] for r in a]); om=u<=tau
        dv=evidence(ids,[norm(r['R']['parsed']) for r in a],[norm(r['T']['parsed']) for r in a],[labels[ds,i]['gold'] for i in ids],om)
        for k in ['N','R_correct','Text_correct','policy_correct','skipped','changed','G_pp']:ev['dev_'+k]=dv[k]
        tt=unique([r for r in timing if r['dataset']==ds],lambda r:(r['id'],r['mode']))
        panel=[r['id'] for r in timing if r['dataset']==ds and r['mode']=='policy']
        save=np.array([tt[i,'reference']['latency_ms']-tt[i,'policy']['latency_ms'] for i in panel])
        assert len(panel)==128
        ci=bootstrap_columns([save])[0]
        ev.update(dev_timing_panel_N=128,dev_timing_saving_ms=float(save.mean()),dev_timing_saving_lo95_ms=ci[0],dev_timing_saving_hi95_ms=ci[1],
                  dev_accuracy_source=source('records/x03_llama_receiver/results/runs',labels_path,f'records/x03_llama_receiver/records/populations/{ds}_dev.jsonl'),
                  dev_timing_source=source('records/e16/replay_llama/records/e2e_requests.jsonl'),
                  timing_population_note='development 128-question panel; not the unseen accuracy population')
    write('helpful_text_joint_evidence.csv',rows)
    base=[]
    for ds in ['obqa','arc']:
        ids,ans,mp=medium(ds,'dev'); gold=np.array([labels[ds,i]['gold'] for i in ids])
        for setting,a,protocol,src in [
            ('Qwen3-1.7B alone',ans['R'],'free generation + frozen parser',mp),
            ('Qwen3-8B alone',[norm(labels[ds,i]['o_R']) for i in ids],'free generation + frozen parser',labels_path),
            ('medium Text',ans['T'],'free generation + frozen parser',mp),
            ('medium C2C',ans['C'],'free generation + frozen parser',mp),
            ('large Text',[norm(labels[ds,i]['o_T']) for i in ids],'free generation + frozen parser',labels_path),
            ('large C2C',[norm(labels[ds,i]['o_C']) for i in ids],'free generation + frozen parser',labels_path)]:
            correct=int((np.asarray(a)==gold).sum())
            base.append(dict(dataset=ds.upper(),population='exposed development',setting=setting,N=len(ids),correct=correct,accuracy_pct=100*correct/len(ids),
                             answer_protocol=protocol,source=source(src,labels_path)))
        hf=f'records/e09b_e09c/records/large_{ds}_dev_plain_rank2.jsonl'; hh=unique(jl(hf))
        correct=sum(hh[i]['argmax_helper_label']==g for i,g in zip(ids,gold))
        base.append(dict(dataset=ds.upper(),population='exposed development',setting='Qwen2.5-7B-Instruct alone',N=len(ids),correct=correct,
                         accuracy_pct=100*correct/len(ids),answer_protocol='E9c fixed-prefix label-probability argmax; NOT free generation',source=source(hf,labels_path)))
    write('single_model_baselines.csv',base)
    assert [r['correct'] for r in base] == [490,617,523,486,645,598,640,217,268,221,202,274,266,262]
    return rows,base,ho_gold


def lost_correction(gold):
    ids,ans,af=medium('obqa','cal'); di,dans,df=medium('obqa','dev')
    pr=f'records/{MED}/probes/obqa_cal.jsonl'; dr=f'records/{MED}/probes/obqa_dev.jsonl'
    pp=unique(jl(pr)); dp=unique(jl(dr))
    u=np.array([pp[i]['ProbeMax'] for i in ids]); du=np.array([dp[i]['ProbeMax'] for i in di])
    r,t=np.asarray(ans['R']),np.asarray(ans['T']); d=r!=t
    cf=f'results/{MED}/thresholds/obqa.json'; cfg=js(cf); cuts=cfg['thresholds']
    assert cuts==risk.thresholds(cfg['fit_scores']) and cfg['q_grid']==risk.GRID
    ledger,dep=risk.calibrate(u,d,cuts); assert dep['q']==0
    saved=cr(f'results/{MED}/calibration/obqa_T.csv')
    assert len(saved)==20
    for a,b in zip(ledger,saved):
        assert (a['routed'],a['changed'])==(int(b['routed']),int(b['changed']))
        assert abs(a['p_value']-float(b['p_value']))<1e-12 and not a['accepted']
    # Sequential label oracle: only low-u disagreement IDs are queried, in a fixed tie order.
    order=sorted(np.flatnonzero(d),key=lambda j:(u[j],ids[j])); rows=[]; used=set()
    for eps in [.01,.02]:
        queried=[]; lost=0; at=0; chosen=None; first_rej=None; early_stop=None
        allowable=max(k for k in range(len(ids)+1) if binom.cdf(k,len(ids),eps)<=.001)
        for q,tau in zip(risk.GRID,cuts):
            m=np.asarray(risk.mask(u,q,tau),bool)
            while at<len(order) and m[order[at]]:
                j=order[at]; at+=1; queried.append(j); used.add(ids[j])
                lost+=int(t[j]==gold[ids[j]] and r[j]!=gold[ids[j]])
                if early_stop is None and lost>allowable: early_stop=len(queried)
            p=float(binom.cdf(lost,len(ids),eps))
            if p<=.001: chosen=dict(q=q,tau=tau,k_L=lost,p=p,D_dep=len(queried),cal_skipped=int(m.sum()))
            else: first_rej=q; break
        first_rejected_lost=lost if first_rej is not None else None
        if chosen is None:
            chosen=dict(q=0.,tau=None,k_L=0,p=None,D_dep=0,cal_skipped=0)
        mask=np.asarray(risk.mask(du,chosen['q'],chosen['tau']),bool)
        ev=evidence(di,dans['R'],dans['T'],[gold[i] for i in di],mask)
        assert ev['Text_correct']-ev['R_correct']==33
        ev.update(epsilon=eps,delta_per_candidate=.001,N_cal=len(ids),max_acceptable_q=chosen['q'],threshold=chosen['tau'],
                  cal_lost_corrections=chosen['k_L'],cal_lost_denominator=len(ids),cal_lost_p=chosen['p'],cal_skipped=chosen['cal_skipped'],
                  D_all=int(d.sum()),labels_at_selected_threshold=chosen['D_dep'],labels_needed_E21_Dseq=len(queried),
                  labels_early_stop_at_rejection=early_stop,first_rejected_q=first_rej,cal_max_accepted_lost_count=allowable,
                  first_rejected_lost_count=first_rejected_lost,loss_test_passed=chosen['q']>0,
                  change_test_q=0,joint_certificate_q=None,
                  delta_acc_pp=-ev['loss_vs_Text_pp'],delta_acc_lo95_pp=-ev['loss_vs_Text_pp_hi95'],delta_acc_hi95_pp=-ev['loss_vs_Text_pp_lo95'],
                  analysis_status='retrospective exposed development; label-assisted loss-only test; no label-free behavior guarantee',
                  source=source(af,df,pr,dr,cf,GOLD_OBQA,'code/analysis/e21/scripts/e21_1_cert.py'))
        rows.append(ev)
    write('medium_text_lost_correction.csv',rows)
    return rows


def norm(x): return 'INVALID' if x is None else x


if __name__=='__main__':
    # As run, a site storage-policy assertion stood here (output under the shared filesystem, not the home directory).
    print(json.dumps(dict(model='not loaded',dataset=str(GOLD_OBQA),cache=os.getenv('XDG_CACHE_HOME'),output=str(OUT),
                          checkpoint='none',log=str(OUT/'execution.log'),temporary=str(OUT/'tmp')),indent=2))
    assert hashlib.sha256(PARSER.read_bytes()).hexdigest()=='d05978f400cf69aa6bddc56168a260ac3ea9158abaa640e9331602dc9bfcc2f9'
    risk=module(RISK,'r0_frozen_risk'); parser=module(PARSER,'r0_frozen_parser')
    disc=disciplines(); joint,base,gold=joint_and_baselines(); lost=lost_correction(gold)
    print('DISCIPLINES',json.dumps([{k:r[k] for k in ['reference','strategy','dev_skipped','dev_changed','coverage']} for r in disc]))
    print('JOINT',json.dumps([{k:r[k] for k in ['setting','R_correct','Text_correct','policy_correct','G_pp','remaining_gain_pp','loss_vs_Text_pp','retention_descriptive']} for r in joint]))
    print('BASELINES',json.dumps([{k:r[k] for k in ['dataset','setting','N','correct','accuracy_pct']} for r in base]))
    print('LOST',json.dumps([{k:r[k] for k in ['epsilon','max_acceptable_q','labels_needed_E21_Dseq','labels_early_stop_at_rejection','labels_at_selected_threshold','skipped','delta_acc_pp','delta_acc_lo95_pp','delta_acc_hi95_pp','retention_descriptive']} for r in lost]))
    print('ALL ASSERTIONS PASSED; no model forward, fit, GPU, scheduler or network action')

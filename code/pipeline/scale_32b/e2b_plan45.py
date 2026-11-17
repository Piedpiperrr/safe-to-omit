"""E2b-only plan overlay. Existing scientific/operational modules stay byte-identical.

Release note: the submit, retry and estimate subcommands of the run (job submission, the one-retry submission
helper and the resource re-estimate) are not included.

Only e2_debug_ops.plan('e2b') is replaced in this process. Other processes,
including E2a, continue to use the unchanged modules and original plans.
"""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import sys

import e2_debug_ops as ops
from e2b_gate import sha,verify_freeze,freeze_path

HERE=Path(__file__).resolve().parent
PLAN=HERE/'e2b_plan45.json'
FREEZE=HERE/'E2B_PLAN45_FREEZE.json'
_BASE_PLAN=ops.plan


def verify():
    base=verify_freeze();r=ops.read(FREEZE)
    assert r['status']=='FROZEN_OPERATIONAL_PLAN45'
    assert sha(freeze_path())==r['scientific_freeze_sha256']
    for p,h in r['files'].items():assert sha(p)==h,p
    for p,h in r['pilot_source_sha256'].items():assert sha(p)==h,p
    return r


def install(plan):
    """Parameter overlay only; no generation, parser, score, or test changes."""
    ops.plan=lambda kind:copy.deepcopy(plan) if kind=='e2b' else _BASE_PLAN(kind)


def activate():
    r=verify();install(ops.read(PLAN));return r


def wave_units(stage,wave):
    plan=ops.plan('e2b');keys=plan['submission_waves'][stage][wave]
    return [u for u in ops.units('e2b',stage) if u.get('parent',u['key']) in keys]


def completed(u):
    return ops.completed('e2b',u,sha(freeze_path()))


def analyze(args):
    from e2b_analyze import Analysis
    a=Analysis()
    # Retain the scientific analysis map; bind the added operational gate/plan as evidence.
    a.source(FREEZE);a.source(PLAN)
    if args.stage=='pilot':a.pilot(args)
    elif args.stage in ['dev','heldout']:a.evaluate(args.stage)
    elif args.stage=='export-cal':a.export_cal()
    else:getattr(a,args.stage)()
    print('E2B_PLAN45_CPU_COMPLETE',args.stage,'INTERNAL_PENDING_RECHECK')


def worker(args):
    manifest=ops.read(args.manifest);u=manifest['units'][args.rank]
    assert manifest['kind']=='e2b' and u==ops.unit_by_key('e2b',u['key'])
    assert u['stage']!='pilot' and u['operational_freeze']==FREEZE.name
    import run_e2b_formal
    sys.argv=[str(HERE/'run_e2b_formal.py'),'--manifest',args.manifest,'--rank',str(args.rank)]
    run_e2b_formal.main()


def export(args):
    import export_e2_recheck as x
    x.DOCS += ['E2B_PLAN45_HANDOFF_ZH.md',FREEZE.name,PLAN.name]
    x.export(args.stage)


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='command',required=True)
    sp.add_parser('verify')
    p=sp.add_parser('worker');p.add_argument('--manifest',required=True);p.add_argument('--rank',type=int,required=True)
    p=sp.add_parser('analyze');p.add_argument('stage',choices=['pilot','fit','cal','export-cal','dev','heldout','timing','report'])
    p.add_argument('--resource-feasible',action='store_true');p.add_argument('--resource-note',default='')
    p=sp.add_parser('export');p.add_argument('stage',choices=['cal','heldout','timing'])
    args=ap.parse_args();r=activate()
    if args.command=='verify':print('PLAN45_GATE_PASS',r['commit'],r['utc'],str(FREEZE));return
    globals()[args.command](args)

if __name__=='__main__':main()

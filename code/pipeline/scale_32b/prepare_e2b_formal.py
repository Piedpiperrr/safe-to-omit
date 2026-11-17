"""Prepare the revised formal draft without changing the submitted smoke config."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    assert not (HERE/'FREEZE.txt').exists()
    assert not (HERE/'e2b_runs').exists(), 'Draft revisions must precede formal output'
    cfg=json.loads((HERE/'e2b_config.json').read_text())
    cfg.update(score='logsumexp_nonmax_label_logprob',diagnostic_score='1-max(p)',nonfinite_route='Text',
               primary_policy='joint_eps001',heldout_cp_upper_quantile=.975,
               pilot_min_net_correct=6,pilot_gate_purpose='gain-retention analysis priority only',
               familywise_bonferroni_bound=.06,full_stage_shards=4,full_stage_nodes=4,
               deadline_utc='2026-11-16T18:00:00+00:00',below_gate_resource_decision_cutoff_utc='2026-11-15T06:00:00+00:00',
               resource_plan_status='PENDING_SMOKE_MEASUREMENTS_AND_USER_CONFIRMATION',
               walltime_hours={'pilot':3,'fit_rest':4,'cal':3,'dev':2,'heldout':2,'replay':6})
    cfg.update(score_labels=['A','B','C','D'],score_nonmax_labels=3,
               timing_arms=['policy','reference','receiver_alone'],
               timing_policy_order=['joint_eps001','behavior','joint_eps002'],
               timing_arm_rotation='left by (panel_ordinal+fixed_policy_index) modulo 3',
               scope_claim='model-scale extension on previously exposed benchmarks')
    estimate=HERE/'E2B_RESOURCE_ESTIMATE.json'
    if estimate.exists():
        measured=json.loads(estimate.read_text())
        assert measured['purpose']=='resource_planning_only_no_scientific_analysis' and measured['N']==16
        cfg['resource_plan_status']='MEASURED_AWAITING_USER_CONFIRMATION'
        cfg['resource_estimate_sha256']=hashlib.sha256(estimate.read_bytes()).hexdigest()
        cfg['walltime_hours']={s['stage']:s['requested_cap_h'] for s in measured['stages']}
    (HERE/'e2b_formal_config.json').write_text(json.dumps(cfg,indent=2)+'\n')


if __name__=='__main__':main()

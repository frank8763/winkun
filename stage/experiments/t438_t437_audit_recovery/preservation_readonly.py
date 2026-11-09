"""Stored-event diagnosis only: stdlib, no model/native/optimizer execution."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[3]
def main():
    evidence=[]
    for run,arm in [('t437_joint_closed_loop_rl','fresh_rl'),('t439_representation_rl_control','representation_rl')]:
        path=ROOT/'stage/runs'/run/arm/'summary.json';s=json.loads(path.read_text())
        for rr in s['result']['rounds']:
            fit=rr['stages'][-1]['fit'];g=fit['gradient_components'][0]
            norm=g['preservation_gradient_norm'];pg=g['raw_PG_norm']
            evidence.append(dict(run=run,round=rr['round'],first_preservation_gradient_norm=norm,
                first_preservation_gradient_norm_squared=norm*norm,
                projection_threshold_in_original_code=1e-30,
                original_projection_norm_condition_passes=norm*norm>1e-30,
                first_raw_PG_norm=pg,first_projected_PG_norm=g['projected_PG_norm'],
                first_PG_unchanged_by_projection=pg==g['projected_PG_norm'],
                first_old_mean_loss=fit['trace'][0][1],first_six_mean_loss=fit['trace'][0][2],
                accepted=rr['accepted']))
    assert len(evidence)==12
    inactive=[r for r in evidence if not r['original_projection_norm_condition_passes']]
    assert len(inactive)==8 and all(r['first_PG_unchanged_by_projection'] for r in inactive)
    code=ROOT/'stage/experiments/t418_evening_repair_chain/repair_core.py'
    out=dict(state='STORED_EVENTS_CONFIRM_FIRST_STEP_PROJECTION_INACTIVE',new_native=0,new_Actor_forward=0,new_optimizer=0,
        scope='T437 8 rounds + T439 4 rounds; first optimizer step only. Eight start from original weights; four T437 rounds start from the one temporarily retained update.',
        norm_inactive_first_steps=8,norm_eligible_first_steps=4,
        initial_analysis_assertion_corrected='The first read-only assertion assumed all12 began at original weights; raw logs show rounds4-7 ofT437 use a kept update. Assertion rejected before writing any evidence/model/run.',
        code_path=str(code),code_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),
        mechanism='Self-distillation quadratic loss has zero derivative at identical policy; the implemented first-order reference projection is initially inactive. This is an expected limitation, not proof of a numerical bug.',
        causal_limit='Does not establish sole cause of failures or that tighter constraints can acquire new skill. No new threshold proposed; no experiment launched.',evidence=evidence)
    path=ROOT/'stage/docs/review_20261109/FIRST_STEP_PRESERVATION_EVIDENCE.json'
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(state=out['state'],rounds=len(evidence),new_native=0,new_optimizer=0)))
if __name__=='__main__':main()

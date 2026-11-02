"""Register one fixed finite feasibility experiment before any new native step."""
from pathlib import Path
import json,sys
from t295_contract import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t295_hindsight_support'

def main():
    assert not OUT.exists();OUT.mkdir()
    original=ROOT/'stage/runs/t287_time_aligned_credit/protocol.json';p=json.loads(original.read_text())
    records=[json.loads(line) for line in (original.parent/'cases.jsonl').read_text().splitlines()]
    sources=[];reset_ids=['dev6_LEFT_13','dev6_RIGHT_13'];goals=[]
    for reset in reset_ids:
        fixed=next(r for r in records if r['id']=='time_aligned_credit_r0_'+reset+'_s0');a=load(fixed['path']);j=(len(a['states'])-1)//2;goal=a['next_poses'][j,:2].tolist()
        goals.append(dict(reset_id=reset,goal_xy=goal,selection_reference_id=fixed['id'],selection_index=j,selection_rule='next_pose at floor((N-1)/2) of fixed historical round0 seed0, chosen before all T295 observations; no replacement sampling'))
        job=next(v for v in p['training_jobs'] if v['id']==reset)
        for i in range(4):
            r=next(r for r in records if r['id']=='time_aligned_credit_r0_'+reset+'_s'+str(i));r=dict(r,job=job,auxiliary_goal_xy=goal)
            sources.append(r)
    inputs=dict(p['input_hashes']);inputs[str(original)]=C.sha(original)
    for f in HERE.glob('*.py'):inputs[str(f)]=C.sha(f)
    for f in [original.parent/'cases.jsonl',ROOT/'stage/experiments/t294_paired_action_value/t294_native.py',ROOT/'stage/experiments/t278_fixed_goal_starts/t278_guard.py',ROOT/'stage/experiments/t290_exact_duplicate_store/store.py']:
        inputs[str(f)]=C.sha(f)
    for r in sources:
        f=resolve(r['path']);assert C.sha(f)==r['sha256'];inputs[str(f)]=C.sha(f);f=Path(r['job']['exact_snapshot']);inputs[str(f)]=C.sha(f)
    plan=dict(experiment='T295_HPG_GOAL_HISTORY_NATIVE_SUPPORT_FEASIBILITY',source_protocol=str(original),policy_hash=p['policy_hash'],checkpoint=p['checkpoint'],sources=sources,auxiliary_goals=goals,reset_ids=reset_ids,input_hashes=inputs,
        ESS_min=2.,max_normalized_weight=.75,max_abs_terminal_log_weight=20.,
        max_commands=16,max_physics=448000,wall_seconds=1200,final_reserve_seconds=90,audit_wall_seconds=300,
        episode_bytes=320*1024,partial_bytes=384*1024,payload_bytes=8*320*1024+384*1024,metadata_bytes=1024**2,final_bytes=4*1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        scientific_scope='Eight fixed historical safeT255 conditional-AR trajectories at original6m LEFT/RIGHT centres, original full native replay plus alternate-goal recorded-action prefix. No new independent behavior data, no Actor/optimizer/critic updates, no original6m evaluation or promotion/7/8. One fixed midpoint future achieved position per reset, four source paths share that target; uniform finite two-goal auxiliary distribution conditional on reset, fixed historical data-adaptive goals are DEVELOPMENT ONLY, not an unbiased original6m objective/continuous-goal integral. Any future estimator training would need a separately registered fixed-target fresh-behavior protocol. All13 current/historygoal fields reconstructed from actual original factory/history generation witnesses; initial counterfactual history explicitly rewritten from those witnessed generating poses,107nongoal unchanged. Native alternate actions forced from historical records for data validation, not policy success.2ms firsthit radius.2 CONTACT priority; truncate insidehold with true packet/reward/duration/integration; unknown originalgoal stopped suffix censored. Goal-independent native integration prefix must match. Full original means/logp exact, true goalconditional AR density recomputed with previousmean dependence; reject noninvertible native rounded actions, no density replacement/clipping. Terminal prefix weights logstable; exploratory finite-four support screen BOTH reset groups ESS>=2,maxweight<=.75,abs each terminal logweight<=20; not a confidence bound/safety certificate or proof of learning feasibility. All original/alternate successes/failures/censoring separated, no thresholds/goals/seeds/noise changes. End independent zero-native zero-optimizer audit, all originals/models/failures preserved and complete RAM remote memberSHA backup then source sync. Native-exclusive output, global1GiB/free40GiB, no Phase/planner/waypoint/teacher/router.')
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+plan['final_bytes']<plan['global_bytes']
    for dest in [HERE/'protocol.json',OUT/'protocol.json']:dest.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(registered=True,sources=len(sources),goals=goals,current_global_bytes=S.used(ROOT/'stage/runs'),worst_global_bytes=S.used(ROOT/'stage/runs')-S.used(OUT)+plan['final_bytes'])))

if __name__=='__main__':main()

"""Lock the known 96 records and a disjoint native reset-family split."""
from event_model import *
import ast
import shutil
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t272_event_value_pilot'


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    paths=[ROOT/'stage/runs/t267_causal_credit_course/training.jsonl',ROOT/'stage/runs/t271_frozen_feature_exploration/exploration.jsonl']
    old=list(map(json.loads,paths[0].read_text().splitlines()))
    new=[r for r in map(json.loads,paths[1].read_text().splitlines()) if r['mode']=='feature_copula']
    assert len(old)==80 and len(new)==16
    records=[]
    for r in old+new:
        j=r['job'];family=json.dumps([j['delta_x_m'],j['delta_y_m'],j['delta_yaw_deg']])
        split='test' if [j['delta_x_m'],j['delta_y_m'],j['delta_yaw_deg']]==[-.03,-.03,0.] else 'train'
        records.append(dict(id=r['id'],path=r['path'],sha256=r['sha256'],physics_steps=r['physics_steps'],
            complete=r['complete'],success=r['success'],collision=r['collision'],timeout=r['timeout'],
            side=j['case']['side'],distance_m=j['case']['distance_m'],cap=j['case']['step_cap'],family=family,split=split,
            policy_hash=r['policy_hash'],historical_overlap=bool(r.get('exact_historical_overlap')),
            signature=[r['array_hashes']['states'],r['array_hashes']['actions'],r['integration_sha256']]))
    assert sum(r['split']=='test' for r in records)==16
    inputs=[*paths,ROOT/'stage/docs/long_distance_20261101/T271_NEXT_DATA_AUDIT.json',
        ROOT/'stage/experiments/t262_event_contract/contract.py',ROOT/'stage/experiments/t262_event_contract/contract_recovery.py',
        ROOT/'stage/runs/t267_causal_credit_course/final_checkpoint.pt.gz',ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz',
        ROOT/'stage/runs/t267_causal_credit_course/protocol.json',ROOT/'stage/runs/t271_frozen_feature_exploration/protocol.json',
        *sorted(HERE.glob('*.py'))]
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    plan=dict(experiment='T272',scope='One offline competing-event forecast pilot, original goals, no Actor/native update',
        records=records,input_hashes={str(f):sha(f) for f in inputs},seed=2026111201,
        optimizer_steps=500,batch_size=64,learning_rate=.001,horizons_seconds=[1,2,4,8,16,32],
        horizons_physics=[500,1000,2000,4000,8000,16000],train_wall_seconds=210,wall_seconds=300,
        checkpoint_bytes=512*1024,evaluation_bytes=768*1024,metadata_bytes=768*1024,final_bytes=2*1024**2,
        global_bytes=1024**3,min_free_bytes=40*1024**3,physics_steps=0,Actor_updates=0,
        Actor_checkpoint=str(ROOT/'stage/runs/t267_causal_credit_course/final_checkpoint.pt.gz'),
        predictive_screen=dict(Brier_total_vs_frequency_max_ratio=.90,both_event_Brier_no_worse=True,max_five_bin_ECE=.10),
        automatic_Actor_coupling=False,action_ranking_identified=False,new_six_m_navigation_gate=False,
        contract='Actual original120 causal state plus current logged action only. No goal/history relabeling, future inputs, case/seed/policy IDs, native simulator, old SAC critic, BC, planner, waypoints, runtime teacher or routing. Fixed native offset family(-.03,-.03,0) held out in every task/behavior policy; no exact trajectory hashes cross split. Train-only episode-balanced normalization; uniform episode then decision sampling; six cumulative first-event targets with known original cap and actual native goal/contact order. Shared competing-risk recursion conserves mass and enforces monotonicity. One fixed seed, one 500-step fit, no grid, best-checkpoint selection or repeat. Train-frequency and untrained-model baselines are descriptive. Test used only for predeclared initial/final metrics, never updates or selection. No Actor gradients regardless of the predictive screen; mixed-behavior observational prediction is not changed-policy Q, causal action ranking or a safety certificate. All source/failures/final Adam retained. Phase isolated.')
    assert used(ROOT/'stage/runs')+plan['final_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>=plan['min_free_bytes']
    OUT.mkdir();save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan)
    print(json.dumps(dict(state='REGISTERED',train_episodes=80,test_episodes=16,
        reserve_full_bytes=plan['final_bytes'],global_if_full=used(ROOT/'stage/runs')+plan['final_bytes'])),flush=True)


if __name__=='__main__':main()

"""Register one new model-directed counterfactual hypothesis, not a noise grid."""
from t273_native import *
from proposals import propose
import shutil
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t273_value_direction_probe'


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    prior=json.loads((ROOT/'stage/runs/t269_same_state_controls/protocol.json').read_text())
    rr=list(map(json.loads,(ROOT/'stage/runs/t269_same_state_controls/cases.jsonl').read_text().splitlines()))
    reference=next(r for r in rr if r['id']=='control_18000')
    summary=json.loads((ROOT/'stage/runs/t272_event_value_pilot/summary.json').read_text())
    assert summary['state']=='COMPLETE' and summary['report']['predictive_screen_passed']
    assert json.loads((ROOT/'stage/docs/long_distance_20261101/T272_AUDIT.json').read_text())['state']=='POSTRUN_EVIDENCE_RECONCILED'
    assert json.loads((ROOT/'stage/docs/long_distance_20261101/release_receipt_T272.json').read_text())['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p={k:prior[k] for k in ['checkpoint','initial_base','policy_hash','job']}
    p.update(experiment='T273',scope='Fixed same-native-state value direction calibration only',start_step=18000,
        snapshot=str(ROOT/'stage/runs/t269_same_state_controls/anchor_18000.npz'),
        context=str(ROOT/'stage/runs/t269_same_state_controls/context_18000.json'),reference=reference,
        event_checkpoint=str(ROOT/'stage/runs/t272_event_value_pilot/event_checkpoint.pt.gz'),
        marginal_sigma_radius=3.,scoring_horizon_seconds=16,wall_seconds=300,max_commands=6,max_physics=60000,
        episode_bytes=384*1024,metadata_bytes=768*1024,final_bytes=3*1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        commands_order=['control','control_exact_repeat','up','up_exact_repeat','down','down_exact_repeat'],
        contract='One known six-metre LEFT state at36s. Offline locked T272 head double-precision gradient of Fgoal(16)-Fcontact(16) under original current120+one native action; positive and negative three marginal-sigma ellipsoid directions, bounded and rounded once. No radius/state/noise/seed search. Scorer evaluated only before native simulation, not a runtime teacher/planner/router; native worker receives literal first-hold action and then uses the same frozen T267 Actor. Original6m goal/.2radius/56s absolute cap,100nativehold/.002physics unchanged. Restore full T269 state/observation/previous-action/hold; two controls must match every10array/integration before any new intervention. Every unique action repeats independently exact; six commands total, zero learning. New interventions are diagnostic, never BC/on-policy training or deployment. Actual sixteen-second event score comparison requires up>=control>=down and up>down, ties are not evidence. One context cannot establish expectation calibration, global ranking, old-task safety or six-metre stability. Keep all failures; no automatic Actor coupling or7/8. Phase isolated.')
    p['proposals']=propose(p)
    paths=set(prior['input_hashes'])
    paths.update([p['snapshot'],p['context'],reference['path'],p['event_checkpoint'],
        str(ROOT/'stage/runs/t269_same_state_controls/protocol.json'),str(ROOT/'stage/runs/t269_same_state_controls/cases.jsonl'),
        str(ROOT/'stage/runs/t272_event_value_pilot/protocol.json'),str(ROOT/'stage/runs/t272_event_value_pilot/summary.json'),
        str(ROOT/'stage/docs/long_distance_20261101/T272_AUDIT.json'),
        str(ROOT/'stage/docs/long_distance_20261101/T272_EXTERNAL_DIAGNOSTIC.json'),
        str(ROOT/'stage/docs/long_distance_20261101/T272_STRATIFIED.json'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T272.json'),
        str(ROOT/'stage/experiments/t272_event_value_pilot/event_model.py'),*map(str,HERE.glob('*.py'))])
    p['input_hashes']={s:C.sha(s) for s in sorted(paths)}
    assert S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();C.save(HERE/'protocol.json',p);C.save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',proposals=p['proposals']['score16'],actions=p['proposals']['actions'])),flush=True)


if __name__=='__main__':main()

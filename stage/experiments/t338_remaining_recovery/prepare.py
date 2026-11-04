from t338r_core import *

def main():
    assert not OUT.exists()
    partial=read(DOC/'T338_PARTIAL_AUDIT.json');assert partial['state']=='PASS_PARTIAL_EVIDENCE_ONLY_ORIGINAL_RUN_REMAINS_FAILED'
    for group in ['T338','T338-REVIEW']:
        assert read(ROOT/('stage/docs/long_distance_20261101/release_receipt_'+group+'.json'))['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=read(ROOT/'stage/experiments/t338_capg_recovery/protocol.json');s=read(OLD_OUT/'summary.json');launch=read(OLD_OUT/'launch.json')
    assert s['updates']==dict(live_actor=0,shadow_actor=0)
    rows=[json.loads(v) for v in (OLD_OUT/'cases.jsonl').read_text().splitlines()]
    cachedrows=[r for r in rows if r['role']=='training'];assert len(cachedrows)==4
    p.update(experiment='T338_REMAINING_BUDGET_NUMERIC_ENTRY_RECOVERY',
        prior_commands=s['commands'],prior_physics=s['physics_steps'],cached_training_rows=cachedrows,
        cached_replays=[json.loads(v) for v in (OLD_OUT/'replays.jsonl').read_text().splitlines() if json.loads(v)['id'] in {r['id'] for r in cachedrows}],
        max_commands=900-s['commands'],max_physics=24000000-s['physics_steps'],
        max_training_episodes=28,max_compact_cases=128,final_bytes=12*1024**2,payload_bytes=10*1024**2,metadata_bytes=2*1024**2,
        combined_final_bytes=14*1024**2,absolute_training_deadline_epoch=launch['training_deadline_epoch'],
        absolute_deadline_epoch=launch['native_deadline_epoch'],absolute_all_work_deadline_epoch=launch['launch_epoch']+7800,
        max_registered_commands=127,worst_registered_physics=4512000,
        additional_discarded_preflight_Adam=4,combined_discarded_preflight_Adam=8,
        original_scientific_config_unchanged=True,no_added_seed_or_round=True)
    assert time.time()<p['absolute_training_deadline_epoch']-180
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OLD_OUT)+p['final_bytes']<=p['combined_final_bytes']
    assert 28*p['training_episode_bytes']+4*p['partial_bytes']+2*p['model_bytes']+128*p['compact_bytes']+p['metadata_bytes']<p['final_bytes']
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(ROOT/'stage/experiments/t338_failure_review/canonical.py'),str(DOC/'T338_PARTIAL_AUDIT.json'),str(DOC/'T338R_PLAN.md'),str(OLD_OUT/'summary.json'),str(OLD_OUT/'launch.json'),str(OLD_OUT/'initial_baseline.json')}
    files|={r['path'] for r in cachedrows}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED_REMAINING_ONLY',train_end=p['absolute_training_deadline_epoch'],native_end=p['absolute_deadline_epoch'],max_new_commands=p['max_commands'],max_new_physics=p['max_physics'],combined_final_bytes=K.S.used(OLD_OUT)+p['final_bytes'])))

if __name__=='__main__':main()

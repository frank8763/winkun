from t339_core import *

def main():
    assert not OUT.exists()
    p=read(ROOT/'stage/experiments/t338_remaining_recovery/protocol.json')
    for group in ['T338R','T338R-REVIEW']:
        assert read(ROOT/('stage/docs/long_distance_20261101/release_receipt_'+group+'.json'))['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert read(DOC/'T338R_PARTIAL_AUDIT.json')['complete_success']==1
    old=ROOT/'stage/runs/t338_remaining_recovery'
    rows=[json.loads(v) for v in (old/'cases.jsonl').read_text().splitlines()]
    assert len(rows)==4 and all(r['arm']=='capg' and r['role']=='training' and r['complete'] for r in rows)
    assert sum(r['success'] for r in rows)==1 and not any(r['collision'] for r in rows)
    p.update(experiment='T339_CACHED_ONPOLICY_SUCCESS_TO_DETERMINISTIC_CONTROL',arms=['capg'],
        cached_training_rows=rows,frontiers=[240,280,200,150,0],test_prefixes=[240,280,200,150,0],
        epochs=8,max_live_Actor_Adam=8,max_shadow_Actor_Adam=8,preflight_discarded_Adam=2,
        final_bytes=8*1024**2,payload_bytes=6*1024**2,metadata_bytes=2*1024**2,
        training_episode_bytes=512*1024,partial_bytes=512*1024,model_bytes=512*1024,compact_bytes=8192,max_compact_cases=64,
        max_commands=64,max_physics=2400000,preflight_geometry_commands=1,
        max_registered_commands=40,worst_registered_physics=1560000,
        wall_seconds=1800,final_reserve_seconds=600,audit_seconds=600,all_work_seconds=3600,
        minimum_launch_window_seconds=1800,per_batch_admission_seconds=180,optimizer_admission_seconds=120,
        no_new_training_collection=True,no_new_seed_or_grid=True,no_7m_8m_training=True,
        contrast='Frozen initial deterministic control vs same CAPG Actor after exactly one8epoch PPO batch; no algorithm superiority comparison',
        purpose='First verify four cached native trajectories, then test whether one legitimate on-policy update learns deterministic recovery',
        protocol_version='new_user_authorized_experiment_20261104_not_old_budget_extension')
    for key in ['absolute_training_deadline_epoch','absolute_deadline_epoch','absolute_all_work_deadline_epoch',
                'combined_final_bytes','cached_replays','prior_commands','prior_physics','no_added_seed_or_round']:
        p.pop(key,None)
    p['additional_native_replay_only_not_new_training']=4
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(DOC/'T339_PLAN.md'),str(DOC/'T338R_PARTIAL_AUDIT.json'),str(old/'summary.json'),str(old/'protocol.json')}
    files|={row['path'] for row in rows}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    assert 5*p['training_episode_bytes']+4*p['partial_bytes']+p['model_bytes']+64*p['compact_bytes']+p['metadata_bytes']<p['final_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',full_reserved_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'],max_native_commands=40,new_training_collection=0,live_Actor_Adam=8)))

if __name__=='__main__':main()

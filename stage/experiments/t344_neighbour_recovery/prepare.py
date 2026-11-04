from t344_core import *

def main():
    assert not OUT.exists()
    for n in ['T340','T342','T343']:
        assert read(DOC/(n+'_AUDIT.json'))['state']=='PASS'
        assert read(ROOT/'stage/docs/long_distance_20261101'/('release_receipt_'+n+'.json'))['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    old=read(ROOT/'stage/experiments/t340_demo_supported_recovery/protocol.json')
    keep=['learning_protocol','initial_patch','initial_policy_hash','course_protocol','reference','input_mean','input_scale','base_protocol','old_jobs','course_dev_jobs','original_jobs','executed_std','tanh_sigma','seed','actor_lr','ppo_clip']
    p={k:old[k] for k in keep}
    source=next(json.loads(l) for l in (ROOT/'stage/runs/t342_balanced_consolidation/cases.jsonl').read_text().splitlines() if json.loads(l)['role']=='final_wide5' and json.loads(l)['job_id']=='wide_dev_1')
    assert source['success'] and not source['collision']
    sources=copy.deepcopy(old['demo_sources'])
    sources.append(dict(name='autonomous_center_T340_actual_own_trajectory',path=str(ROOT/'stage/runs/t341_wheel_speed_transfer/episodes/navigation_x1_primary.npz'),start=0,stop=336))
    p.update(experiment='T344_TWO_NEIGHBOUR_ONPOLICY_RECOVERY_PILOT',arms=['capg'],student_initial_model=str(ROOT/'stage/runs/t340_demo_supported_recovery/final.pt.gz'),
        student_initial_policy_hash='58f5551259496bbf6123bd7700ab5bce8dc9d58dbf47b9b2b1e94944a2d65e46',
        supporting_model=str(ROOT/'stage/runs/t342_balanced_consolidation/final.pt.gz'),supporting_source_row=source,supporting_policy_hash=source['policy_hash'],
        demo_sources=sources,training_job_ids=['wide_dev_1','wide_dev_2'],training_seed=20261104344,rounds=2,training_each_round=4,epochs=8,
        demo_rows_per_source=32,demo_rl_weight=.05,demo_decay=.5,live_Actor_Adam=16,shadow_Actor_Adam=16,preflight_discarded_Adam=4,
        wall_seconds=3600,final_reserve_seconds=1200,audit_seconds=1200,all_work_seconds=6000,optimizer_admission_seconds=120,per_batch_admission_seconds=90,
        max_commands=128,max_physics=4000000,preflight_geometry_commands=5,final_bytes=7*1024**2,
        full_bytes=384*1024,partial_bytes=384*1024,max_full_cases=9,model_bytes=256*1024,
        compact_bytes=4096,max_compact_cases=69,metadata_bytes=int(1.5*1024**2),payload_bytes=int(5.5*1024**2),
        global_bytes=1024**3,min_free_bytes=40*1024**3,original_full_gate_in_this_pilot=False,
        contract='New bounded recovery-coverage hypothesis,not extensionT340/342. Start exactT340final+freshAdam; sameCAPG120Actor/reward/actorLR/PPOclip/8epochs/actualstd/wide1.17m/80s/originalspeed5force120. Two fixed whole-episode start families wide_dev1(dx-.03),wide_dev2(dx+.03),one round4samples each,all8newtrain+8exactnativefull12/history/draw replay. Fixed human/selfsuccess support includes centerT340336rows plus one actually regenerated safeT342wide_dev1 trajectory+exactreplay; no arbitrary expert labels on learner failures/no demonstration onpolicy logp. Same-side group4 independent trajectories for timeLOO,allfailures in realRL. 16live16shadowmax,4discardedpreflight+audit16historical separate,rollback wholeActorAdam on interrupted block. Initialwide5; perroundwide5;finalold20/wide5/narrow2+27independent model native readback. Pilot never executesfull275/54/fresh16/reload78,missingnull/notstable6/no7or8. No planner/map/waypoint/runtime teacher/router/Phase. No seed/noise/weight/extra-round grid,allrawfailure retained.')
    p['registered_max_commands']=5+2+5+16+10+27+27
    p['registered_worst_physics']=(2+5+16+10+27+27)*40000
    assert p['registered_max_commands']==92<=p['max_commands'] and p['registered_worst_physics']<=p['max_physics']
    assert 9*p['full_bytes']+4*p['partial_bytes']+p['model_bytes']+69*p['compact_bytes']+p['metadata_bytes']<p['final_bytes']
    files=set(old['input_hashes'])|{str(f) for f in HERE.glob('*.py')}
    files|={p['student_initial_model'],p['supporting_model'],str(ROOT/'stage/runs/t342_balanced_consolidation/cases.jsonl'),str(ROOT/'stage/experiments/t340_demo_supported_recovery/protocol.json'),str(ROOT/'stage/experiments/t342_balanced_consolidation/protocol.json'),str(DOC/'T344_PLAN.md')}
    files|={r['path'] for r in sources}|{j['exact_snapshot'] for j in [*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs']]}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',new_training=8,source_regenerations=2,max_live_Actor_Adam=16,registered_commands=92,worst_physics=p['registered_worst_physics'],reserved_global=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'])))

if __name__=='__main__':main()

from t340_core import *

def main():
    assert not OUT.exists()
    assert read(DOC/'T339_AUDIT.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T339.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    old=read(ROOT/'stage/experiments/t339_success_to_control/protocol.json')
    keep=['learning_protocol','initial_patch','initial_policy_hash','course_protocol','reference','input_mean','input_scale','base_protocol',
          'old_jobs','course_dev_jobs','original_jobs','final_old_jobs','executed_std','tanh_sigma','seed','ppo_clip','actor_lr','prefix_native_references']
    p={k:old[k] for k in keep};gate=read(ROOT/'stage/experiments/t312_online_goal_ppo/protocol.json')
    p['six_jobs']=read(p['base_protocol'])['development_jobs'];p['fresh_jobs']=gate['heldout_jobs']
    p['reload_jobs']=[*gate['old_centers'],*p['six_jobs'],*p['fresh_jobs']]
    success=next(r for r in old['cached_training_rows'] if r['success'])
    p.update(experiment='T340_DEMO_SUPPORTED_CAPG_ONPOLICY_RECOVERY_ENGINEERING_PILOT',arms=['capg'],
        demo_sources=[dict(name='human_full',path=str(ROOT/'stage/runs/t328_wide_gap_80s/attempt_01.npz'),start=0,stop=320),
                      dict(name='human_correction',path=p['reference'],start=150,stop=318),
                      dict(name='self_success',path=success['path'],start=0,stop=76)],
        source_success_row=success,frontiers=[240,150,0],rounds=4,training_each_round=4,epochs=8,
        warm_steps=300,warm_lr=.001,demo_rows_per_source=32,demo_rl_weight=.05,demo_decay=.5,
        max_live_warm_Adam=300,max_shadow_warm_Adam=300,max_live_Actor_Adam=32,max_shadow_Actor_Adam=32,
        preflight_discarded_Adam=4,wall_seconds=5400,final_reserve_seconds=1500,audit_seconds=1800,all_work_seconds=8400,
        per_batch_admission_seconds=180,optimizer_admission_seconds=120,max_commands=600,max_physics=18000000,
        final_bytes=13*1024**2,payload_bytes=int(10.5*1024**2),metadata_bytes=int(2.5*1024**2),
        training_episode_bytes=384*1024,partial_bytes=384*1024,compact_bytes=4096,max_compact_cases=480,
        model_bytes=512*1024,max_training_episodes=16,preflight_geometry_commands=1,
        global_bytes=1024**3,min_free_bytes=40*1024**3,
        no_original_scene_or_speed_or_time_change=True,retain_research_updates_without_old_task_rollback=True,
        pilot_scope='Joint engineering change:unclipped-location imitation warmstart+human/self-success support+fresh onpolicy updates+physical prefix curriculum;not isolated algorithm attribution')
    assert len(p['final_old_jobs'])==275 and len(p['six_jobs'])==54 and len(p['fresh_jobs'])==16 and len(p['reload_jobs'])==78
    p['max_registered_commands']=1+3+32+12+27+275+54+16+78
    p['worst_registered_physics']=3*40000+32*40000+12*40000+27*40000+(275+54+16+78)*28000
    assert p['max_registered_commands']<=p['max_commands'] and p['worst_registered_physics']<=p['max_physics']
    assert 16*p['training_episode_bytes']+4*p['partial_bytes']+2*p['model_bytes']+480*p['compact_bytes']+p['metadata_bytes']<p['final_bytes']
    files=set(old['input_hashes'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(DOC/'T340_PLAN.md'),str(DOC/'T339_AUDIT.json'),str(ROOT/'stage/experiments/t312_online_goal_ppo/protocol.json'),str(ROOT/'stage/runs/t339_success_to_control/preflight.json')}
    files|={v['path'] for v in p['demo_sources']}
    files|={j['exact_snapshot'] for j in [*p['final_old_jobs'],*p['six_jobs'],*p['fresh_jobs'],*p['reload_jobs']]}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',reserved_global_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'],max_training=16,max_registered_commands=p['max_registered_commands'],worst_registered_physics=p['worst_registered_physics'])))

if __name__=='__main__':main()

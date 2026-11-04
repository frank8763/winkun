from t338_core import *

def main():
    assert not OUT.exists()
    assert read(DOC/'T337_AUDIT.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T337.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    old=read(ROOT/'stage/experiments/t336_reverse_online/protocol.json')
    keep=['learning_protocol','initial_patch','initial_policy_hash','course_protocol','reference','input_mean','input_scale',
        'global_bytes','min_free_bytes','base_protocol','course_dev_jobs','original_jobs','old_jobs','final_old_jobs']
    p={k:old[k] for k in keep}
    historical=[json.loads(v) for v in (ROOT/'stage/runs/t336_reverse_online/cases.jsonl').read_text().splitlines()]
    p['prefix_native_references']={str(k):next(r['path'] for r in historical if r['role']=='training' and r['arm']=='reverse' and r['prefix']==k) for k in [240,280]}
    p.update(experiment='T338_PAIRED_EXECUTED_ACTION_CAPG_PPO_RECOVERY',seed=20261104338,arms=['tanh','capg'],
        frontiers=[240,280,0],fixed_training_prefix=240,rounds=4,training_each_round=4,epochs=8,
        actor_lr=.0003,ppo_clip=.2,tanh_sigma=math.exp(-2.3),executed_std=[.035,.030],
        boundary_midpoint=EDGE,entropy_coefficient=0.,suffix_max_decisions=160,
        final_bytes=14*1024**2,payload_bytes=12*1024**2,metadata_bytes=2*1024**2,
        training_episode_bytes=224*1024,partial_bytes=224*1024,model_bytes=512*1024,compact_bytes=4096,max_compact_cases=512,
        max_training_episodes=32,max_live_Actor_Adam=64,max_shadow_Actor_Adam=64,preflight_discarded_Adam=4,
        wall_seconds=5400,final_reserve_seconds=1800,audit_seconds=1200,all_work_seconds=7800,
        max_commands=900,max_physics=24000000,preflight_geometry_commands=1,
        no_full_six_gate_or_promotion=True,no_7m_8m_training=True,
        contrast='Distribution and location parametrization/calibration as one joint factor; same PPO-MC acquisition without old-task rollback in both arms.',
        estimator='physical-time-aligned other-three-trajectory reward-to-go baseline; zero after peer termination; MC only; no Q/V/BC/entropy',
        scope='fixed48s wide80 recovery;56s control; original start/narrow tests separate; final old20 diagnostic, NOT old275 retention proof')
    # Worst registered command allocation, including every baseline and replay.
    p['max_registered_commands']=1+275+4+64+16+2*(20+5+2)
    p['worst_registered_physics']=275*28000+4*40000+64*40000+16*40000+2*(20*28000+5*40000+2*28000)
    assert p['max_registered_commands']<=p['max_commands'] and p['worst_registered_physics']<=p['max_physics']
    assert 32*p['training_episode_bytes']+4*p['partial_bytes']+2*p['model_bytes']+512*p['compact_bytes']+p['metadata_bytes']<p['final_bytes']
    files=set(old['input_hashes'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(DOC/n) for n in ['T338_PLAN.md','RESEARCH_DECISION.md','T337_AUDIT.json']}
    files|={str(ROOT/'stage/experiments/t336_reverse_online/protocol.json'),p['initial_patch'],p['reference']}
    files|=set(p['prefix_native_references'].values())|{str(ROOT/'stage/runs/t336_reverse_online/cases.jsonl')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',worst_commands=p['max_registered_commands'],worst_physics=p['worst_registered_physics'],full_reservation_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'])))

if __name__=='__main__':main()

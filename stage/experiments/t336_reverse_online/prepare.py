from t336_core import *
from datetime import datetime
from zoneinfo import ZoneInfo

def main():
    assert not OUT.exists();p=read(ROOT/'stage/experiments/t333_remaining_aggregation/protocol.json')
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T335.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert read(ROOT/'stage/docs/distance_20261103/T335_AUDIT.json')['state']=='PASS'
    patch=read(ROOT/'stage/runs/t333_remaining_aggregation/patches.json')['recovery_candidate']
    def epoch(text):return datetime.fromisoformat(text).replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    p.update(experiment='T336_NATIVE_REVERSE_CURRICULUM_PAIRED_SAC',
        learning_protocol=str(ROOT/'stage/experiments/t333_remaining_aggregation/protocol.json'),
        initial_patch=patch['path'],initial_policy_hash=patch['policy_hash'],
        course_protocol=str(ROOT/'stage/experiments/t331_policy_prefix_retry/protocol.json'),
        reference=str(ROOT/'stage/runs/t331_policy_prefix_retry/attempt_01.npz'),
        prior_trajectories=[str(ROOT/'stage/runs/t328_wide_gap_80s/attempt_01.npz'),str(ROOT/'stage/runs/t331_policy_prefix_retry/attempt_01.npz')],
        seed=20261103336,arms=['fixed','reverse'],frontiers=[280,240,200,160,120,80,40,0],
        rounds=8,training_each_round=4,advance_last_safe=2,consecutive_rejections_stop=2,
        critic_warmup_steps=1000,updates_per_round=1000,batch_size=128,old_rows_per_task=16,
        initial_logstd=-2.3,actor_lr=.0001,critic_lr=.0003,alpha=.001,tau=.005,retention_weight=100.,
        absolute_training_deadline_epoch=epoch('2026-11-04T06:30:00'),
        absolute_deadline_epoch=epoch('2026-11-04T07:15:00'),
        absolute_all_work_deadline_epoch=epoch('2026-11-04T08:00:00'),
        wall_seconds=34200,final_reserve_seconds=2700,audit_seconds=3600,
        max_commands=2000,max_physics=60000000,preflight_geometry_commands=1,
        final_bytes=22*1024**2,payload_bytes=14*1024**2+2*1280*1024+832*4096,metadata_bytes=2*1024**2,
        training_episode_bytes=224*1024,compact_bytes=4096,model_bytes=1280*1024,max_compact_cases=832,
        max_training_episodes=64,max_live_actor_Adam=16000,max_live_critic_Adam=18000,
        preflight_synthetic_Adam=20,scientific_control_not_prevalidated=True)
    # Core experiment gate: the original narrow 56s task remains untouched.
    base=read(p['base_protocol']);p['old_jobs']=base['acceptance_guard_jobs']
    gate_path=ROOT/'stage/experiments/t312_online_goal_ppo/protocol.json';gate=read(gate_path)
    p['final_old_jobs']=[*gate['old_development'],*gate['old_regression64']];p['six_jobs']=base['development_jobs']
    p['fresh_jobs']=gate['heldout_jobs'];p['reload_jobs']=[*gate['old_centers'],*p['six_jobs'],*p['fresh_jobs']]
    assert len(p['final_old_jobs'])==275 and len(p['six_jobs'])==54 and len(p['fresh_jobs'])==16 and len(p['reload_jobs'])==78
    p['full_gate_candidate_rule']='most wide5 safe, then most original2 safe, then reverse on exact ties; one selected arm only'
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(receipt),str(ROOT/'stage/docs/distance_20261103/T335_AUDIT.json'),str(DOC/'PLAN.md'),
        p['learning_protocol'],p['initial_patch'],p['course_protocol'],p['reference'],*p['prior_trajectories'],str(gate_path)}
    files|={j['exact_snapshot'] for j in [*p['final_old_jobs'],*p['six_jobs'],*p['fresh_jobs'],*p['reload_jobs']]}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert time.time()<p['absolute_training_deadline_epoch']-1800
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',full_reservation_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'])))

if __name__=='__main__':main()

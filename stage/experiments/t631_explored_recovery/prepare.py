"""Prepare finite explored recoveries and conditional sampled-action learning."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/day_20261118/t631'


def main():
    read=lambda p:json.loads(Path(p).read_text())
    p=read(ROOT/'stage/runs/t630_early_departure_recovery/protocol.json')
    old=read(ROOT/'stage/runs/t630_early_departure_recovery/summary.json')
    prior=read(ROOT/'stage/runs/t629_offline_night_relay/recovery/summary.json')['result']
    jobs=[dict(r['candidate_source']['request']['job'],source_path=r['candidate_source']['path'],
               student_takeover_first=r['departure']['first'],explore_after_prefix=True)
          for r in old['results']]
    assert len(jobs)==2 and all(not r['qualified'] for r in old['results'])
    anchors=[r for r in prior['candidate_source_rows'] if r['job_id'] in p['new_ids']]
    assert len(anchors)==2 and all(r['success'] and not r['collision'] and r['field_detail']['field_qualified_success'] for r in anchors)
    total=10*sum(j['case']['step_cap'] for j in jobs)
    total+=sum(j['case']['step_cap'] for j in p['evaluation_jobs'])
    total+=sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids'])
    p.update(id='T631_EXPLORED_RECOVERY',out=str(ROOT/'stage/runs/t631_explored_recovery'),
             authorization_approved=False,authorization_file=str(DOC/'AUTHORIZATION.json'),
             student_initial=p['recovery_fallback_model'],recovery_jobs=jobs,new_anchor_rows=anchors,
             seed=63120261118,raw_sigma=[.2,.2],recovery_steps=64,Actor_each_cap=64,
             new_actor_updates_cap=64,actor_steps=0,new_Critic=0,new_optimizer_preflight=0,
             max_physics=total,max_commands=79,main_seconds=1800,max_main_seconds=1800,
             audit_seconds=900,transport_seconds=300,source_seconds=180,max_bytes=32*2**20,
             absolute_learning_stop=0,absolute_native_stop=0,absolute_all_stop=0,
             relative_all_stop_seconds=3600,no_new_training=False,
             no_automatic_learning_after_qualification=False,
             no_automatic_RL_after_imitation=True,new_remote_raw_bytes_cap=32*2**20,
             actor_updates_new_cap=64,shadow_actor_updates_cap=64,
             historical_actor_updates_cap=64,critic_updates_new_cap=0,
             preflight_discarded_Adam_cap=0,discarded_optimizer_preflight_updates=0,
             optimizer_class='SGD',new_RL=0,rounds=1,
             scope='T631 fixed eight actual-state recovery explorations; conditional 64-step sampled-success supervision; no mixed on-policy RL',
             day_shared_accounting='Independent T631 quota requires explicit new approval; never reuse T630 zero-update or expired night quota')
    # Stage T630's zero-update authorization must not be imported as T631 quota.
    for key in ['authorization_sha256','registration_epoch','reset_jobs','candidate_sources']:
        p.pop(key,None)
    for f in HERE.glob('*.py'):
        if f.name!='prepare.py':p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for f in [ROOT/'stage/runs/t630_early_departure_recovery/summary.json',
              ROOT/'stage/runs/t630_early_departure_recovery/audit.json',
              ROOT/'stage/runs/t630_early_departure_recovery/SEALED.json']:
        p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for r in anchors:p['inputs'][r['path']]=r['sha256']
    for f,h in p['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    DOC.mkdir(parents=True,exist_ok=True)
    (HERE/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    full=8*2**21;partial=3*2**21;compact=51*p['compact_bytes']
    bound=full+partial+compact+p['model_bytes']+6*2**20
    assert bound<p['max_bytes'] and used+p['max_bytes']+131072<2**30
    b=dict(state='PREPARED_NOT_APPROVED_NOT_LAUNCHED',max_physics=total,max_commands=79,
           Actor_live=64,Actor_shadow=64,Actor_independent_history=64,Critic=0,optimizer_preflight=0,
           main_seconds=1800,audit_seconds=900,all_seconds_from_registration=3600,
           local_max_bytes=p['max_bytes'],full_arrays_bound=full,partial_bound=partial,
           compact_bound=compact,model_bound=p['model_bytes'],metadata_reserve=6*2**20,
           physical_storage_worst_bound=bound,global_current_bytes=used,
           global_worst_bytes=used+p['max_bytes']+131072,global_cap=2**30,
           CPU_percent_cap=400,CPU=[8,10,12,14],GPU_float64_learning_only=True,
           RAM_cap=4*2**30,swap_cap=0)
    (DOC/'BUDGET.json').write_text(json.dumps(b,indent=2)+'\n')
    print(json.dumps(b))


if __name__=='__main__':main()

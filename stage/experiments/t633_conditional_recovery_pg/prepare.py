"""Prepare only; new suffix PG quota is independent of completed T632."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/day_20261118/t633'


def main():
    read=lambda f:json.loads(Path(f).read_text())
    p=read(ROOT/'stage/runs/t632_taskwise_recovery/protocol.json')
    s=read(ROOT/'stage/runs/t632_taskwise_recovery/summary.json')['result']
    assert not s['accepted'] and s['final49']['success']==34
    student=s['proposal'];jobs=p['evaluation_jobs']
    anchor=next(j for j in jobs if j['id']=='t502_field_sector1_offset2')
    recovery=dict(p['field_handoff_job'],explore_after_prefix=True)
    total=10*recovery['case']['step_cap']+2*anchor['case']['step_cap']
    total+=sum(j['case']['step_cap'] for j in jobs)+sum(j['case']['step_cap'] for j in jobs if j['id'] in p['key_replay_ids'])
    p.update(id='T633_CONDITIONAL_RECOVERY_PG',out=str(ROOT/'stage/runs/t633_conditional_recovery_pg'),
             authorization_approved=False,authorization_file=str(DOC/'AUTHORIZATION.json'),
             student_initial=student,student_success_ids=s['final49']['success_ids'],
             recovery_job=recovery,new_anchor_job=anchor,actor_steps=8,recovery_steps=8,
             Actor_each_cap=8,new_actor_updates_cap=8,actor_updates_new_cap=8,
             shadow_actor_updates_cap=8,historical_actor_updates_cap=8,
             max_physics=total,max_commands=71,max_bytes=24*2**20,new_remote_raw_bytes_cap=32*2**20,
             main_seconds=1800,max_main_seconds=1800,audit_seconds=900,seed=63320261118,
             absolute_learning_stop=0,absolute_native_stop=0,absolute_all_stop=0,
             new_RL=8,new_teacher_queries=0,new_Critic=0,new_optimizer_preflight=0,
             scope='One fresh current-T632 four-trajectory conditional suffix PG batch; one fresh successful field1 anchor; 8SGD shadow/history; original34 union student34 retention and autonomous field recovery required')
    for key in ['registration_epoch','field_recovery_rows','new_anchor_rows','field_handoff_job']:
        p.pop(key,None)
    for f in HERE.glob('*.py'):
        if f.name!='prepare.py':p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for f in [ROOT/'stage/runs/t632_taskwise_recovery/summary.json',ROOT/'stage/runs/t632_taskwise_recovery/audit.json',
              ROOT/'stage/runs/t632_taskwise_recovery/SEALED.json',Path(student['path'])]:
        p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for f,h in p['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    DOC.mkdir(parents=True,exist_ok=True);(HERE/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    bound=5*2**21+3*2**21+50*p['compact_bytes']+p['model_bytes']+6*2**20
    assert bound<p['max_bytes'] and used+p['max_bytes']+131072<2**30
    b=dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',max_physics=total,max_commands=71,
           Actor_live=8,Actor_shadow=8,Actor_independent_history=8,Critic=0,optimizer_preflight=0,
           main_seconds=1800,audit_seconds=900,relative_all_stop_seconds=3600,
           local_max_bytes=p['max_bytes'],physical_storage_worst_bound=bound,
           global_current_bytes=used,global_worst_bytes=used+p['max_bytes']+131072,global_cap=2**30,
           CPU_percent_cap=400,CPU=[8,10,12,14],GPU_float64_learning_only=True,RAM_cap=4*2**30,swap_cap=0)
    (DOC/'BUDGET.json').write_text(json.dumps(b,indent=2)+'\n');print(json.dumps(b))


if __name__=='__main__':main()

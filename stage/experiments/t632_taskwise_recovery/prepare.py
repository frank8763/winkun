"""Prepare the taskwise learner only; never register, simulate or optimize."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/day_20261118/t632'


def main():
    read=lambda f:json.loads(Path(f).read_text())
    p=read(ROOT/'stage/runs/t631_explored_recovery/protocol.json')
    prior=read(ROOT/'stage/runs/t631_explored_recovery/summary.json')
    assert prior['Actor_updates_started']==dict(live=0,shadow=0)
    field=[r for r in prior['result']['exploration'] if r['job_id']=='t502_field_sector4_offset2'
           and r['success'] and not r['collision'] and r['field_detail']['field_qualified_success']]
    assert len(field)==3
    old=read(ROOT/'stage/runs/t630_early_departure_recovery/summary.json')
    nominal=next(r['baseline'] for r in old['results'] if r['id']=='wide117_8m_dev6_LEFT_13')
    assert nominal['success'] and not nominal['collision'] and nominal['field_detail']['field_qualified_success']
    handoff=dict(field[0]['request']['job'],explore_after_prefix=False)
    total=sum(j['case']['step_cap'] for j in p['evaluation_jobs'])
    total+=sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids'])
    total+=2*handoff['case']['step_cap']
    p.update(id='T632_TASKWISE_RECOVERY',out=str(ROOT/'stage/runs/t632_taskwise_recovery'),
             authorization_approved=False,authorization_file=str(DOC/'AUTHORIZATION.json'),
             field_recovery_rows=field,nominal_eight_row=nominal,field_handoff_job=handoff,
             max_physics=total,max_commands=61,main_seconds=1800,max_main_seconds=1800,
             audit_seconds=900,transport_seconds=300,source_seconds=180,max_bytes=16*2**20,
             new_remote_raw_bytes_cap=24*2**20,new_teacher_queries=0,new_RL=0,
             absolute_learning_stop=0,absolute_native_stop=0,absolute_all_stop=0,
             scope='T632 one 64-SGD task-balanced fit: three qualified field recoveries, verified nominal eight-metre trajectory, two new success anchors, old33 function retention; one handoff plus full49; no new teacher or exploration')
    for key in ['registration_epoch','authorization_sha256','recovery_jobs']:
        p.pop(key,None)
    for f in HERE.glob('*.py'):
        if f.name!='prepare.py':p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for name in ['t630_early_departure_recovery','t631_explored_recovery']:
        for filename in ['summary.json','audit.json','SEALED.json']:
            f=ROOT/'stage/runs'/name/filename;p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for row in [*field,nominal]:p['inputs'][row['path']]=row['sha256']
    for f,h in p['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    DOC.mkdir(parents=True,exist_ok=True)
    (HERE/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    bound=2**21+3*2**21+49*p['compact_bytes']+p['model_bytes']+6*2**20
    assert bound<p['max_bytes'] and used+p['max_bytes']+131072<2**30
    b=dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',max_physics=total,max_commands=61,
           Actor_live=64,Actor_shadow=64,Actor_independent_history=64,Critic=0,
           optimizer_preflight=0,new_teacher_queries=0,new_RL=0,main_seconds=1800,
           audit_seconds=900,relative_all_stop_seconds=3600,local_max_bytes=p['max_bytes'],
           physical_storage_worst_bound=bound,global_current_bytes=used,
           global_worst_bytes=used+p['max_bytes']+131072,global_cap=2**30,
           CPU_percent_cap=400,CPU=[8,10,12,14],GPU_float64_learning_only=True,
           RAM_cap=4*2**30,swap_cap=0)
    (DOC/'BUDGET.json').write_text(json.dumps(b,indent=2)+'\n');print(json.dumps(b))


if __name__=='__main__':main()

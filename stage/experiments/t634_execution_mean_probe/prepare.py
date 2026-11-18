"""Freeze a finite zero-training readout test and partial old evidence audit."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/day_20261118/t634'


def main():
    read=lambda f:json.loads(Path(f).read_text())
    p=read(ROOT/'stage/runs/t633_conditional_recovery_pg/protocol.json')
    s=read(ROOT/'stage/runs/t632_taskwise_recovery/summary.json')['result']
    ids=['t502_field_sector4_offset2','t502_field_sector1_offset2','wide117_8m_dev6_LEFT_13']
    controls=[next(r for r in s['evaluation_rows'] if r['job_id']==id) for id in ids]+[s['handoffs'][0]]
    jobs=[r['request']['job'] for r in controls]
    total=4*sum(j['case']['step_cap'] for j in jobs)
    total+=sum(j['case']['step_cap'] for j in p['evaluation_jobs'])
    total+=sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids'])
    p.update(id='T634_EXECUTION_MEAN_PROBE',out=str(ROOT/'stage/runs/t634_execution_mean_probe'),
             authorization_approved=False,authorization_file=str(DOC/'AUTHORIZATION.json'),
             sentinel_jobs=jobs,control_reference_rows=controls,
             new_live_Actor=0,new_shadow_Actor=0,independent_previous_history_SGD=8,
             Actor_each_cap=0,actor_steps=0,recovery_steps=0,new_RL=0,new_Critic=0,
             new_actor_updates_cap=0,actor_updates_new_cap=0,shadow_actor_updates_cap=0,
             historical_actor_updates_cap=8,new_optimizer_preflight=0,
             max_physics=total,max_commands=75,max_bytes=32*2**20,new_remote_raw_bytes_cap=40*2**20,
             main_seconds=1200,max_main_seconds=1200,audit_seconds=600,previous_audit_seconds=600,
             absolute_learning_stop=0,absolute_native_stop=0,absolute_all_stop=0,
             scope='Frozen T632 mode versus exact clipped-Gaussian expected action;4 fixed states each+exact, conditional49;0 new live training. Separate8-step historical T633 reconstruction reports PARTIAL due overwritten compact.')
    for key in ['registration_epoch','recovery_job','new_anchor_job']:
        p.pop(key,None)
    for f in HERE.glob('*.py'):
        if f.name!='prepare.py':p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for path in ['stage/runs/t633_conditional_recovery_pg/summary.json','stage/runs/t633_conditional_recovery_pg/cases.jsonl.gz',
                 'stage/runs/t633_conditional_recovery_pg/SEALED.json','stage/runs/t633_conditional_recovery_pg/LEARNING_COMPLETE.json']:
        f=ROOT/path;p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for f,h in p['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    DOC.mkdir(parents=True,exist_ok=True);(HERE/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    bound=8*2**21+3*2**21+49*p['compact_bytes']+8*2**20
    assert bound<p['max_bytes'] and used+p['max_bytes']+131072<2**30
    b=dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',max_physics=total,max_commands=75,
           live_Actor=0,shadow_Actor=0,new_Critic=0,independent_previous_history_SGD=8,
           main_seconds=1200,audit_seconds=600,previous_audit_seconds=600,relative_all_stop_seconds=3600,
           local_max_bytes=p['max_bytes'],physical_storage_worst_bound=bound,
           global_current_bytes=used,global_worst_bytes=used+p['max_bytes']+131072,global_cap=2**30,
           CPU=[8,10,12,14],CPU_percent_cap=400,RAM_cap=4*2**30,swap_cap=0,
           GPU_only_for_old_discarded_history_reconstruction=True)
    (DOC/'BUDGET.json').write_text(json.dumps(b,indent=2)+'\n');print(json.dumps(b))


if __name__=='__main__':main()

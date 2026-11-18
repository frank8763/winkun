"""Prepare a zero-update qualification, never register or start simulation."""
from pathlib import Path
import copy
import hashlib
import json
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/day_20261118/t630'


def main():
    read=lambda p:json.loads(Path(p).read_text())
    p=copy.deepcopy(read(ROOT/'stage/runs/t629_offline_night_relay/protocol.json'))
    recovery=read(ROOT/'stage/runs/t629_offline_night_relay/recovery/summary.json')['result']
    old=['t502_field_sector4_offset2','wide117_8m_dev6_LEFT_13']
    sources=[r for r in recovery['candidate_source_rows'] if r['job_id'] in old]
    assert len(sources)==2 and all(not r['success'] for r in sources)
    failed=next(j for c in p['neighborhood_chunks'] for j in c
                if j['id']=='t502_field_sector1_offset2_T629_dx0.0_yaw2.0')
    control=next(j for j in p['profile_jobs'] if j['id']=='t502_field_sector1_offset2')
    # Per old task: baseline + exact + one selected mixed recovery + exact.
    physics=4*sum(r['request']['job']['case']['step_cap'] for r in sources)
    physics+=2*(failed['case']['step_cap']+control['case']['step_cap'])
    p.update(id='T630_EARLY_DEPARTURE_RECOVERY',out=str(ROOT/'stage/runs/t630_early_departure_recovery'),
             authorization_approved=False,authorization_file=str(DOC/'AUTHORIZATION.json'),
             candidate_sources=sources,reset_jobs=[failed,control],
             max_physics=physics,max_commands=18,main_seconds=900,max_main_seconds=900,
             audit_seconds=600,transport_seconds=300,source_seconds=180,
             max_bytes=24*2**20,full_bytes=2**21,fresh_bytes=2**21,
             Actor_each_cap=0,new_actor_updates_cap=0,actor_steps=0,recovery_steps=0,
             new_Critic=0,new_optimizer_preflight=0,allow_final407=False,
             absolute_learning_stop=0,absolute_native_stop=0,absolute_all_stop=0,
             relative_all_stop_seconds=3600,stage_cpus=[8,10,12,14],
             no_new_training=True,no_automatic_learning_after_qualification=True,
             source_remote_required=True)
    for key in ['absolute_learning_stop','absolute_native_stop','absolute_all_stop']:
        assert p[key]==0
    for f in HERE.glob('*.py'):
        if f.name!='prepare.py':p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for r in sources:p['inputs'][r['path']]=r['sha256']
    for f in [ROOT/'stage/runs/t629_offline_night_relay/SEALED.json',
              ROOT/'stage/runs/t629_offline_night_relay/recovery/summary.json']:
        p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for f,h in p['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    DOC.mkdir(parents=True,exist_ok=True)
    (HERE/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    assert used+p['max_bytes']+131072<2**30
    budget=dict(state='PREPARED_NOT_APPROVED_NOT_LAUNCHED',physics=physics,commands=18,
                new_optimizer=0,max_new_local_bytes=p['max_bytes'],current_global_bytes=used,
                worst_global_bytes=used+p['max_bytes']+131072,global_cap=2**30,
                max_main_seconds=900,max_audit_seconds=600,
                proposed_all_stop_seconds_from_launch=3600,
                CPU_physical_cores=[8,10,12,14],CPU_percent_cap=400,GPU=False,
                actual_unique_RAM_bytes_cap=4*2**30,swap_bytes_cap=0,
                old_night_remaining_quota_not_reused=True)
    (DOC/'BUDGET.json').write_text(json.dumps(budget,indent=2)+'\n')
    print(json.dumps(budget))


if __name__=='__main__':main()

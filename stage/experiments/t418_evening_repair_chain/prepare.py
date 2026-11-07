"""Register the fixed, offline-capable evening mechanism chain; no launch here."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,time,shutil,ast

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t418_evening_repair_chain';DOC=ROOT/'stage/docs/day_20261107'
ORIGIN=ROOT/'stage/runs/t417_representation_adaptation'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def used(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())

def budget(p,base,phase,rounds):
    """Upper bounds include every conditional full validation and replay."""
    joint=[*p['old8_jobs'],*p['left5_jobs'],p['right_job']]
    old6=next(j for j in p['old8_jobs'] if j['id']=='dev_task6_13')
    left3=next(j for j in p['left5_jobs'] if j['id']=='wide_dev_3')
    repair_jobs=[old6,left3]
    jobs=[*joint]  # all14 initial full histories, including failures
    if phase!='rl':
        # Two positive controls + exact replay; then at most two queried
        # recoveries + exact replay per round. Failed jobs are fixed, not a grid.
        jobs+=repair_jobs*2
        for _ in range(rounds):jobs+=repair_jobs*2+joint+[left3]*8+joint
        full_sources=14+2+2*rounds+4*rounds
        full_bytes=14*p['full_bytes']+(2+2*rounds)*p['query_full_bytes']+4*rounds*p['full_bytes']
        model_count=2+3*rounds # stable before, BC, RL proposals; sampling/final
        exact_count=2+2*rounds+4*rounds+2+78
    else:
        for _ in range(rounds):jobs+=[left3]*8+joint
        full_sources=14+4*rounds
        full_bytes=(14+4*rounds)*p['full_bytes']
        # No stable before snapshot needed: no new teachers in this phase.
        model_count=2+rounds
        exact_count=4*rounds+2+78
    jobs+=joint+list(p['centres'])*2+p['main_jobs']+base['final_old_jobs']+p['fresh_jobs']+p['reload_jobs']
    compact_sources=len(jobs)-full_sources-exact_count
    assert compact_sources>=0
    physics=sum(j['case']['step_cap'] for j in jobs)
    commands=4+len(jobs) # four native worker setup commands
    # Queries are issued serially; at most one worker can hold the larger
    # teacher tape while the other three require ordinary full-payload reserve.
    partial_reserve=(3*p['full_bytes']+p['query_full_bytes']) if phase!='rl' else 4*p['full_bytes']
    bytes_worst=full_bytes+compact_sources*5*1024+model_count*512*1024+768*1024+partial_reserve
    return dict(phase=phase,max_rounds=rounds,worst_physics=physics,worst_main_commands=commands,
        worst_own_bytes=bytes_worst,full_source_count=full_sources,exact_native_count_max=exact_count,
        compact_source_count_max=compact_sources,model_count_max=model_count,
        model_bytes=512*1024,metadata_bytes=768*1024,partial_reserve_bytes=partial_reserve)

def prepare():
    assert not OUT.exists()
    old=read(ORIGIN/'summary.json');assert old['state']=='COMPLETE'
    assert read(ORIGIN/'audit.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T417.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for name in ['launch.json','driver_started.json','main_started.json','workers.json']:
        row=read(ORIGIN/name)
        for pid in [row[k] for k in ['pid','parent'] if type(row.get(k)) is int]+row.get('workers',[]):
            assert not Path('/proc',str(pid)).exists(),pid
    p=read(ORIGIN/'protocol.json');base=read(p['source_protocol'])
    prior=p['prior_day_actual'].copy()
    prior['physics']+=old['physics'];prior['commands']+=old['commands']+8;prior['main_wall_s']+=old['elapsed_s']
    proposal=old['rounds'][0]['bc']['proposed_model'];assert not old['rounds'][0]['bc']['accepted']
    assert sha(proposal['path'])==proposal['sha256']
    ids=old['rounds'][0]['bc']['checks']['success_ids'];assert len(ids)==12
    tz=ZoneInfo('Asia/Shanghai')
    p.update(id='T418_FIXED_EVENING_ACTUAL_DISTRIBUTION_REPAIR_CHAIN',root_out=str(OUT),out=str(OUT),
        student_start_model=proposal['path'],student_start_sha256=proposal['sha256'],query_sources=[],
        phases=[dict(name='repair',phase='repair',max_rounds=2),dict(name='protect',phase='protect',max_rounds=2),dict(name='rl',phase='rl',max_rounds=4)],
        final_bytes=20*2**20,root_metadata_bytes=2*2**20,model_bytes=512*1024,full_bytes=384*1024,
        max_physics=16_000_000,max_commands=600,main_seconds=1800,learning_seconds=1200,audit_seconds=600,
        manifest_seconds=120,backup_seconds=300,all_seconds=2520,minimum_free_bytes=40*2**30,
        learning_stop_absolute=datetime(2026,11,7,20,0,tzinfo=tz).timestamp(),
        native_stop_absolute=datetime(2026,11,7,20,20,tzinfo=tz).timestamp(),
        all_stop_absolute=datetime(2026,11,7,21,0,tzinfo=tz).timestamp(),
        bc_steps=200,epochs=8,minimum_cycle_seconds=300,teacher_first=20,prior_day_actual=prior,
        day_limits=dict(physics=50_000_000,commands=2600,main_wall_s=20000),
        root_limits=dict(day_physics=50_000_000,day_commands=2600,day_main_wall_s=20000,global_bytes=2**30,minimum_free_bytes=40*2**30),
        initial_expected_policy_hash=proposal['policy_hash'],initial_expected_success_ids=ids,
        repair_job_ids=['dev_task6_13','wide_dev_3'],
        preflight_discarded_Adam=8,preflight_zero_physics_commands=8,preflight_native=0,
        scientific_mechanisms=dict(repair='New successful actual-student-state teacher corrections plus zero-initial-loss self-history KL; extra only.',
            protect='Additional encoder/core adaptation and old-anchor-gradient-priority halfspace; new joint engineering mechanism, not isolated module attribution.',
            rl='Fresh pure-student original-start PPO with valid prior corrections as auxiliary support; no new teacher rollouts.'),
        bc_admission='All initial success IDs retained and at least one previously failing fixed job repaired; no six-metre stability claim.',
        final_full_admission='Same candidate joint14 all GOAL before wide54+old275; both all GOAL before16fresh then78 named-model exact native readback.',
        query_deduplication='policy_hash_job_teacher_kind_array_digest; unique labels once; all physical repeats charged',
        no_restart_or_parameter_grid=True,safe_T255_frozen=True,narrow_paused=True,
        deployment='Single causal120 recurrent Actor, no map/planner/waypoint/runtime teacher/router.')
    p.pop('historical_control',None)
    initial_rows=[json.loads(line) for line in (ORIGIN/'cases.jsonl').read_text().splitlines() if json.loads(line)['role'].startswith('BC_0_proposal_')]
    assert len(initial_rows)==14
    p['initial_control_rows']=initial_rows
    # Every external byte inherited from the audited predecessor remains locked.
    files=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}
    files.update([proposal['path'],str(ORIGIN/'summary.json'),str(ORIGIN/'audit.json'),str(ORIGIN/'local_manifest.json'),str(ORIGIN/'cases.jsonl'),str(receipt)])
    files.update(row['path'] for row in initial_rows)
    for job in [*p['old8_jobs'],*p['left5_jobs'],p['right_job'],*p['main_jobs'],*p['fresh_jobs'],*p['reload_jobs'],*p['centres'],*base['final_old_jobs']]:
        if job.get('exact_snapshot'):files.add(job['exact_snapshot'])
    p['inputs']={str(f):sha(f) for f in sorted(files)}
    plans={item['name']:budget(p,base,item['phase'],item['max_rounds']) for item in p['phases']}
    for item in plans.values():
        assert item['worst_physics']<=p['max_physics'],item
        assert item['worst_main_commands']+8<=p['max_commands'],item
        assert item['worst_own_bytes']<=p['final_bytes'],item
    p['phase_budgets']=plans
    p['worst_phase_physics']={name:value['worst_physics'] for name,value in plans.items()}
    p['worst_phase_commands']={name:value['worst_main_commands'] for name,value in plans.items()}
    p['max_chain_reserved_bytes']=p['root_metadata_bytes']+len(p['phases'])*p['final_bytes']
    assert used(ROOT/'stage/runs')+p['max_chain_reserved_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    assert time.time()+p['main_seconds']+p['audit_seconds']+p['manifest_seconds']<p['all_stop_absolute']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    DOC.mkdir(parents=True,exist_ok=True);OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T418_BUDGET.json',dict(state='FIXED_MECHANISMS_CONDITIONAL_ACTUAL_REMAINDER_ADMISSION',prior_day_actual=prior,
        phase_budgets=plans,per_phase_caps=dict(physics=16_000_000,commands=600,own_bytes=20*2**20,main_s=1800,audit_s=600),
        root_metadata_bytes=p['root_metadata_bytes'],global_reserved=used(ROOT/'stage/runs')+p['max_chain_reserved_bytes'],
        all_phases_not_guaranteed=True,preflight_discarded_Adam=8,preflight_zero_physics_commands=8,
        deadlines={k:p[k] for k in ['learning_stop_absolute','native_stop_absolute','all_stop_absolute']}))
    print('T418_REGISTERED_FIXED_CHAIN_NOT_LAUNCHED',flush=True)

if __name__=='__main__':prepare()

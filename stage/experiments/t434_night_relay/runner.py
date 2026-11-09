"""Register, authenticate, independently preflight and launch one offline relay."""
from t434_core import *
import subprocess,ast,dis,builtins,types,importlib.util
from datetime import datetime
from zoneinfo import ZoneInfo

def prepare():
    assert not OUT.exists();DOC.mkdir(exist_ok=True)
    prior=ROOT/'stage/runs/t432_offline_night'
    assert read(prior/'SEALED.json')['all_scientific_processes_stopped']
    assert read(ROOT/'stage/docs/overnight_20261108/FINAL_REPORT.json')['all_scientific_processes_stopped']
    for pid in [154281,154283,154284]:assert not Path('/proc',str(pid)).exists()
    for arm in ['data','b','c']:assert read(prior/arm/'audit.json')['state']=='PASS'
    p=copy.deepcopy(read(prior/'protocol.json'))
    p['predecessor_launch_sha256']=sha(prior/'launch.json');p.pop('predecessor_processes',None)
    # Fixed task order: all seven previously failed source tasks. No seed/grid
    # is selected after observing the new relay's results.
    focus=list(p['recovery_target_ids']);assert len(focus)==7
    source_rows=[json.loads(line) for line in Path(p['current_cases_path']).read_text().splitlines()];byid={r['job_id']:r for r in source_rows}
    panel=set(focus)|{j['id'] for j in p['old8_jobs']}|{j['id'] for j in p['additional_old_jobs']}
    six_good=[r['job_id'] for r in source_rows if r['job_id'].startswith('wide117_80_') and r['success']]
    for prefix in ['wide117_80_dev6_LEFT_','wide117_80_dev6_RIGHT_']:
        panel.update([i for i in six_good if i.startswith(prefix)][:4])
    panel.update(['wide117_80_dev6_LEFT_15','wide117_80_dev6_LEFT_18','wide117_80_dev6_LEFT_21','dev_task6_13','dev_task6_15'])
    assert len(panel)<=32 and set(focus)<=panel and panel<=set(byid)
    stages=dict(fresh_rl=dict(main_seconds=5400,audit_seconds=1200,max_physics=14000000,max_commands=600,max_bytes=18*2**20),
      iterative_recovery=dict(main_seconds=5400,audit_seconds=1200,max_physics=9000000,max_commands=450,max_bytes=12*2**20),
      representation_rl=dict(main_seconds=4200,audit_seconds=1200,max_physics=11000000,max_commands=500,max_bytes=18*2**20),
      gate=dict(main_seconds=1800,audit_seconds=300,max_physics=5900000,max_commands=165,max_bytes=32*2**20),
      eight=dict(main_seconds=900,audit_seconds=180,max_physics=2240000,max_commands=65,max_bytes=2**20))
    p.update(id='T434_OFFLINE_RELAY_FRESH_RL_ITERATIVE_RECOVERY_REPRESENTATION',out=str(OUT),
      relay_dataset_path=str(prior/'data/summary.json'),anchor_source_model_sha256=p['student_start_sha256'],
      stage_budgets=stages,root_limits=dict(max_physics=40000000,max_commands=2000,max_bytes=83*2**20),
      predecessor_root=str(prior),predecessor_docs=str(ROOT/'stage/docs/overnight_20261108'),
      maximum_main_wall_seconds=18000,reserve_gate_for_learning=True,reserve_eight_for_learning=False,
      validation_reserve_seconds=1200,backup_seconds=600,source_seconds=180,
      rounds=dict(fresh_rl=6,iterative_recovery=2,representation_rl=4),
      maximum_audit_Adam=dict(fresh_rl=96,iterative_recovery=192,representation_rl=64),
      focus_order=focus,panel_ids=sorted(panel),seed_base=20261108434,
      deviation_position_m=.03,deviation_yaw_rad=.05,intervene_earlier_holds=5,
      preflight_discarded_Adam=24,preflight_zero_physics_commands=8,
      new_authorization='User Nov8 explicitly requests sequential offline relay after B/C instead of ending early. Entire T432 plus relay remains40Mphys/2000cmd/18000main, Nov9 08:30allstop; no duplicate old experiment.',
      pilot='Fresh own-policyPPO extra4450 ->actual new-state verified recovery+BC/PPO ->conditional encoder/core/extra10722PPO. Different mechanisms, no claim isolated causality/novelty/safety.',
      parent_success_retention='Each final kept stage must retain all accepted parent full329 successIDs and gain>=1; intermediate panel/progress only research, not full6m.',
      local_capacity='T435 reversible archived JSON encoding only after source sync; complete per-stage worst + gate32MiB capacity checked from actual usage, not promised all worst stages fit.',
      no_T432_active_change=True,initial_RL_steps=0)
    # Only unchanged stopped data are inputs. The predecessor source publication
    # and mutable docs are not model/data dependencies.
    paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{
      str(prior/'protocol.json'),str(prior/'SEALED.json'),str(prior/'launch.json'),
      str(prior/'data/summary.json'),str(prior/'data/audit.json'),str(prior/'data/anchors_c.json'),
      str(DOC/'PLAN.md'),str(DOC/'METHOD_SOURCES.md'),
      str(ROOT/'stage/experiments/t435_relay_reserve/codec.py'),str(DOC/'RESERVE_CANDIDATES.json')}
    for ent in read(prior/'data/summary.json')['result']['anchors_c']:paths.add(ent['path'])
    for file in prior.rglob('*.jsonl'):
        if file.name in ['cases.jsonl','replays.jsonl']:
            fresh={j['id'] for j in p['fresh_jobs']}
            assert not any(json.loads(line)['job_id'] in fresh for line in file.read_text().splitlines()),'Fresh already consumed by predecessor'
            p['fresh_ledger'][str(file)]=sha(file);paths.add(str(file))
    p['inputs']={f:sha(f) for f in sorted(paths)}
    candidates=read(DOC/'RESERVE_CANDIDATES.json');candidate_paths={str(ROOT/r['original_path']) for r in candidates['rows']}
    assert not candidate_paths&set(p['inputs'])
    assert time.time()+5400+1200+900<p['deadline_all']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',night_shared=dict(physics=40000000,commands=2000,main_seconds=18000),stage_budgets=stages,all_worst_stages_not_guaranteed_to_fit=True,reserve_gate32MiB_5900kphysics165cmd=True,global_limit=2**30,min_free=40*2**30,metadata_max=2*2**20,preflight_Adam=24,preflight_commands=8,source_start_successes=322,panel_count=len(panel),cuts={k:p['deadline_'+k] for k in ['learning','native','all']}))
    print('T434_REGISTERED',len(panel),flush=True)

def static(p):
    for f in HERE.glob('*.py'):
        ast.parse(f.read_text());spec=importlib.util.spec_from_file_location('stage434_check_'+f.stem,f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        def check(code):
            bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(m) and not hasattr(builtins,i.argval)}
            for c in code.co_consts:
                if isinstance(c,types.CodeType):bad|=check(c)
            return bad
        for fn in vars(m).values():
            if isinstance(fn,types.FunctionType) and fn.__module__==m.__name__:assert not check(fn.__code__),(f.name,check(fn.__code__))
    return True

def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();static(p)
    for stage in ['fresh_rl','iterative_recovery','representation_rl']:
        for label in ['A','B']:
            subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'contract.py'),'probe',str(OUT/'protocol.json'),stage,label,str(OUT/f'clone_{stage}_{label}.json')],cwd=ROOT,check=True,timeout=240)
        subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'contract.py'),'compare',str(OUT/'protocol.json'),stage,str(OUT/f'clone_{stage}_A.json'),str(OUT/f'clone_{stage}_B.json'),str(OUT/f'compare_{stage}.json')],cwd=ROOT,check=True,timeout=60)
    ban_learning();counts=mp.get_context('spawn').RawArray('q',5);folder=OUT/'preflight_native';folder.mkdir();pool=Pool(dict(p,out=str(folder),relay_stage='fresh_rl',native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        ids=[c.pid for c in pool.children];job=p['main_jobs'][0];requests=[dict(model=p['student_start_model'],job=job,mode='student',seed=None,cancel_probe=True)]*4
        assert all(v['state']=='CANCEL_PASS' for v in pool.request(requests)) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=24,zero_physics_commands=8,three_profiles_independent_clones=True,synthetic_streams_not_native_navigation=True,full_actual_history_step_sequence_and_density_strict=True,all_four_spawn_workers_cancelled_and_stopped=True,actual_Actor_Adam_reject_interrupt_restore=True))
    E.N.manifest(OUT);print('T434_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS';assert read(ROOT/'stage/runs/t435_relay_reserve/audit.json')['state']=='PASS';assert not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    before=E.N.K.S.used(ROOT/'stage/runs');assert before+p['stage_budgets']['fresh_rl']['max_bytes']+p['stage_budgets']['gate']['max_bytes']<2**30
    now=time.time();limit=p['deadline_all']-now-10;assert limit>5400+1200+900
    with (DOC/'CONSOLE.log').open('xb') as log:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T434','--why=Authorized offline Stage relay with bounded overnight total','timeout','--signal=TERM','--kill-after=2',str(int(limit)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+limit,source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),research_status_sha256=sha(ROOT/'stage/docs/RESEARCH_STATUS.md'),offline_science=True,all_stop_nov9_0830=True,predecessor_verified_complete=True,night_budget_shared_not_reset=True))
    print('T434_ACTUALLY_LAUNCHED',child.pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

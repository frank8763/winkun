"""Fix inherited output routing, complete only the fixed missing evaluations."""
from pathlib import Path
import copy,io,os,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t540_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t539_terminal_success_pg';a=read(prior/'audit.json');s=read(prior/'summary.json');assert a['state']=='PASS' and a['scope']=='ONE_REAL_SGD_AND_TEN_COMPLETED_CASES_PARTIAL_ONLY'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids'])
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T539.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    old=read(prior/'protocol.json');c=R.Previous.configuration(old);model=s['result']['model'];assert sha(model['path'])==model['sha256']
    end=old['absolute_all_stop'];assert time.time()+1000<end and time.time()+480<end-600
    jobs=c['evaluation_jobs'][10:];cap=sum(j['case']['step_cap'] for j in jobs);assert cap+s['physics']<=old['max_physics'] and 26+s['commands']<=old['max_commands']
    p=dict(id='T540_FROZEN_MISSING22_FIRSTPASS_COMPLETION',source_registration=str(prior/'protocol.json'),previous_cases=str(prior/'misdirected/cases.jsonl'),candidate=model,source_protocol=c['source_protocol'],max_physics=cap,max_commands=26,max_bytes=368*1024,compact_bytes=10*1024,main_seconds=480,learning_seconds=0,audit_seconds=120,transport_seconds=180,source_seconds=120,absolute_learning_stop=end-1800,absolute_native_stop=end-600,absolute_all_stop=end,stage_cpus=[14,15],baseline_field_success_ids=c['initial_field_success_ids'],scope='Same immutable T539 candidate,only22missing original-start first-pass cases.0newlearning,0old10rerun,0exact32. No promotion even if first pass improves. Previous235017physics and16commands remain; combined<=original T5393.201204M/68cmd. Fixed output namespace explicitly overrides inherited config; no use of ended T531 for outputs.',authorization='User asked next experiment; necessary bounded output routing repair and completion of already authorized evaluation,within same20:00all deadline and original physical/command envelope. No new training/parameter choice/seed/budget expansion.')
    p['inputs']={str(f):sha(f) for f in [*HERE.glob('*.py'),prior/'protocol.json',prior/'audit.json',prior/'summary.json',prior/'misdirected/cases.jsonl',DOC/'T539_OUTPUT_RELOCATION.json',DOC/'T540_PLAN.md',Path(model['path'])]}
    p['source_scientific_inputs']=old['inputs']
    assert R.used(ROOT/'stage/runs')+p['max_bytes']+65536<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
    cfg=R.configuration(p);assert cfg['out']==str(OUT) and len(cfg['remaining_jobs'])==22
    save(DOC/'T540_BUDGET.json',dict(new_physics_max=cap,previous_T539_physics=235017,combined_max=cap+235017,original_envelope=old['max_physics'],new_commands=26,combined_commands_max=42,own_bytes=p['max_bytes'],global_reserved=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes'],new_Actor_updates=0,main_seconds=480,audit_seconds=120,backup_seconds=180,source_seconds=120,native_stop=end-600,all_stop=end))
    print('T540_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    root=read(OUT/'protocol.json');p=R.configuration(root);R.verify(p,True);assert not (OUT/'preflight.json').exists()
    # Exact regression for the inherited T531 output-path bug.
    assert p['out']==str(OUT) and p['out']!=read(p['actor_configuration'])['out']
    for f,h in root['source_scientific_inputs'].items():assert sha(f)==h,f
    for j in p['remaining_jobs']:
        n=(j['case']['step_cap']+99)//100;a=dict(actions=R.np.zeros((n,2),R.np.float32),native_counts=R.np.full(n,100,R.np.uint8),end_flags=R.np.zeros((n,3),R.np.uint8));a['end_flags'][-1,1]=1
        blob=R.pack(a);assert len(blob)<=p['compact_bytes']
    assert R.Previous.L.A.frozen(p,p['candidate']['path']).phash()==p['candidate']['policy_hash']
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,output_route_forced_to_current_run=True,all22_worst_compact_shapes_fit=True,frozen_candidate_hash_pass=True));print('T540_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

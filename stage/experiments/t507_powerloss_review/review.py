"""Read-only audit/archive of interrupted T507; no navigation or new learning.

Original incomplete status, zero-byte interrupted NPZ and all raw files remain.
Optimizer reconstruction is an explicitly discarded historical audit clone.
"""
from pathlib import Path
import fcntl,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OLD=ROOT/'stage/runs/t507_joint_success_transfer';OUT=ROOT/'stage/runs/t507_powerloss_review'
DOC=ROOT/'stage/docs/review_20261112'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);tmp=Path(str(p)+'.partial');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def files():return sorted(f for f in OLD.rglob('*') if f.is_file())
def original_manifest():return [dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in files()]
def guard(p):
    assert time.time()<p['absolute_all_stop'] and shutil.disk_usage(ROOT).free>=40*2**30
    assert sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())<=2**30
    assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<=p['own_cap']
    assert original_manifest()==read(OUT/'original_manifest.json')['files']
    for pid in p['old_pids']:assert not Path('/proc',str(pid)).exists()
def prepare():
    assert not OUT.exists();p=read(OLD/'protocol.json');g=read(OLD/'pilot/worker_generations.json');ids=[read(OLD/'launch.json')['pid']]+[pid for w in g for pid in [w['parent'],*w['workers']]]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    assert not (OLD/'pilot/summary.json').exists() and not (OLD/'pilot/audit.json').exists()
    assert time.time()+900<p['absolute_all_stop'];OUT.mkdir()
    manifest=original_manifest();save(OUT/'original_manifest.json',dict(files=manifest,scope='All original T507 files including empty interrupted NPZ and logs; frozen read-only after reboot'))
    plan=dict(id='T507_POWERLOSS_FORENSIC_REVIEW',old_pids=ids,original_protocol_sha256=sha(OLD/'protocol.json'),source_sha256=sha(__file__),own_cap=4*2**20,absolute_all_stop=min(p['absolute_all_stop'],time.time()+1800),audit_seconds=600,backup_seconds=900,native=0,new_Actor_Adam=0,maximum_discarded_historical_Actor_Adam=32,Value_Adam=0,old_directory_never_restarted=True,CPU=[14,15],CPUQuota='100%',GPU=False)
    save(OUT/'protocol.json',plan);guard(plan);print('POWERLOSS_REVIEW_REGISTERED_ZERO_NATIVE',flush=True)
def audit():
    c=read(OUT/'protocol.json');guard(c);assert not (OUT/'audit.json').exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('forensic audit600s')));signal.alarm(c['audit_seconds'])
    sys.path.insert(0,str(ROOT/'stage/experiments/t507_joint_success_transfer'));import t507_runtime as R
    L,G,P,E,M=R.L,R.G,R.P,R.E,R.M;p=read(OLD/'protocol.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    P.frozen=L.A.frozen;rows=G.rows(OLD/'pilot');rep=G.rows(OLD/'pilot','replays.jsonl');blocks=G.rows(OLD/'pilot','blocks.jsonl')
    assert len(rows)==len(rep)==18 and len(blocks)==1 and blocks[0]['accepted']
    arrays=[]
    for r in rows:
        assert r['complete'] and r['success'] and not r['collision'] and not r['administrative_censor'] and sha(r['path'])==r['sha256']
        a=E.N.S.arrays(r['path']);assert all(G.digest(v)==r['array_hashes'][k] for k,v in a.items()) and G.digest(a)==r['array_digest']
        assert int(a['native_counts'].sum())==r['physics'];M.audit_full_history(p,r,a);arrays.append(a)
    by={r['path']:r for r in rows}
    for r in rep:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    obj=L.make(p);before=G.digest(P.state(*obj));assert before==blocks[0]['before_joint_hash']
    assert not (OUT/'audit_optimizer_calls.jsonl').exists()
    def optimizer_event(stage,k):
        with (OUT/'audit_optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([stage,k])+'\n')
    update=L.fit(obj,L.groups(p),p,lambda:guard(c),optimizer_event)
    block=blocks[0];assert update==block['update'] and G.digest(P.state(*obj))==block['model']['joint_hash'] and sha(block['model']['path'])==block['model']['sha256']
    ids=R.native_successes(p,rows,arrays);assert len(ids)==18 and ids=={j['id'] for j in p['guard_jobs']} and sorted(ids)==block['native_success_ids']
    expected=[[w+'_block0',stage,k] for w in ['live','shadow'] for k in range(update['Actor_Adam']) for stage in ['BEFORE','AFTER']]
    assert G.rows(OLD/'pilot','optimizer_calls.jsonl')==expected
    complete=sum(r['physics'] for r in rows+rep);last=read(OLD/'pilot/status.json');assert last['state']=='FINAL_REMAINING_FIELD14' and last['commands']==40
    upper=complete+sum(j['case']['step_cap'] for j in p['remaining_field_jobs'][:2]);assert complete<=last['physics']<=upper
    missing=set(p['new_task_ids']);assert missing<=ids
    model=block['model'];report=dict(state='INTERRUPTED_BY_POWER_LOSS_COMPLETED_BLOCK_AUDIT_PASS',original_execution_complete=False,original_final_gate_complete=False,
        original_directory_unchanged=True,all_original_processes_gone=True,new_native=0,new_Actor_Adam=0,discarded_historical_Actor_Adam=update['Actor_Adam'],Value_Adam=0,
        completed_blocks=1,actual_live_supervised_Actor_Adam=32,actual_shadow_supervised_Actor_Adam=32,old8=G.R.M.tally(rows[:8]),completed_field10=G.R.M.tally(rows[8:]),new_task_ids=sorted(missing),new_tasks_success=3,
        completed_guard18=G.R.M.tally(rows),completed_exact_replays=18,full_field24=None,remaining14_valid_results=0,unresolved_outstanding_native_requests=[j['id'] for j in p['remaining_field_jobs'][:2]],remaining_not_dispatched12=[j['id'] for j in p['remaining_field_jobs'][2:]],
        full_field24_final_exact_NOT_executed=True,unrecorded_results_not_assigned_success_or_failure=True,recorded_complete_physics=complete,last_counter_physics_lower_bound=last['physics'],total_physics_lower_bound=last['physics'],total_physics_upper_bound=upper,total_physics_exact=None,
        recorded_dispatched_main_commands=40,normal_shutdown_commands_not_executed=2,preflight_zero_physics_commands=6,final_model=model,
        finite18panel_pass_NOT_final_deployment_promotion=True,previous_T505_accepted_best_untouched=True,empty_interrupted_files=[x for x in read(OUT/'original_manifest.json')['files'] if x['bytes']==0],remote='PENDING',epoch=time.time())
    guard(c);save(OUT/'audit.json',report);save(DOC/'T507_POWERLOSS_REPORT.json',report);signal.alarm(0);print('POWERLOSS_EXISTING18_AUDIT_PASS_NO_NEW_NATIVE',flush=True)
def backup():
    c=read(OUT/'protocol.json');guard(c);a=read(OUT/'audit.json');assert a['state']=='INTERRUPTED_BY_POWER_LOSS_COMPLETED_BLOCK_AUDIT_PASS'
    spec=importlib.util.spec_from_file_location('_t507_powerloss_archive',ROOT/'stage/experiments/t493_wide8_night/transport.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T);T.OLD.install();api=T.OLD.T
    T.stopped=lambda:guard(c);deadline=min(c['absolute_all_stop'],time.time()+c['backup_seconds']);release=api.B.api('GET','releases/tags/stage-eight-only-2026-11-11');receipts=[]
    originals=files()
    for i in range(0,len(originals),80):receipts.append(T.archive('T507_powerloss_original_'+str(i//80),originals[i:i+80],release,deadline,api))
    meta=sorted(f for f in OUT.rglob('*') if f.is_file());receipts.append(T.archive('T507_powerloss_review',meta,release,deadline,api));guard(c)
    receipt=dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T507_powerloss',original_run_complete=False,original_files_preserved_including_zero_byte=True,url=release['html_url'],members=sum(x['members'] for x in receipts),child_receipts=[x['group'] for x in receipts],bytes=sum(x['bytes'] for x in receipts),temporary_disk_archive_bytes=0,verified_epoch=time.time())
    save(ROOT/'stage/docs/long_distance_20261101/release_receipt_T507_powerloss.json',receipt)
    report=dict(a,remote=receipt['state'],archive_receipt='stage/docs/long_distance_20261101/release_receipt_T507_powerloss.json');save(DOC/'T507_POWERLOSS_REPORT.json',report)
    print('POWERLOSS_ALL_ORIGINAL_FILES_REMOTE_SHA_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

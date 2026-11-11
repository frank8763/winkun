"""One eight-only finite pilot and byte-reversible local capacity admission."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,gzip,hashlib,io,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t488_runtime as T
L,P,E,M,R,torch,np=T.L,T.P,T.E,T.M,T.R,T.torch,T.np
read,save,sha,digest=T.read,T.save,T.sha,T.digest
OUT=T.OUT;DOC=ROOT/'stage/docs/review_20261111'

def idle():
    for f in Path('/proc').glob('[0-9]*/cmdline'):
        if int(f.parent.name)==os.getpid():continue
        try:args=f.read_bytes().split(b'\0')
        except OSError:continue
        assert not any(os.fsdecode(a).startswith(str(ROOT/'stage/experiments')+'/') for a in args[1:]),('Other Stage process',f.parent.name)

def prepare():
    idle();assert not OUT.exists()
    old=read(ROOT/'stage/runs/t487_eight_only/protocol.json');report=read(DOC/'T487_FINAL_REPORT.json')
    base=report['selected']
    assert base['wide8']['success']==35 and base['wide8']['contact']==1 and base['accepted']
    assert sha(base['model']['path'])==base['model']['sha256']
    p=copy.deepcopy(old);stop=datetime(2026,11,11,20,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp();assert time.time()+3300<stop
    by={j['id']:j for j in p['wide8_jobs']}
    ids=['wide117_8m_dev6_LEFT_01','wide117_8m_dev6_LEFT_19','wide117_8m_dev6_LEFT_08','wide117_8m_dev6_LEFT_26']
    diag_ids=['wide117_8m_dev6_LEFT_01','wide117_8m_dev6_LEFT_08','wide117_8m_dev6_LEFT_26','wide117_8m_dev6_RIGHT_00']
    originals={x['job_id']:x for x in map(json.loads,(ROOT/'stage/runs/t487_eight_only/pair/cases.jsonl').read_text().splitlines())}
    assert all(x not in base['success_ids'] for x in ids)
    baseline=dict(model=base['model'],wide8=base['wide8'],success_ids=[x for x in base['success_ids'] if x.startswith('wide117_8m_')],wide6=None,old275=None,not_jointly_promoted=True)
    p.update(id='T488_LEFT8_FAILURE_TRAJECTORY_AND_PPO',out=str(OUT),initial=base['model'],initial_result=baseline,
        initial_success_ids=baseline['success_ids'],training_jobs=[by[x] for x in ids],diagnostic_jobs=[by[x] for x in diag_ids],diagnostic_original_rows={x:originals[x] for x in diag_ids},right_anchor_id=diag_ids[-1],capture_summary_path=str(OUT/'capture/summary.json'),seed=20261111488,epochs=16,
        target_kl=.005,keep_weight=1.,max_bytes=12*2**20,full_bytes=800*1024,compact_bytes=8192,model_bytes=1792*1024,
        max_physics=5000000,root_max_physics=5000000,max_commands=100,root_max_commands=100,
        main_seconds=2400,root_max_main_seconds=2400,learning_seconds=600,audit_seconds=300,
        all_seconds=3300,absolute_all_stop=stop,absolute_native_stop=stop-600,absolute_learning_stop=stop-1800,
        preflight_commands=6,user_authorization='Nov11 user explicitly requests next round only8metres. Old/six training and evaluation NOT scheduled; singleActor only. Existing20:30allstop.',
        acceptance='One complete54eight development. Accept only success>35 and contacts<=1, all valid/no censor. Lost previous8-ID successes explicitly reported; no old/six stability claim.',
        mechanism='Initialize retained eight-onlyT48735/54. Reconstruct four exact frozenpose histories incl LEFT timeout/turncollision and RIGHT success, then four new LEFT failed-state episodes with pooled near/far advantages; preserve one actual RIGHT successful fullhistory output. Same PPOreward/sigma/lambda1/KL.005, no old/six. Joint sampling+anchor pilot not original/causalproof.')
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(ROOT/'stage/runs/t487_eight_only/pair/summary.json'),str(ROOT/'stage/runs/t487_eight_only/pair/audit.json'),str(ROOT/'stage/runs/t487_eight_only/pair/cases.jsonl'),str(DOC/'T488_PLAN.md'),p['initial']['path']}
    p['inputs']={f:sha(f) for f in sorted(files)};p['protected_files']=dict(old['protected_files']);p['protected_files'][p['initial']['path']]=p['initial']['sha256']
    # Only byte-reversible text encoding. All raw model/arrays/errors/protocols unchanged.
    excluded=set(files)|set(p['protected_files']);selected=[]
    needed=max(0,M.used()+p['max_bytes']+300000-2**30)
    # Completed historical metadata remain byte-exact locally; explicit restore
    # access replaces bare filenames. Never encode a pinned input/canonical.
    sys.path.insert(0,str(ROOT/'stage/experiments/t486_adam_memory_projection'))
    import t486_cache as K
    excluded=K._protected(excluded)
    for folder in sorted((ROOT/'stage/runs').glob('t4*')):
        if folder.name in ['t487_eight_only','t488_left_failure_focus']:continue
        for f in folder.rglob('*'):
            if not f.is_file() or f.suffix not in ['.json','.jsonl','.log'] or str(f) in excluded:continue
            if any(w in f.name.lower() for w in ['error','fail','ledger','receipt','manifest','summary','audit','preflight']):continue
            b=f.read_bytes();z=gzip.compress(b,mtime=0);dest=Path(str(f)+'.t488.gz')
            if len(b)-len(z)<2000 or dest.exists():continue
            selected.append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),gzip_path=str(dest),gzip_bytes=len(z),gzip_sha256=hashlib.sha256(z).hexdigest()))
    selected.sort(key=lambda x:x['bytes']-x['gzip_bytes'],reverse=True)
    rows=[];released=0
    for row in selected:
        if released>=needed+100000:break
        rows.append(row);released+=row['bytes']-row['gzip_bytes']
    assert released>=needed+100000,('Insufficient byte-reversible local logs',released,needed)
    p['local_encoding']=rows
    assert 8*p['full_bytes']+54*p['compact_bytes']+p['model_bytes']+int(3.5*2**20)<=p['max_bytes']
    worst=2*sum(j['case']['step_cap'] for j in p['training_jobs']+p['diagnostic_jobs'])+sum(j['case']['step_cap'] for j in p['wide8_jobs']);assert worst<=p['root_max_physics']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T488_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',worst_physics=worst,max_physics=5000000,worst_commands=84,max_commands=100,
        own_bytes=p['max_bytes'],logical_global_limit=2**30,all_stop=stop,main_seconds=2400,audit_seconds=300,backup_source_seconds=360,
        local_log_expected_release=released,local_logs=len(rows),live_Adam_max32=True,old275=None,wide6=None,fresh16=None,reload78=None))

def encode():
    idle();p=read(OUT/'protocol.json');assert not (OUT/'encoding_receipt.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    before=M.used();done=[]
    for row in p['local_encoding']:
        f=Path(row['path']);b=f.read_bytes();z=gzip.compress(b,mtime=0);dest=Path(row['gzip_path'])
        assert sha(f)==row['sha256'] and len(b)==row['bytes'] and hashlib.sha256(z).hexdigest()==row['gzip_sha256']
        assert not dest.exists() and M.used()+len(z)+10000<=2**30
        with dest.open('xb') as x:x.write(z);x.flush();os.fsync(x.fileno())
        assert gzip.decompress(dest.read_bytes())==b;dest.chmod(0o444)
        # Local immutable gzip is byte-exact before raw pathname relocation.
        done.append(row);save(OUT/'encoding_progress.json',dict(files=done,all_original_bytes_stay_local=True,T486_remote_still_pending=True))
        f.unlink()
    assert M.used()-M.used(OUT)+p['max_bytes']<=2**30
    save(OUT/'encoding_receipt.json',dict(state='LOCAL_LOSSLESS_COMPLETE',pid=os.getpid(),before=before,after=M.used(),files=len(done),original_bytes_local=True,T486_remote_pending_NOT_claimed_backed=True))

def encoding_audit():
    idle();p=read(OUT/'protocol.json');d=read(OUT/'encoding_receipt.json');assert not Path('/proc',str(d['pid'])).exists()
    for row in p['local_encoding']:
        z=Path(row['gzip_path']).read_bytes();b=gzip.decompress(z)
        assert hashlib.sha256(z).hexdigest()==row['gzip_sha256'] and hashlib.sha256(b).hexdigest()==row['sha256'] and len(b)==row['bytes']
        assert not Path(row['path']).exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert M.used()-M.used(OUT)+p['max_bytes']<=2**30
    save(OUT/'encoding_audit.json',dict(state='PASS',native=0,optimizer=0,all_original_log_bytes_local=True,no_model_or_array_deletion=True,remote_T486_pending=True))

def preflight():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'encoding_audit.json')['state']=='PASS'
    # New Actor behavior is checked on actual histories, not treated as oldpolicy data.
    old=read(ROOT/'stage/runs/t481_fresh16_rehearsal/capture/summary.json')['result']['samples']
    obj=L.make(p);shadow=L.make(p);arrays=[]
    with torch.no_grad():
        for r in old:
            a=E.N.S.arrays(r['path']);x=torch.tensor(a['states'],device=next(obj[0].parameters()).device);loc=obj[0].sequence(x)
            # Fixed synthetic-on-history fixture; NOT collected behavior or training data.
            a=dict(a);a['locations']=loc.cpu().numpy();a['logp']=P.L.density(loc,loc.new_tensor(a['raw_actions']),loc.new_tensor(a['atom']),p['raw_sigma']).cpu().numpy();arrays.append(a)
    arrays=arrays[:2]+arrays[-2:]
    fixture=dict(p,fixture_anchor_rows=[next(row for row in old if row['success'] and '_RIGHT_' in row['job_id'])])
    aa=L.fit(obj,arrays,fixture,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,arrays,fixture,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
    b=io.BytesIO();torch.save(P.state(*obj),b);assert digest(torch.load(io.BytesIO(b.getvalue()),weights_only=False))==digest(P.state(*obj))
    for z in [obj,shadow]:P.load(z,P.checkpoint(p['initial']['path'])['joint_state']);assert digest(P.state(*z))==p['initial']['joint_hash']
    folder=OUT/'preflight_native';folder.mkdir();counts=R.M.mp.get_context('spawn').RawArray('q',3)
    pool=T.T.Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        replies=pool.request([dict(model=p['initial']['path'],job=j,seed=None,cancel_probe=True) for j in p['training_jobs'][:2]])
        assert all(x['state']=='CANCEL_PASS' for x in replies) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=6,discarded_Adam=4*aa['epochs'],synthetic_fixture_NOT_onpolicy_or_new_navigation=True,shadow_and_whole_joint_rollback_exact=True))

def launch():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+3300<p['absolute_all_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T488','--why=Bounded eight-only policy pilot','timeout','--signal=TERM','--kill-after=2','3300','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=now+3300,source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),eight_only=True,all_stop_authorized=p['absolute_all_stop']))
    print('T488_ACTUALLY_LAUNCHED',c.pid,flush=True)

def text(original):
    f=Path(original)
    if f.exists():return f.read_text()
    row=next(r for r in read(OUT/'protocol.json')['local_encoding'] if Path(r['path'])==f)
    z=Path(row['gzip_path']).read_bytes();assert hashlib.sha256(z).hexdigest()==row['gzip_sha256']
    b=gzip.decompress(z);assert hashlib.sha256(b).hexdigest()==row['sha256'];return b.decode()
if __name__=='__main__':globals()[sys.argv[1]]()

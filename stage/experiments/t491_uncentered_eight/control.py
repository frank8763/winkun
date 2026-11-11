"""Bounded two-side eight metre PPO, actual-success anchors and exact accounting."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,gzip,hashlib,io,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t491_runtime as T
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
    previous=ROOT/'stage/runs/t490_balanced_eight'
    old=read(previous/'protocol.json');report=read(DOC/'T490_FINAL_REPORT.json');base=report['selected']
    assert report['all_scientific_processes_stopped'] and report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert base['wide8']['success']==39 and base['wide8']['contact']==1
    assert read(previous/'capture/audit.json')['state']=='PASS' and read(previous/'pair/audit.json')['state']=='PASS'
    assert read(previous/'pair/summary.json')['result']['chosen_arm'] is None
    assert sha(base['model']['path'])==base['model']['sha256']==old['initial']['sha256']
    p=copy.deepcopy(old);stop=datetime(2026,11,11,21,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    assert time.time()+3000<stop
    p.update(id='T491_UNCENTERED_EIGHT_PPO',out=str(OUT),initial=base['model'],initial_result=base,
        initial_success_ids=base['success_ids'],capture_summary_path=str(previous/'capture/summary.json'),capture_audit_path=str(previous/'capture/audit.json'),
        max_bytes=5*2**20,max_physics=3000000,root_max_physics=3000000,max_commands=70,root_max_commands=70,
        main_seconds=1800,root_max_main_seconds=1800,learning_seconds=600,audit_seconds=180,
        all_seconds=3000,absolute_all_stop=stop,absolute_native_stop=stop-600,absolute_learning_stop=stop-1800,
        mechanism='Exactly T490 initial mature Actor/Value/Adam, same four actual collected trajectories and two actual successful anchor histories, same16 PPO epochs/KL/reward/speed/cap. ONLY advantage normalization changed: raw physical finite G-V divided by per-side episode-balanced RMS, no temporal mean subtraction. Signed advantages retained, NOT SIL/BC or new data. One deterministic matched-data diagnostic candidate, not novel algorithm.',
        user_authorization='Nov11 user requests continue improving if not allpass; same new21:30 hardstop. Old T490 sealed and not restarted.',
        previous_physics=0,previous_commands=0,previous_main_seconds=0)
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(previous/'protocol.json'),str(previous/'capture/summary.json'),str(previous/'capture/audit.json'),str(previous/'pair/summary.json'),str(previous/'pair/audit.json'),str(DOC/'T491_PLAN.md'),str(DOC/'T491_FAILURE_ANALYSIS.json'),p['initial']['path']}
    cap=read(previous/'capture/summary.json')['result']
    for row in cap['samples']+cap['diagnostics']:
        assert sha(row['path'])==row['sha256']
        files.add(row['path'])
    assert len(cap['samples'])==4 and all(r['success'] and not r['collision'] and r['loaded_model_sha256']==p['initial']['sha256'] for r in cap['samples'])
    p['inputs']={f:sha(f) for f in sorted(files)}
    sys.path.insert(0,str(ROOT/'stage/experiments/t486_adam_memory_projection'));import t486_cache as K
    excluded=K._protected(set(files)|set(p['protected_files'])|set(p.get('constructor_reads',[])))
    candidates=[];needed=max(0,M.used()+p['max_bytes']+300000-2**30)
    for folder in sorted((ROOT/'stage/runs').glob('t4*')):
        for f in folder.rglob('*'):
            if not f.is_file() or f.suffix not in ['.json','.jsonl','.log'] or str(f) in excluded:continue
            if any(w in f.name.lower() for w in ['error','fail','ledger','receipt','manifest','summary','audit','preflight']):continue
            raw=f.read_bytes();z=gzip.compress(raw,mtime=0);dest=Path(str(f)+'.t491.gz')
            if len(raw)-len(z)>2000 and not dest.exists():candidates.append(dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),gzip_path=str(dest),gzip_bytes=len(z),gzip_sha256=hashlib.sha256(z).hexdigest()))
    selected=[];released=0
    for row in sorted(candidates,key=lambda row:row['bytes']-row['gzip_bytes'],reverse=True):
        if released>=needed+100000:break
        selected.append(row);released+=row['bytes']-row['gzip_bytes']
    assert released>=needed+100000,(released,needed)
    p['local_encoding']=selected
    worst=sum(j['case']['step_cap'] for j in p['wide8_jobs']);assert worst<=3000000
    assert 54*p['compact_bytes']+p['model_bytes']+int(2.8*2**20)<=p['max_bytes']
    OUT.mkdir();encoded=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n'
    (HERE/'protocol.json').write_text(encoded);(OUT/'protocol.json').write_text(encoded)
    save(DOC/'T491_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',worst_physics=worst,max_physics=3000000,worst_commands=64,max_commands=70,own_bytes=p['max_bytes'],global_limit=2**30,all_stop=stop,main_seconds=1800,audit_seconds=180,backup_source_seconds=420,local_text_expected_release=released,local_text_files=len(selected),new_training_episodes=0,reused_exact_T490_episodes=4,old275=None,wide6=None,fresh16=None,reload78=None))

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
        done.append(row);save(OUT/'encoding_progress.json',dict(files=done,all_original_bytes_stay_local=True))
        f.unlink()
    assert M.used()-M.used(OUT)+p['max_bytes']<=2**30
    save(OUT/'encoding_receipt.json',dict(state='LOCAL_LOSSLESS_COMPLETE',pid=os.getpid(),before=before,after=M.used(),files=len(done),original_bytes_local=True,no_model_or_array_deletion=True))

def encoding_audit():
    idle();p=read(OUT/'protocol.json');d=read(OUT/'encoding_receipt.json');assert not Path('/proc',str(d['pid'])).exists()
    for row in p['local_encoding']:
        z=Path(row['gzip_path']).read_bytes();b=gzip.decompress(z)
        assert hashlib.sha256(z).hexdigest()==row['gzip_sha256'] and hashlib.sha256(b).hexdigest()==row['sha256'] and len(b)==row['bytes'] and not Path(row['path']).exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert M.used()-M.used(OUT)+p['max_bytes']<=2**30
    save(OUT/'encoding_audit.json',dict(state='PASS',native=0,optimizer=0,all_original_text_bytes_local=True,no_model_or_array_deletion=True))

def preflight():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'encoding_audit.json')['state']=='PASS'
    old=read(ROOT/'stage/runs/t481_fresh16_rehearsal/capture/summary.json')['result']['samples']
    obj=L.make(p);shadow=L.make(p);arrays=[]
    with torch.no_grad():
        for r in old:
            a=E.N.S.arrays(r['path']);x=torch.tensor(a['states'],device=next(obj[0].parameters()).device);loc=obj[0].sequence(x)
            a=dict(a);a['locations']=loc.cpu().numpy();a['logp']=P.L.density(loc,loc.new_tensor(a['raw_actions']),loc.new_tensor(a['atom']),p['raw_sigma']).cpu().numpy();arrays.append(a)
    arrays=arrays[:2]+arrays[-2:]
    fixture=dict(p,fixture_anchor_rows=[next(r for r in old if r['success'] and side in r['job_id']) for side in ['LEFT','RIGHT']])
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
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=6,discarded_Adam=4*aa['epochs'],synthetic_fixture_NOT_onpolicy_or_new_navigation=True,shadow_and_whole_joint_rollback_exact=True,two_anchor_mean_verified=True))

def launch():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+3000<p['absolute_all_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T491','--why=Bounded uncentered eight metre PPO',
            'timeout','--signal=TERM','--kill-after=2','3000','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=now+3000,source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),eight_only=True,all_stop_authorized=p['absolute_all_stop']))
    print('T491_ACTUALLY_LAUNCHED',c.pid,flush=True)

def text(original):
    f=Path(original)
    if f.exists():return f.read_text()
    row=next(r for r in read(OUT/'protocol.json')['local_encoding'] if Path(r['path'])==f)
    z=Path(row['gzip_path']).read_bytes();assert hashlib.sha256(z).hexdigest()==row['gzip_sha256']
    b=gzip.decompress(z);assert hashlib.sha256(b).hexdigest()==row['sha256'];return b.decode()

if __name__=='__main__':globals()[sys.argv[1]]()

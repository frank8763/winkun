"""A new Monday bounded comparison, with independent fixed stop times."""
from pathlib import Path
import copy,fcntl,importlib.util,json,os,shutil,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t699_matched_value'))
# Import scientific dependencies first; historical generic run ownership must
# be established before importing any continuation's process-only helpers.
import t699_native as N
import t699_common as C
import t701_actor as A
read,save,sha,used=C.read,C.save,C.sha,C.used
OUT=ROOT/'stage/runs/t701_compact_retention';DOC=ROOT/'stage/docs/day_20261123/t701';UNIT='stage-t701.service'
spec=importlib.util.spec_from_file_location('t701_sequential_queue',ROOT/'stage/experiments/t699_unexecuted_continuation/run.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
R.OUT,R.DOC,R.HERE,R.UNIT=OUT,DOC,HERE,UNIT
_outside=None;_last_resource=[0.]

def old_stopped():
    spec=importlib.util.spec_from_file_location('t701_completed_review_guard',ROOT/'stage/experiments/t700r_import_isolation/review.py')
    G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G);G.M.stopped()
    assert read(ROOT/'stage/runs/t700r_import_isolation/audit.json')['state']=='PASS'

def stopped():
    old_stopped()
    prep=subprocess.check_output(['systemctl','--user','show','stage-t701-preflight.service','-p','MainPID','-p','ActiveState'],text=True)
    assert 'MainPID=0' in prep and ('ActiveState=inactive' in prep or 'ActiveState=failed' in prep)
    s=subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','-p','ActiveState'],text=True)
    assert 'MainPID=0' in s and ('ActiveState=inactive' in s or 'ActiveState=failed' in s)
    if (OUT/'DRIVER_STARTED.json').exists():
        r=read(OUT/'SEALED.json');assert r['all_scientific_processes_stopped'] and r['helper_stopped']
        assert all(not Path('/proc',str(pid)).exists() for pid in r['scientific_pids']+[read(OUT/'DRIVER_STARTED.json')['pid']])
    else:assert not (OUT/'LAUNCH_STARTED.json').exists()

def resource_wait(deadline):
    if time.time()-_last_resource[0]<1:return
    while True:
        assert time.time()<deadline
        try:C.resources();_last_resource[0]=time.time();return
        except AssertionError as error:
            if not isinstance(error.args[0],tuple) or not str(error.args[0][0]).startswith('Phase physical CPU overlap'):raise
            # No other cgroup is touched. Fixed deadlines keep advancing.
            time.sleep(.25)

def verify(p,*args,force=False):
    global _outside
    root=read(OUT/'protocol.json');assert time.time()<min(root['absolute_all_stop'],p['absolute_all_stop'])
    assert sha(OUT/'protocol.json')==sha(HERE/'protocol.json')
    own=used(OUT);assert own<root['max_bytes']
    if _outside is None or force:
        now=used(ROOT/'stage/runs')-own
        if _outside is not None:assert now==_outside,'Another Stage output changed during sole-writer science'
        _outside=now
    assert own+_outside<10*2**30 and shutil.disk_usage(ROOT).free>40*2**30
    resource_wait(root['absolute_all_stop'])
    for f,h in root['inputs'].items():
        q=Path(f);assert [q.stat().st_size,q.stat().st_mtime_ns]==root['input_stats'][f],f
        if force:assert sha(q)==h,f

_original_worker=N.worker
def worker(pipe,index,p,counts):
    N.A.frozen=A.frozen
    return _original_worker(pipe,index,p,counts)

def native_child(action,name):
    N.OUT,N.DOC,N.HERE=OUT,DOC,HERE;N.verify=verify;N.worker=worker;N.A.frozen=A.frozen
    if action=='audit':
        lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:N.audit(name)
        finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    else:N.evaluate(name)

def count_calls():
    values=dict(Actor_live=0,Actor_shadow=0,Actor_independent=0)
    f=OUT/'fit/fit_calls.jsonl'
    if f.exists():
        for row in f.read_text().splitlines():
            x=json.loads(row)
            if x['stage']=='BEFORE':values['Actor_'+x['who']]+=1
    return values

def account(items,state):
    base=read(OUT/'FIT_LEDGER.json') if (OUT/'FIT_LEDGER.json').exists() else dict(children=[])
    all_items=base['children']+items
    totals=dict(physics=sum(x.get('physics',0) for x in all_items),commands=sum(x.get('commands',0) for x in all_items),
                science_seconds=sum(x['elapsed_seconds'] for x in all_items if not x['audit']),
                audit_seconds=sum(x['elapsed_seconds'] for x in all_items if x['audit']),**count_calls())
    assert all(totals[k]<=81 for k in ['Actor_live','Actor_shadow','Actor_independent'])
    save(OUT/'LEDGER.json',dict(children=all_items,totals=totals))
    l=read(C.WEEKEND/'LEDGER.json');studies=l.setdefault('new_monday_experiments',[])
    found=[j for j in studies if j['id']=='T701'];assert len(found)==1;j=found[0];before=j.get('accounted',{})
    for k,v in totals.items():assert v>=before.get(k,0);l[k]+=v-before.get(k,0)
    j.update(state=state,accounted=totals)
    for k in ['Actor_live','Actor_shadow','Actor_independent']:assert l[k]<=4096
    assert l['physics']<=1500000000 and l['commands']<=45000 and l['science_seconds']<=180000 and l['audit_seconds']<=18000
    save(C.WEEKEND/'LEDGER.json',l)

R.verify,R.child,R.account=verify,native_child,account

def preflight():
    old_stopped();C.resources();assert not OUT.exists();OUT.mkdir();(OUT/'fit').mkdir()
    p=read(ROOT/'stage/runs/t699_matched_value/protocol.json')
    A.torch.set_num_threads(1);parent=A.P.frozen(p,p['initial']['path'])
    q=A.Actor(p,len(parent.compat_radii),2);missing,extra=q.load_state_dict(parent.state_dict(),strict=False)
    assert set(missing)=={'new_centers','new_radii','new_weights'} and not extra
    q.new_centers.fill_(1e6)
    q.sample=False;parent.sample=False;q.rng=None;parent.rng=None;q.reset_noise();parent.reset_noise()
    sample=p['protected403_rows'][0];arrays=A.Old.Existing.L.T.R.E.N.S.arrays(sample['path'])
    h=None;old_h=None
    with A.torch.no_grad():
        for row in arrays['states'][:8]:
            x=A.torch.tensor(row[None]);cmd,_,h,_=q.step(x,h);old,_,old_h,_=parent.step(x,old_h)
            assert A.Old.Existing.same_float32(cmd.numpy(),old.numpy())
    model=A.export(q,p,OUT/'preflight_zero_head.pt.gz');restored=A.frozen(p,model['path'])
    assert A.digest(q.state_dict())==A.digest(restored.state_dict()) and restored.new_centers.dtype==A.torch.float32
    # Exact full-history inspector and corruption rejection are inherited from
    # the just-completed500-row T700R review; exercise new worker binding here.
    import unittest.mock as mock
    seen=[];before=os.sched_setaffinity
    def witness(pipe,index,pc,counts):
        assert N.N.P.frozen is A.frozen;seen.append(index)
    try:
        with mock.patch.object(N.N.G.T,'worker',witness):worker(None,0,dict(p,stage_cpus=list(range(8,16))),None)
    finally:os.sched_setaffinity=before
    assert seen==[0]
    save(DOC/'PREFLIGHT.json',dict(state='PASS',zero_head_CPU_batch1_exact=True,real_loader_dtype_and_state_exact=True,
        actual_native_worker_new_loader_binding=True,completed_T700R_inspector_and_corruption_gate_reused=True,
        new_native=0,new_optimizer=0,new_Actor=0,resource=C.resources()))

def register():
    stopped();assert read(DOC/'PREFLIGHT.json')['state']=='PASS';assert not (OUT/'protocol.json').exists()
    raw=json.loads((ROOT/'stage/runs/t699_matched_value/protocol.json').read_text());old=read(ROOT/'stage/runs/t699_matched_value/protocol.json')
    auth=read(DOC/'AUTHORIZATION.json');l=read(C.WEEKEND/'LEDGER.json')
    assert all(l[k]+81<=4096 for k in ['Actor_live','Actor_shadow','Actor_independent'])
    assert l['transport_seconds']+600<18000
    p=copy.deepcopy(raw);fit=read(ROOT/'stage/runs/t699_matched_value/fit/summary.json')
    assert sha(fit['finals']['replay_108']['model']['path'])==fit['finals']['replay_108']['model']['sha256']
    p.update(id='T701_COMPACT_RETENTION_FINITE_PILOT',out=str(OUT),authorization_approved=True,
             original_T699_deadlines_unchanged=True,new_Monday_authorization=True,registration_epoch=time.time(),
             absolute_learning_stop=auth['absolute_learning_stop'],absolute_native_stop=auth['absolute_native_stop'],absolute_all_stop=auth['absolute_all_stop'],
             small_replay_initial=fit['finals']['replay_108']['model'],max_bytes=512*2**20,global_bytes=10*2**30,
             max_Actor_live=81,max_Actor_shadow=81,max_Actor_independent=81,
             schedule=[['known','quant32'],['new','quant32'],['known','replay_refined108'],['new','replay_refined108']],
             known_main_seconds=2800,known_audit_seconds=360,new_main_seconds=650,new_audit_seconds=180)
    p['max_physics']=2*sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['screen_jobs']+p['new_jobs']+p['new_exact_jobs'])
    p['max_commands']=2*(len(p['evaluation_jobs'])+len(p['screen_jobs'])+len(p['new_jobs'])+len(p['new_exact_jobs'])+28)
    assert l['physics']+p['max_physics']<1500000000 and l['commands']+p['max_commands']<45000
    assert l['science_seconds']+1200+2*(2800+650)<180000 and l['audit_seconds']+900+2*(360+180)<18000
    assert time.time()+1200+900+2*(2800+360+650+180)+300<p['absolute_all_stop']
    assert time.time()+1200+30<p['absolute_learning_stop']
    inputs=dict(old['inputs'])
    extra=set(HERE.glob('*.py'))|{DOC/'AUTHORIZATION.json',DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'COMPACT_RETENTION_PRIMARY_REVIEW.md',
          ROOT/'stage/runs/t699_matched_value/fit/summary.json',ROOT/'stage/runs/t699_matched_value/fit/audit.json',Path(p['small_replay_initial']['path']),
          ROOT/'stage/runs/t700r_import_isolation/audit.json'}
    inputs.update({str(f):sha(f) for f in extra});rows=[[f,h,Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f,h in sorted(inputs.items())]
    (HERE/'inputs.json').write_text(json.dumps(rows,separators=(',',':'))+'\n');assert (HERE/'inputs.json').stat().st_size<2**21
    p['_input_manifest']=dict(path=str(HERE/'inputs.json'),sha256=sha(HERE/'inputs.json'))
    blob=json.dumps(p,separators=(',',':'))+'\n';assert len(blob.encode())<2**21
    (OUT/'protocol.json').write_text(blob);(HERE/'protocol.json').write_text(blob)
    l.setdefault('new_monday_experiments',[]).append(dict(id='T701',state='RESERVED',out=str(OUT),Actor_each_cap=81,max_physics=p['max_physics'],max_commands=p['max_commands']))
    save(C.WEEKEND/'LEDGER.json',l);save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',absolute_all_stop=p['absolute_all_stop'],inputs=len(inputs),new_authorization_not_extension=True))
    verify(read(OUT/'protocol.json'),force=True);print('T701_REGISTERED',len(inputs),p['max_physics'],p['max_commands'])

def sync():
    stopped();p=read(OUT/'protocol.json');l=read(C.WEEKEND/'LEDGER.json')
    assert l['transport_seconds']+600<18000 and time.time()+300<p['absolute_all_stop']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T701 source300')));signal.alarm(295)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as Archive
    Archive.stopped=stopped;Archive.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'));import t626_sync_source
    t626_sync_source.main();stopped();signal.alarm(0)

def launch():
    stopped();p=read(OUT/'protocol.json');verify(p,force=True);assert not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time()-3);assert seconds>10200
    cmd=['systemd-run','--user','--unit=stage-t701','--collect','--property=Type=exec','--property=CPUQuota=800%',
         '--property=CPUWeight=10','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=Nice=19','--property=MemoryMax=8G',
         '--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={seconds}',
         f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit',
         '--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=8','--setenv=MKL_NUM_THREADS=8',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver']
    c=subprocess.run(cmd,text=True,capture_output=True);assert c.returncode==0,c.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit=UNIT,source_remote=receipt['remote_commit']));print(c.stderr)

def driver():
    p=read(OUT/'protocol.json');verify(p,force=True);assert not (OUT/'LAUNCH_STARTED.json').exists()
    save(OUT/'LAUNCH_STARTED.json',dict(pid=os.getpid(),epoch=time.time()));items=[];ids=[]
    try:
        for mode,seconds in [('main',1200),('audit',900)]:
            with (OUT/('fit_'+mode+'.log')).open('xb') as log:
                child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'fit.py'),mode],stdout=log,stderr=log,start_new_session=True)
                ids.append(child.pid);start=time.time();save(OUT/'status.json',dict(state='ACTIVE',child='fit',action=mode,child_pid=child.pid,epoch=start))
                try:child.wait(timeout=seconds);assert child.returncode==0,('fit',mode,child.returncode)
                except BaseException:
                    R.scoped_signal(child.pid,signal.SIGCONT);R.scoped_signal(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:R.scoped_signal(child.pid,signal.SIGKILL);child.wait()
                    raise
                finally:
                    items.append(dict(name='fit_'+mode,pid=child.pid,audit=mode=='audit',elapsed_seconds=time.time()-start,returncode=child.returncode))
                    save(OUT/'FIT_LEDGER.json',dict(children=items));account([],'ACTIVE')
        assert read(OUT/'fit/audit.json')['state']=='PASS'
        R.driver()
    except BaseException:
        error=traceback.format_exc();report=dict(state='FAILED',error=error,scientific_pids=ids,helper_stopped=True,
            all_scientific_processes_stopped=all(not Path('/proc',str(i)).exists() for i in ids),retained='T698',epoch=time.time())
        save(OUT/'SEALED.json',report);save(OUT/'FINAL_REPORT.json',report);save(OUT/'status.json',report);save(DOC/'FINAL_REPORT.json',report)
        account([],'FAILED')
    finally:
        if not (OUT/'DRIVER_STARTED.json').exists():save(OUT/'DRIVER_STARTED.json',read(OUT/'LAUNCH_STARTED.json'))
        if (OUT/'SEALED.json').exists():
            r=read(OUT/'SEALED.json');r['scientific_pids']=sorted(set(r['scientific_pids']+ids));r['all_scientific_processes_stopped']=all(not Path('/proc',str(i)).exists() for i in r['scientific_pids'])
            save(OUT/'SEALED.json',r);save(OUT/'FINAL_REPORT.json',r);save(DOC/'FINAL_REPORT.json',r)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ['evaluate','audit']:native_child(mode,sys.argv[2])
    else:{'preflight':preflight,'register':register,'sync':sync,'launch':launch,'driver':driver,'helper':R.helper}[mode]()

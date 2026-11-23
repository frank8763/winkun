"""Only unexecuted T701 native blocks; pure-stdlib helper dispatched first."""
from pathlib import Path
import importlib.util,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t701r_native_entry';DOC=ROOT/'stage/docs/day_20261123/t701r';UNIT='stage-t701r.service'
# The helper's actual /usr/bin/python3 must never import model/MuJoCo modules.
if __name__=='__main__' and sys.argv[1]=='helper':
    spec=importlib.util.spec_from_file_location('t701r_pure_resource_helper',ROOT/'stage/experiments/t699_unexecuted_continuation/run.py')
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    test=len(sys.argv)>2 and sys.argv[2]=='preflight'
    H.OUT=OUT/'helper_preflight' if test else OUT;H.DOC=DOC/'helper_preflight' if test else DOC
    H.UNIT='stage-t701r-helper-test.service' if test else UNIT
    H.helper();sys.exit(0)
sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
# Model-owning imports establish the legacy generic run namespace only in
# science/transport processes using the original qualified environment.
spec=importlib.util.spec_from_file_location('t701r_verified_original_controller',ROOT/'stage/experiments/t701_compact_retention/run.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
C=M.C;read,save,sha,used=C.read,C.save,C.sha,C.used
M.OUT,M.DOC,M.HERE,M.UNIT=OUT,DOC,HERE,UNIT
R=M.R;R.OUT,R.DOC,R.HERE,R.UNIT=OUT,DOC,HERE,UNIT
_verified_worker=M.worker
def worker(pipe,index,p,counts):
    # Spawn must serialize a function owned by this real entry module, rather
    # than the process-local importlib alias used for read-only code reuse.
    return _verified_worker(pipe,index,p,counts)
M.worker=worker

def old_stopped():
    spec=importlib.util.spec_from_file_location('t701r_old_review_guard',ROOT/'stage/experiments/t700r_import_isolation/review.py')
    G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G);G.M.stopped()
    old=ROOT/'stage/runs/t701_compact_retention';seal=read(old/'SEALED.json')
    assert seal['state']=='FAILED' and seal['all_scientific_processes_stopped'] and seal['helper_stopped']
    ids=seal['scientific_pids']+[read(old/'DRIVER_STARTED.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    for unit in ['stage-t701.service','stage-t701-finalizer.service','stage-t701-preflight.service']:
        s=subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState'],text=True)
        assert 'MainPID=0' in s and ('ActiveState=inactive' in s or 'ActiveState=failed' in s)
    assert not any((old/name).exists() for name in ['known_quant32','new_quant32','known_replay_refined108','new_replay_refined108'])
    assert read(old/'fit/audit.json')['state']=='PASS'

def stopped():
    old_stopped()
    for unit in [UNIT,'stage-t701r-helper-test.service']:
        s=subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState'],text=True)
        assert 'MainPID=0' in s and ('ActiveState=inactive' in s or 'ActiveState=failed' in s)
    if (OUT/'DRIVER_STARTED.json').exists():
        seal=read(OUT/'SEALED.json');assert seal['all_scientific_processes_stopped'] and seal['helper_stopped']
        assert all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids']+[read(OUT/'DRIVER_STARTED.json')['pid']])
    else:assert not (OUT/'launch.json').exists()

def account(items,state):
    totals=dict(physics=sum(i.get('physics',0) for i in items),commands=sum(i.get('commands',0) for i in items),
        science_seconds=sum(i['elapsed_seconds'] for i in items if not i['audit']),audit_seconds=sum(i['elapsed_seconds'] for i in items if i['audit']))
    save(OUT/'LEDGER.json',dict(children=items,totals=totals,new_Actor=0,new_optimizer=0))
    l=read(C.WEEKEND/'LEDGER.json');j=next(j for j in l['new_monday_experiments'] if j['id']=='T701R');before=j.get('accounted',{})
    for k,v in totals.items():assert v>=before.get(k,0);l[k]+=v-before.get(k,0)
    j.update(state=state,accounted=totals);save(C.WEEKEND/'LEDGER.json',l)
R.account=account

def register():
    old_stopped();auth=read(ROOT/'stage/docs/day_20261123/t701/AUTHORIZATION.json');old=ROOT/'stage/runs/t701_compact_retention'
    assert not OUT.exists();OUT.mkdir();(OUT/'fit').mkdir()
    p=json.loads((old/'protocol.json').read_text());p.update(id='T701R_ONLY_UNEXECUTED_NATIVE',out=str(OUT),new_Actor=0,new_optimizer=0,
        original_T701_stays_FAILED=True,reused_fit_root=str(old),max_bytes=p['max_bytes']-used(old))
    assert p['absolute_all_stop']==auth['absolute_all_stop'] and time.time()+2*(2800+360+650+180)+180<p['absolute_all_stop']
    l=read(C.WEEKEND/'LEDGER.json');assert l['transport_seconds']+120<=18000
    for name in ['summary.json','audit.json']:(OUT/'fit'/name).write_bytes((old/'fit'/name).read_bytes())
    save(OUT/'REUSED_FIT.json',dict(model_paths_unchanged=True,source_summary_sha256=sha(old/'fit/summary.json'),source_audit_sha256=sha(old/'fit/audit.json'),new_Actor=0,new_optimizer=0,source_models_already_remote_primary_verified=True))
    # Test the exact helper interpreter/entry/cgroup before source registration.
    test=OUT/'helper_preflight';test.mkdir();(DOC/'helper_preflight').mkdir()
    q=dict(p,absolute_all_stop=time.time()+1.5);save(test/'protocol.json',q);save(test/'status.json',dict(child_pid=None))
    cmd=['systemd-run','--user','--unit=stage-t701r-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15',
        '--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',
        f'--property=WorkingDirectory={ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight']
    r=subprocess.run(cmd,text=True,capture_output=True);assert r.returncode==0,r.stderr
    assert read(DOC/'helper_preflight/AFFINITY_GUARD.json')['state']=='ENDED'
    save(DOC/'PREFLIGHT.json',dict(state='PASS',actual_system_python_helper_cgroup_and_loop=True,no_model_import_in_helper=True,new_native=0,new_Actor=0))
    original=read(old/'protocol.json');inputs=dict(original['inputs'])
    for f in list(HERE.glob('*.py'))+[DOC/'PLAN.md',DOC/'PREFLIGHT.json',old/'fit/summary.json',old/'fit/audit.json',old/'SEALED.json',old/'helper.log']:
        inputs[str(f)]=sha(f)
    for model in read(old/'fit/summary.json')['finals'].values():assert sha(model['model']['path'])==model['model']['sha256'];inputs[model['model']['path']]=model['model']['sha256']
    rows=[[f,h,Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f,h in sorted(inputs.items())]
    (HERE/'inputs.json').write_text(json.dumps(rows,separators=(',',':'))+'\n');assert (HERE/'inputs.json').stat().st_size<2**21
    p['_input_manifest']=dict(path=str(HERE/'inputs.json'),sha256=sha(HERE/'inputs.json'))
    blob=json.dumps(p,separators=(',',':'))+'\n';assert len(blob.encode())<2**21
    (OUT/'protocol.json').write_text(blob);(HERE/'protocol.json').write_text(blob)
    l['new_monday_experiments'].append(dict(id='T701R',state='RESERVED',out=str(OUT),new_Actor=0,original_T701_absolute_all_stop_unchanged=p['absolute_all_stop'],accounted={}))
    save(C.WEEKEND/'LEDGER.json',l);save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',inputs=len(inputs),absolute_all_stop=p['absolute_all_stop'],source_action_seconds=60))
    M.verify(read(OUT/'protocol.json'),force=True);print('T701R_REGISTERED_ZERO_NEW_FIT')

def sync():
    stopped();l=read(C.WEEKEND/'LEDGER.json');p=read(OUT/'protocol.json')
    assert l['transport_seconds']+120<=18000 and time.time()+60<p['absolute_all_stop']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T701R source60 cap')));signal.alarm(55)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as Archive
    Archive.stopped=stopped;Archive.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'));import t626_sync_source
    t626_sync_source.main();stopped();signal.alarm(0)

def launch():
    stopped();p=read(OUT/'protocol.json');M.verify(p,force=True);assert not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip();assert receipt['root_tree']==tree
    seconds=int(p['absolute_all_stop']-time.time()-5)
    cmd=['systemd-run','--user','--unit=stage-t701r','--collect','--property=Type=exec','--property=CPUQuota=800%',
        '--property=CPUWeight=10','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=Nice=19','--property=MemoryMax=8G',
        '--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=3',f'--property=RuntimeMaxSec={seconds}',
        f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit',
        '--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver']
    c=subprocess.run(cmd,text=True,capture_output=True);assert c.returncode==0,c.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),source_remote=receipt['remote_commit'],root_tree=tree,unit=UNIT));print(c.stderr)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ['evaluate','audit']:M.native_child(mode,sys.argv[2])
    else:{'register':register,'sync':sync,'launch':launch,'driver':R.driver}[mode]()

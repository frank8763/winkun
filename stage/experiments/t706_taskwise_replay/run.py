"""Fixed-feature convex constrained replay; all original jobs remain completed."""
from pathlib import Path
import importlib.util,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t706_taskwise_replay';DOC=ROOT/'stage/docs/day_20261123/t706';UNIT='stage-t706.service'
if __name__=='__main__' and sys.argv[1]=='helper':
    spec=importlib.util.spec_from_file_location('t706_pure_resource_helper',ROOT/'stage/experiments/t699_unexecuted_continuation/run.py')
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    test=len(sys.argv)>2 and sys.argv[2]=='preflight'
    H.OUT=OUT/'helper_preflight' if test else OUT;H.DOC=DOC/'helper_preflight' if test else DOC
    H.UNIT='stage-t706-helper-test.service' if test else UNIT
    H.helper();sys.exit(0)
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
spec=importlib.util.spec_from_file_location('t706_reused_native_controller',ROOT/'stage/experiments/t701_compact_retention/run.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
sys.path.insert(0,str(ROOT/'stage/experiments/t703_projected_local'))
import t703_actor as A
C=M.C;read,save,sha,used=C.read,C.save,C.sha,C.used
M.OUT,M.DOC,M.HERE,M.UNIT,M.A=OUT,DOC,HERE,UNIT,A
R=M.R;R.OUT,R.DOC,R.HERE,R.UNIT=OUT,DOC,HERE,UNIT
_worker=M.worker
def worker(pipe,index,p,counts):return _worker(pipe,index,p,counts)
M.worker=worker
verify=M.verify

def unit_stopped(name):
    s=subprocess.run(['systemctl','--user','show',name,'-p','MainPID','-p','ActiveState'],capture_output=True,text=True)
    assert 'MainPID=0' in s.stdout and ('ActiveState=inactive' in s.stdout or 'ActiveState=failed' in s.stdout),(name,s.stdout,s.stderr)

def old_stopped():
    spec=importlib.util.spec_from_file_location('t706_previous_stopped_guard',ROOT/'stage/experiments/t705_constrained_replay/run.py')
    G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G);G.stopped()
    unit_stopped('stage-t705-finalizer.service')
    assert read(ROOT/'stage/runs/t705_constrained_replay/FINAL_REPORT.json')['state']=='COMPLETE'

def stopped():
    old_stopped()
    for unit in [UNIT,'stage-t706-helper-test.service','stage-t706-preflight.service']:unit_stopped(unit)
    if (OUT/'LAUNCH_STARTED.json').exists():
        s=read(OUT/'SEALED.json');assert s['all_scientific_processes_stopped'] and s['helper_stopped']
        ids=s['scientific_pids']+[read(OUT/'LAUNCH_STARTED.json')['pid']]
        assert all(not Path('/proc',str(i)).exists() for i in ids)
    else:assert not (OUT/'launch.json').exists()

def count_calls():
    totals={k:0 for k in ['Solver_live','Solver_shadow','Solver_independent','Candidate_live','Candidate_shadow','Candidate_independent']}
    f=OUT/'fit/fit_calls.jsonl'
    if f.exists():
        for line in f.read_text().splitlines():
            r=json.loads(line)
            if r['stage']=='BEFORE':totals[('Solver_' if r['kind']=='conic_iteration' else 'Candidate_')+r['who']]+=1
    return totals

def account(items,state):
    base=read(OUT/'FIT_LEDGER.json') if (OUT/'FIT_LEDGER.json').exists() else dict(children=[])
    allitems=base['children']+items
    totals=dict(physics=sum(i.get('physics',0) for i in allitems),commands=sum(i.get('commands',0) for i in allitems),
        science_seconds=sum(i['elapsed_seconds'] for i in allitems if not i['audit']),audit_seconds=sum(i['elapsed_seconds'] for i in allitems if i['audit']),**count_calls())
    p=read(OUT/'protocol.json')
    for key in ['Solver_live','Solver_shadow','Solver_independent']:assert totals[key]<=256
    for key in ['Candidate_live','Candidate_shadow','Candidate_independent']:assert totals[key]<=2
    assert totals['physics']<=p['max_physics'] and totals['commands']<=p['max_commands']
    save(OUT/'LEDGER.json',dict(children=allitems,totals=totals,new_Adam=0,new_RL=0,new_Value=0,new_explicit_learning_ledger=True))
    ledger=read(C.WEEKEND/'LEDGER.json');job=next(j for j in ledger['new_monday_experiments'] if j['id']=='T706');before=job.get('accounted',{})
    for key in ['physics','commands','science_seconds','audit_seconds']:
        value=totals[key];assert value>=before.get(key,0);ledger[key]+=value-before.get(key,0)
    assert ledger['Actor_live']==4089 and ledger['Actor_shadow']==4089 and ledger['Actor_independent']==1785
    assert ledger['physics']<=1500000000 and ledger['commands']<=45000 and ledger['science_seconds']<=180000 and ledger['audit_seconds']<=18000
    job.update(state=state,accounted=totals);save(C.WEEKEND/'LEDGER.json',ledger)
R.account=account

_base_verify=M.verify
def verify(p,*args,force=False):
    _base_verify(p,*args,force=force)
    for f,h,n,mt in read(DOC/'DEPENDENCIES.json')['entries']:
        s=Path(f).stat();assert (s.st_size,s.st_mtime_ns)==(n,mt),f
        if force:assert sha(f)==h,f
M.verify=verify;R.verify=verify

def register():
    stopped();assert read(DOC/'PREFLIGHT.json')['state']=='PASS' and not (OUT/'protocol.json').exists()
    p=json.loads((ROOT/'stage/runs/t705_constrained_replay/protocol.json').read_text());old=read(ROOT/'stage/runs/t705_constrained_replay/protocol.json')
    auth=read(DOC/'AUTHORIZATION.json');ledger=read(C.WEEKEND/'LEDGER.json')
    reference=read(ROOT/'stage/runs/t701_compact_retention/fit/summary.json')['finals']['replay_refined108']['model']
    assert sha(reference['path'])==reference['sha256']
    p.update(id='T706_TASKWISE_MEAN_VS_GROUPDRO',out=str(OUT),registration_epoch=time.time(),
        absolute_learning_stop=auth['absolute_learning_stop'],absolute_native_stop=auth['absolute_native_stop'],absolute_all_stop=auth['absolute_all_stop'],
        max_bytes=3*2**30,child_bytes=1280*2**20,global_bytes=10*2**30,
        schedule=[['known','mean_taskwise'],['new','mean_taskwise'],['known','dro_taskwise'],['new','dro_taskwise']],
        max_Actor_live=0,max_Actor_shadow=0,max_Actor_independent=0,new_Actor=0,separate_solver_iterations_each_cap=256,separate_candidate_each_cap=2,new_optimizer=0,new_Value=0,new_Adam=0,new_RL=0,
        repair_initial=reference,fit_main_seconds=900,fit_audit_seconds=600,
        known_main_seconds=2800,known_audit_seconds=360,new_main_seconds=2800,new_audit_seconds=600,
        transport_budget_scope='T706_NEW_SEPARATE_900S',retained='T698')
    p['max_physics']=2*sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['screen_jobs']+p['new_jobs']+p['new_exact_jobs'])
    p['max_commands']=2*(len(p['evaluation_jobs'])+len(p['screen_jobs'])+len(p['new_jobs'])+len(p['new_exact_jobs'])+28)
    assert time.time()+900+600+11200+1920+240+900<p['absolute_all_stop']
    assert time.time()+900+30<p['absolute_learning_stop']
    assert used(ROOT/'stage/runs')+p['max_bytes']<8*2**30
    assert ledger['Actor_live']==4089 and ledger['Actor_shadow']==4089 and ledger['Actor_independent']==1785
    assert ledger['physics']+p['max_physics']<=1500000000 and ledger['commands']+p['max_commands']<=45000
    assert ledger['science_seconds']+12100<=180000 and ledger['audit_seconds']+2520<=18000
    inputs=dict(old['inputs']);extra=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'AUTHORIZATION.json',DOC/'PREFLIGHT.json',DOC.parent/'T706_TASKWISE_REPLAY_PRIMARY_REVIEW.md',DOC/'DEPENDENCIES.json',
      ROOT/'stage/runs/t699_matched_value/DATA.pt.gz',ROOT/'stage/runs/t705_constrained_replay/FINAL_REPORT.json',DOC.parent/'t704/PAIRED_RESULTS.json',ROOT/'stage/runs/t701r_native_entry/known_replay_refined108/summary.json',Path(reference['path'])}
    inputs.update({str(f):sha(f) for f in extra});rows=[[f,h,Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f,h in sorted(inputs.items())]
    (HERE/'inputs.json').write_text(json.dumps(rows,separators=(',',':'))+'\n');assert (HERE/'inputs.json').stat().st_size<2**21
    p['_input_manifest']=dict(path=str(HERE/'inputs.json'),sha256=sha(HERE/'inputs.json'));p.pop('inputs',None);p.pop('input_stats',None)
    blob=json.dumps(p,separators=(',',':'))+'\n';assert len(blob.encode())<2**21
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    ledger['new_monday_experiments'].append(dict(id='T706',state='RESERVED',out=str(OUT),Actor_old_ledger_cap_unchanged=True,separate_solver_each_cap=256,separate_candidate_each_cap=2,max_physics=p['max_physics'],max_commands=p['max_commands'],transport_separate_cap=900));save(C.WEEKEND/'LEDGER.json',ledger)
    save(DOC/'TRANSPORT_LEDGER.json',dict(cap=900,charged_seconds=0,source_commits=[],attempts=[],old_campaign_transport_frozen=ledger['transport_seconds']))
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',inputs=len(inputs),max_physics=p['max_physics'],max_commands=p['max_commands'],absolute_all_stop=p['absolute_all_stop']))
    verify(read(OUT/'protocol.json'),force=True);print('T706_REGISTERED',len(inputs),p['max_physics'],p['max_commands'])

def sync(final=False):
    stopped();p=read(OUT/'protocol.json');assert time.time()+300<p['absolute_all_stop']
    l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+300+(0 if final else 300)<=l['cap']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T706 source300')));signal.alarm(295)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as Guard
    Guard.stopped=stopped;Guard.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'));import t626_sync_source
    t626_sync_source.main();stopped();signal.alarm(0)

def launch():
    stopped();p=read(OUT/'protocol.json');verify(p,force=True);assert not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip();assert receipt['root_tree']==tree
    seconds=int(p['absolute_all_stop']-time.time()-3)
    cmd=['systemd-run','--user','--unit=stage-t706','--collect','--property=Type=exec','--property=CPUQuota=800%','--property=CPUWeight=10',
        '--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=Nice=19','--property=MemoryMax=8G','--property=MemorySwapMax=0',
        '--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={seconds}',f'--property=WorkingDirectory={ROOT}',
        f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',
        '--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver']
    result=subprocess.run(cmd,capture_output=True,text=True);assert result.returncode==0,result.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit=UNIT,source_remote=receipt['remote_commit'],root_tree=tree));print(result.stderr)

def driver():
    p=read(OUT/'protocol.json');verify(p,force=True);assert not (OUT/'LAUNCH_STARTED.json').exists()
    save(OUT/'LAUNCH_STARTED.json',dict(pid=os.getpid(),epoch=time.time()));items=[];ids=[]
    try:
        for mode,seconds in [('main',900),('audit',600)]:
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
                    items.append(dict(name='fit_'+mode,pid=child.pid,audit=mode=='audit',returncode=child.returncode,elapsed_seconds=time.time()-start))
                    save(OUT/'FIT_LEDGER.json',dict(children=items));account([],'ACTIVE')
        assert read(OUT/'fit/audit.json')['state']=='PASS';R.driver()
    except BaseException:
        report=dict(state='FAILED',error=traceback.format_exc(),scientific_pids=ids,helper_stopped=True,
            all_scientific_processes_stopped=all(not Path('/proc',str(i)).exists() for i in ids),retained='T698',epoch=time.time())
        save(OUT/'SEALED.json',report);save(OUT/'FINAL_REPORT.json',report);save(OUT/'status.json',report);save(DOC/'FINAL_REPORT.json',report);account([],'FAILED')
    finally:
        if not (OUT/'DRIVER_STARTED.json').exists():save(OUT/'DRIVER_STARTED.json',read(OUT/'LAUNCH_STARTED.json'))
        if (OUT/'SEALED.json').exists():
            r=read(OUT/'SEALED.json');r['scientific_pids']=sorted(set(r['scientific_pids']+ids));r['all_scientific_processes_stopped']=all(not Path('/proc',str(i)).exists() for i in r['scientific_pids'])
            r.update(id='T706',new_Actor=count_calls()['Candidate_live'],separate_solver_iterations=count_calls()['Solver_live'],new_Adam=0,new_RL=0)
            save(OUT/'SEALED.json',r);save(OUT/'FINAL_REPORT.json',r);save(OUT/'status.json',r);save(DOC/'FINAL_REPORT.json',r)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ['evaluate','audit']:M.native_child(mode,sys.argv[2])
    else:{'register':register,'sync':sync,'launch':launch,'driver':driver}[mode]()

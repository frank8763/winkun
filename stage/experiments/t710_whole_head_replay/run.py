"""Fixed-capacity original versus learner-induced-state replay; old failures preserved."""
from pathlib import Path
import importlib.util,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t710_whole_head_replay';DOC=ROOT/'stage/docs/overnight_20261123_24/t710';UNIT='stage-t710.service'
if __name__=='__main__' and sys.argv[1]=='helper':
    spec=importlib.util.spec_from_file_location('t710_pure_resource_helper',ROOT/'stage/experiments/t699_unexecuted_continuation/run.py')
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    test=len(sys.argv)>2 and sys.argv[2]=='preflight'
    H.OUT=OUT/'helper_preflight' if test else OUT;H.DOC=DOC/'helper_preflight' if test else DOC
    H.UNIT='stage-t710-helper-test.service' if test else UNIT
    H.helper();sys.exit(0)
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
spec=importlib.util.spec_from_file_location('t710_reused_native_controller',ROOT/'stage/experiments/t701_compact_retention/run.py')
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
    spec=importlib.util.spec_from_file_location('t710_previous_stopped_guard',ROOT/'stage/experiments/t709_induced_replay/run.py')
    G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G);G.stopped()
    unit_stopped('stage-t709-finalizer.service')
    assert (ROOT/'stage/runs/t709_induced_replay/FINAL_REPORT.json').exists()

def stopped():
    old_stopped()
    for unit in [UNIT,'stage-t710-helper-test.service','stage-t710-preflight.service']:unit_stopped(unit)
    if (OUT/'LAUNCH_STARTED.json').exists():
        s=read(OUT/'SEALED.json');assert s['all_scientific_processes_stopped'] and s['helper_stopped']
        ids=s['scientific_pids']+[read(OUT/'LAUNCH_STARTED.json')['pid']]
        assert all(not Path('/proc',str(i)).exists() for i in ids)
    else:assert not (OUT/'launch.json').exists()

def count_calls():
    totals={k:0 for k in ['Gradient_live','Gradient_shadow','Gradient_independent','Optimizer_live','Optimizer_shadow','Optimizer_independent']}
    f=OUT/'fit/fit_calls.jsonl'
    if f.exists():
        for line in f.read_text().splitlines():
            r=json.loads(line)
            if r['stage']=='BEFORE':totals[('Gradient_' if r['kind']=='gradient_evaluation' else 'Optimizer_')+r['who']]+=1
    return totals

def account(items,state):
    base=read(OUT/'FIT_LEDGER.json') if (OUT/'FIT_LEDGER.json').exists() else dict(children=[])
    allitems=base['children']+items
    totals=dict(physics=sum(i.get('physics',0) for i in allitems),commands=sum(i.get('commands',0) for i in allitems),science_seconds=sum(i['elapsed_seconds'] for i in allitems if not i['audit']),audit_seconds=sum(i['elapsed_seconds'] for i in allitems if i['audit']),**count_calls())
    p=read(OUT/'protocol.json')
    for who in ['live','shadow','independent']:assert totals['Gradient_'+who]<=320 and totals['Optimizer_'+who]<=256
    assert totals['physics']<=p['max_physics'] and totals['commands']<=p['max_commands']
    save(OUT/'LEDGER.json',dict(children=allitems,totals=totals,new_Adam=0,new_RL=0,new_Value=0,new_explicit_LBFGS_ledger=True))
    ledger=read(C.WEEKEND/'LEDGER.json');job=next(j for j in ledger['new_monday_experiments'] if j['id']=='T710');before=job.get('accounted',{})
    for key in ['physics','commands','science_seconds','audit_seconds']:
        value=totals[key];assert value>=before.get(key,0);ledger[key]+=value-before.get(key,0)
    assert ledger['Actor_live']==4089 and ledger['Actor_shadow']==4089 and ledger['Actor_independent']==1785
    assert ledger['physics']<=1500000000 and ledger['commands']<=45000 and ledger['science_seconds']<=180000 and ledger['audit_seconds']<=18000
    job.update(state=state,accounted=totals);save(C.WEEKEND/'LEDGER.json',ledger)
R.account=account

verify=M.verify;R.verify=verify

def register():
    stopped();assert read(DOC/'PREFLIGHT.json')['state']=='PASS' and not (OUT/'protocol.json').exists()
    old=read(ROOT/'stage/runs/t709_induced_replay/protocol.json');p=dict(old)
    auth=read(DOC/'AUTHORIZATION.json');ledger=read(C.WEEKEND/'LEDGER.json')
    p.update(id='T710_WHOLE_HEAD_NOMINAL_VS_INDUCED_REPLAY',out=str(OUT),registration_epoch=time.time(),
        absolute_learning_stop=auth['absolute_learning_stop'],absolute_native_stop=min(auth['absolute_native_stop'],time.time()+20400),absolute_all_stop=min(auth['absolute_all_stop'],time.time()+21600),
        max_bytes=3*2**30,child_bytes=1280*2**20,global_bytes=10*2**30,schedule=[['known','nominal'],['new','nominal'],['known','induced'],['new','induced']],
        max_Actor_live=0,max_Actor_shadow=0,max_Actor_independent=0,new_Actor=0,new_optimizer=0,new_Value=0,new_Adam=0,new_RL=0,separate_gradient_evaluations_each_cap=320,separate_optimizer_attempts_each_cap=256,
        induced_selection=read(ROOT/'stage/docs/day_20261123/t709/INDUCED_SELECTION.json'),fit_main_seconds=2400,fit_audit_seconds=1200,
        known_main_seconds=2800,known_audit_seconds=360,new_main_seconds=2800,new_audit_seconds=600,transport_budget_scope='T710_NEW_SEPARATE_900S',retained='T698')
    p.pop('separate_linear_solves_each_cap',None)
    p['max_physics']=2*sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['screen_jobs']+p['new_jobs']+p['new_exact_jobs'])
    p['max_commands']=2*(len(p['evaluation_jobs'])+len(p['screen_jobs'])+len(p['new_jobs'])+len(p['new_exact_jobs'])+28)
    assert time.time()+2400+1200+11200+1920+240+900<p['absolute_all_stop']
    assert time.time()+2400+30<p['absolute_learning_stop']
    assert used(ROOT/'stage/runs')+p['max_bytes']<8*2**30
    assert ledger['physics']+p['max_physics']<=1500000000 and ledger['commands']+p['max_commands']<=45000
    assert ledger['science_seconds']+13600<=180000 and ledger['audit_seconds']+3120<=18000
    inputs=dict(old['inputs']);extra=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'AUTHORIZATION.json',DOC/'PREFLIGHT.json',DOC.parent/'PRIMARY_REVIEW.md',DOC.parent/'PLAN.md',DOC.parent/'AUTHORIZATION.json',ROOT/'stage/experiments/t709_induced_replay/fit.py',ROOT/'stage/runs/t709_induced_replay/FINAL_REPORT.json',ROOT/'stage/docs/day_20261123/t709/PAIRED_RESULTS.json',ROOT/'stage/runs/t709_induced_replay/fit/summary.json',ROOT/'stage/runs/t709_induced_replay/fit/audit.json'}
    inputs.update({str(f):sha(f) for f in extra});rows=[[f,h,Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f,h in sorted(inputs.items())]
    (HERE/'inputs.json').write_text(json.dumps(rows,separators=(',',':'))+'\n');assert (HERE/'inputs.json').stat().st_size<2**21
    p['_input_manifest']=dict(path=str(HERE/'inputs.json'),sha256=sha(HERE/'inputs.json'));p.pop('inputs',None);p.pop('input_stats',None)
    blob=json.dumps(p,separators=(',',':'))+'\n';assert len(blob.encode())<2**21
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    ledger['new_monday_experiments'].append(dict(id='T710',state='RESERVED',out=str(OUT),Actor_old_ledger_cap_unchanged=True,separate_gradient_each_cap=320,separate_optimizer_attempts_each_cap=256,max_physics=p['max_physics'],max_commands=p['max_commands'],transport_separate_cap=900));save(C.WEEKEND/'LEDGER.json',ledger)
    save(DOC/'TRANSPORT_LEDGER.json',dict(cap=900,charged_seconds=0,source_commits=[],attempts=[],old_campaign_transport_frozen=ledger['transport_seconds']))
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',inputs=len(inputs),max_physics=p['max_physics'],max_commands=p['max_commands'],absolute_all_stop=p['absolute_all_stop']))
    verify(read(OUT/'protocol.json'),force=True);print('T710_REGISTERED',len(inputs),p['max_physics'],p['max_commands'])

def sync(final=False):
    stopped();p=read(OUT/'protocol.json');assert time.time()+300<p['absolute_all_stop']
    l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+300+(0 if final else 300)<=l['cap']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T710 source300')));signal.alarm(295)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as Guard
    Guard.stopped=stopped;Guard.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'));import t626_sync_source
    t626_sync_source.main();stopped();signal.alarm(0)

def launch():
    stopped();p=read(OUT/'protocol.json');verify(p,force=True);assert not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip();assert receipt['root_tree']==tree
    seconds=int(p['absolute_all_stop']-time.time()-3)
    cmd=['systemd-run','--user','--unit=stage-t710','--collect','--property=Type=exec','--property=CPUQuota=800%','--property=CPUWeight=10',
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
        for mode,seconds in [('main',2400),('audit',1200)]:
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
            r.update(id='T710',new_Actor_candidate_upper=2 if count_calls()['Gradient_live'] else 0,new_optimizer_attempts=count_calls()['Optimizer_live'],new_gradient_evaluations=count_calls()['Gradient_live'],new_Adam=0,new_RL=0)
            save(OUT/'SEALED.json',r);save(OUT/'FINAL_REPORT.json',r);save(OUT/'status.json',r);save(DOC/'FINAL_REPORT.json',r)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ['evaluate','audit']:M.native_child(mode,sys.argv[2])
    else:{'register':register,'sync':sync,'launch':launch,'driver':driver}[mode]()

"""Single-launch sequential overnight comparison and actual cumulative ledger."""
import ast,fcntl,gzip,os,signal,subprocess,sys,time,traceback
from t699_common import *

def count_calls():
    result={key:0 for key in ['Actor_live','Actor_shadow','Actor_independent']}
    f=OUT/'fit/fit_calls.jsonl'
    if f.exists():
        for line in f.read_text().splitlines():
            v=json.loads(line)
            if v['stage']=='BEFORE':result['Actor_'+v['who']]+=1
    return result

def native_preflight(p):
    import t699_actor as A
    import t699_native as N
    import unittest.mock as mock
    import mujoco
    A.torch.set_num_threads(1)
    def forbidden(*args,**kwargs):raise AssertionError('No optimizer or native in actual worker/audit preflight')
    mujoco.mj_step=forbidden;A.torch.optim.Adam.step=forbidden
    child=dict(p,out=str(OUT/'native_preflight'),arm='preflight',max_physics=0,max_commands=14,main_seconds=120,audit_seconds=0)
    N.setup(child)
    sample=p['protected403_rows'][0];a=A.Existing.L.T.R.E.N.S.arrays(sample['path'])
    row=dict(job_id=sample['id'],request=dict(model=p['initial']['path']),seed=None,teacher=False,
        physics=int(a['native_counts'].sum()),array_hashes={k:A.digest(v) for k,v in a.items()})
    N.BASE_INSPECT(p,a,row)
    corrupt=dict(row,array_hashes=dict(row['array_hashes'],actions='0'*64))
    try:N.BASE_INSPECT(p,a,corrupt)
    except AssertionError:pass
    else:raise AssertionError('Corrupt arrays accepted')
    seen=[];old=os.sched_setaffinity
    def witness(pipe,index,pc,counts):
        assert N.N.P.frozen is A.frozen and N.N.G.T.Session is N.N.C.Session;seen.append(index)
    try:
        with mock.patch.object(N.N.G.T,'worker',witness):N.worker(None,0,child,None)
    finally:os.sched_setaffinity=old
    assert seen==[0] and N.N.Pool.__init__.__globals__['worker'] is N.worker
    save(DOC/'NATIVE_PREFLIGHT.json',dict(state='PASS',actual_canonical_worker_loader_and_full_auditor=True,
        corrupted_action_hash_rejected=True,no_source_identity_filter_on_comparators=True,new_native=0,new_optimizer=0))

def register():
    assert not (OUT/'protocol.json').exists();assert read(DOC/'PREFLIGHT.json')['state']=='PASS'
    for unit in ['stage-t699-preparation','stage-t699-preparation-recovery']:
        props=subprocess.check_output(['systemctl','--user','show',unit+'.service','-p','ActiveState','-p','MainPID'],text=True)
        assert 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props)
    p=read(HERE/'protocol.json');native_preflight(p)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    extra=set(HERE.glob('*.py'))|{DOC/'NATIVE_PREFLIGHT.json',DOC/'PREPARATION_RECOVERY.json'}
    p['inputs'].update({str(f):sha(f) for f in extra})
    p['input_stats']={f:[Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f in p['inputs']}
    now=time.time();p.update(authorization_approved=True,registration_epoch=now,absolute_all_stop=min(now+p['root_seconds'],1795395600))
    minimum=p['fit_main_seconds']+p['fit_audit_seconds']+5*(p['known_main_seconds']+p['known_audit_seconds'])+7*(p['new_main_seconds']+p['new_audit_seconds'])+1260
    assert now+minimum<p['campaign_native_stop'],'No full overnight admission before original native stop'
    l=read(WEEKEND/'LEDGER.json');assert not any(j['id']=='T699' for j in l['jobs'])
    for key in ['Actor_live','Actor_shadow','Actor_independent']:assert l[key]+p['max_'+key]<=4096
    assert l['physics']+p['max_physics']<1500000000 and l['commands']+p['max_commands']<45000
    assert l['science_seconds']+p['fit_main_seconds']+5*p['known_main_seconds']+7*p['new_main_seconds']+1260<180000
    assert l['audit_seconds']+p['fit_audit_seconds']+5*p['known_audit_seconds']+7*p['new_audit_seconds']<18000
    l['jobs'].append(dict(id='T699',state='RESERVED',out=str(OUT),max_physics=p['max_physics'],max_commands=p['max_commands'],
        Actor_live_cap=3841,Actor_shadow_cap=3841,Actor_independent_cap=1537,completed_children=[],absolute_all_stop=p['absolute_all_stop']))
    save(WEEKEND/'LEDGER.json',l)
    blob=json.dumps(p,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=now,max_physics=p['max_physics'],max_commands=p['max_commands'],
        absolute_all_stop=p['absolute_all_stop'],input_count=len(p['inputs']),budget_is_agent_chosen_under_user_delegation=True))
    print('T699_REGISTERED_NOT_LAUNCHED')

def launch():
    p=read(OUT/'protocol.json');immutable(p,True);assert not (OUT/'launch.json').exists()
    r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert r['root_tree']==tree
    seconds=int(p['absolute_all_stop']-time.time()-3);assert seconds>0
    cmd=['systemd-run','--user','--unit=stage-t699','--collect','--property=Type=exec','--property=CPUQuota=800%',
        '--property=CPUWeight=10','--property=Nice=19','--property=AllowedCPUs=8 9 10 11 12 13 14 15',
        '--property=MemoryMax=8G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',
        f'--property=RuntimeMaxSec={seconds}',f'--property=WorkingDirectory={ROOT}',
        f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',
        '--setenv=OMP_NUM_THREADS=8','--setenv=MKL_NUM_THREADS=8',
        'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T699','--why=Bounded matched controls',
        str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'driver']
    c=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True);assert c.returncode==0,c.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit='stage-t699',source_remote=r['remote_commit'],root_tree=tree));print(c.stderr)

def publish_account(items,state):
    # One cumulative difference from this root's durable completed-action records, never add progress twice.
    total=dict(physics=sum(i.get('physics',0) for i in items),commands=sum(i.get('commands',0) for i in items),
        science_seconds=sum(i['elapsed_seconds'] for i in items if not i['audit']),
        audit_seconds=sum(i['elapsed_seconds'] for i in items if i['audit']),**count_calls())
    save(OUT/'LEDGER.json',dict(children=items,totals=total))
    l=read(WEEKEND/'LEDGER.json');j=next(j for j in l['jobs'] if j['id']=='T699');old=j.get('accounted',{})
    for key,v in total.items():
        assert v>=old.get(key,0);l[key]+=v-old.get(key,0)
    j.update(state=state,accounted=total,completed_children=[i['name'] for i in items])
    save(WEEKEND/'LEDGER.json',l)

def driver():
    p=read(OUT/'protocol.json');immutable(p,True);assert not (OUT/'DRIVER_STARTED.json').exists()
    os.sched_setaffinity(0,{15});save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time()))
    (OUT/'fit').mkdir();(OUT/'cost').mkdir()
    helper_log=(OUT/'helper.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'t699_common.py')],stdout=helper_log,stderr=helper_log)
    ids=[helper.pid];items=[];error=None;current=None
    def run(name,script,args,seconds,audit=False,child=None):
        nonlocal current
        immutable(p);assert helper.poll() is None
        cutoff=p['absolute_all_stop'] if audit or script=='t699_cost.py' else p['absolute_learning_stop'] if script=='t699_fit.py' else p['campaign_native_stop']
        assert time.time()+seconds+10<cutoff,(name,'original deadline admission')
        if child:
            folder=OUT/child['arm'];folder.mkdir();(folder/'episodes').mkdir()
            cp=dict(child,_root_sha256=sha(OUT/'protocol.json'),out=str(folder),child_bytes=p['child_bytes'],
                absolute_native_stop=min(time.time()+seconds,p['campaign_native_stop']),
                absolute_all_stop=min(time.time()+seconds+child['audit_seconds']+60,p['absolute_all_stop']))
            save(folder/'protocol.json',cp)
        with (OUT/(name+'.log')).open('xb') as log:
            c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/script),*args],cwd=ROOT,stdout=log,stderr=log,
                start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            ids.append(c.pid);current=c;start=time.time()
            save(OUT/'status.json',dict(state='ACTIVE',action=name,child_pid=c.pid,epoch=start,absolute_action_stop=start+seconds))
            try:
                c.wait(timeout=seconds);assert c.returncode==0,(name,c.returncode)
            except BaseException:
                signal_child(c.pid,signal.SIGCONT);signal_child(c.pid,signal.SIGTERM)
                try:c.wait(timeout=3)
                except subprocess.TimeoutExpired:signal_child(c.pid,signal.SIGKILL);c.wait()
                raise
            finally:
                item=dict(name=name,audit=audit,elapsed_seconds=time.time()-start,pid=c.pid,returncode=c.returncode)
                if child:
                    f=OUT/child['arm']/'summary.json';s=read(f) if f.exists() else {}
                    item.update(physics=s.get('physics',child['max_physics']),commands=s.get('commands',child['max_commands']))
                items.append(item);publish_account(items,'ACTIVE');current=None
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        run('fit','t699_fit.py',['main'],p['fit_main_seconds'])
        run('fit_audit','t699_fit.py',['audit'],p['fit_audit_seconds'],True)
        # Cost/actual command drift is observational and never chooses which final models receive native tests.
        for name in p['transfer_models']:
            run('cost_'+name,'t699_cost.py',[name],180)
        for name in p['final_models']:
            arm='known_'+name
            child=dict(arm=arm,model_name=name,suite='known',max_physics=sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['screen_jobs']),
                max_commands=514,main_seconds=p['known_main_seconds'],audit_seconds=p['known_audit_seconds'])
            run(arm,'t699_native.py',['evaluate',arm],p['known_main_seconds'],child=child)
            run(arm+'_audit','t699_native.py',['audit',arm],p['known_audit_seconds'],True)
        for name in p['transfer_models']:
            arm='new_'+name
            child=dict(arm=arm,model_name=name,suite='new',max_physics=sum(j['case']['step_cap'] for j in p['new_jobs']+p['new_exact_jobs']),
                max_commands=len(p['new_jobs'])+len(p['new_exact_jobs'])+14,main_seconds=p['new_main_seconds'],audit_seconds=p['new_audit_seconds'])
            run(arm,'t699_native.py',['evaluate',arm],p['new_main_seconds'],child=child)
            run(arm+'_audit','t699_native.py',['audit',arm],p['new_audit_seconds'],True)
        analyze(p)
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        (OUT/'HELPER_STOP').touch()
        try:helper.wait(timeout=5)
        except subprocess.TimeoutExpired:helper.terminate();helper.wait(timeout=3)
        helper_log.close()
        for f in OUT.glob('*/worker_generations.json'):
            for g in read(f):ids.extend([g['parent'],*g['workers']])
        alive=[pid for pid in set(ids) if Path('/proc',str(pid)).exists()]
        state='FAILED' if error or alive else 'COMPLETE';publish_account(items,state)
        report=dict(state=state,error=error,scientific_pids=sorted(set(ids)),all_scientific_processes_stopped=not alive,
            helper_pid=helper.pid,helper_stopped=helper.poll() is not None,retained_model_unchanged='T698',epoch=time.time())
        save(OUT/'SEALED.json',report);save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
        save(OUT/'status.json',dict(state=state,all_science_stopped=not alive,epoch=time.time()))

def analyze(p):
    rows=[]
    for name in p['transfer_models']:
        c=read(OUT/'cost'/f'{name}.json');new=read(OUT/('new_'+name)/'summary.json')
        assert read(OUT/('new_'+name)/'audit.json')['state']=='PASS'
        if name=='historical_parent':
            known=dict(core_success=390,success=403,lost403=[],source_success_ids=[],historical_reuse=True)
        elif name=='historical_A':
            known=dict(core_success=396,success=410,lost403=[],source_success_ids=p['source_ids'],historical_reuse=True)
        else:
            known=read(OUT/('known_'+name)/'summary.json');assert read(OUT/('known_'+name)/'audit.json')['state']=='PASS'
        rows.append(dict(name=name,cost=c,known=known,new=new))
    save(OUT/'RESULTS.json',dict(state='COMPLETE_PENDING_INTERPRETATION',rows=rows,retained_T698_unchanged=True,
        limited_same_geometry_local_transfer_NOT_unknown_map_generalization=True,no_novelty_or_top_journal_acceptance_claim=True))
    lines=['# T699 matched comparison','', '| Arm | Core407 | Old403 lost | New7 source | New starts | Added bytes | CPU p50 ms |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        k,c,n=r['known'],r['cost'],r['new']
        lines.append(f"| {r['name']} | {k['core_success']} | {len(k['lost403'])} | {len(k['source_success_ids'])} | {n['success']}/{n['total']} | {c['added_tensor_bytes']} | {c['batch1_median_ms']:.4f} |")
    lines+=['','Historical parent/A known-suite rows are reused by frozen model SHA; all new-start rows are fresh and audited. This bounded replay pilot and local test cannot establish family-wide superiority, optimal capacity, unknown-map generalization or top-journal novelty. Failures and all configurations remain available. T698 remains retained.']
    (DOC/'FINDINGS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    {'register':register,'launch':launch,'driver':driver}[sys.argv[1]]()

"""Exact reconstruction of14existing parent-success histories; no learning."""
from pathlib import Path
import sys,ast,contextlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t369_future_action_supervision'))
from t369_core import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t373_factual_parent_support';DOC=ROOT/'stage/docs/narrow_20261105'


def forbid_step(*_a,**_k):raise AssertionError('T373 forbids optimizer/gradient updates')


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes'] and K.S.used(OUT)<=p['final_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def prepare():
    assert not OUT.exists() and read(DOC/'T372_CLOSEOUT.json')['state']=='PASS_LOCAL_CLOSURE_WITH_TRANSPORT_RECOVERY'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T372.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(ROOT/'stage/runs/t371_terminal_head_repair/protocol.json'));rows=read(ROOT/'stage/runs/t371_terminal_head_repair/cases.json')
    jobs={j['id']:j for j in [*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs']]};selected=[]
    for r in rows:
        if r['role']=='initial' and r['success']:
            assert r['complete'] and not r['collision'] and r['policy_hash']==p['repair_parent']['policy_hash'] and not r['full_arrays_saved']
            selected.append(dict(source=r,job=jobs[r['job_id']]))
    assert len(selected)==14 and sum(v['job']['id'].startswith('wide') for v in selected)==3 and sum(v['job']['id'].startswith('dev6') for v in selected)==1
    p.update(id='T373',folder=OUT.name,out=str(OUT),experiment='T373_EXACT_FACTUAL_PARENT_SUCCESS_RECOVERY',selected=selected,
        wall_seconds=900,final_reserve_seconds=300,audit_seconds=300,all_work_seconds=1800,max_commands=29,max_physics=1120000,
        final_bytes=5*1024**2,full_bytes=256*1024,partial_bytes=512*1024,geometry_commands=1,fit_steps=0,max_live_Adam=0,max_shadow_Adam=0,
        preflight_discarded_Adam=0,audit_historical_discarded_Adam=0,new_RL_updates=0,new_native_training_collection=0,contract=(DOC/'T373_PLAN.md').read_text())
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T373_PLAN.md'),str(DOC/'T373_FACTUAL_PARENT_SUPPORT_DRAFT.md'),str(DOC/'T372_CLOSEOUT.json'),str(DOC/'T372_FINDINGS.md'),str(DOC/'T372_PROTECTION_SUPPORT.json'),str(ROOT/'stage/runs/t371_terminal_head_repair/cases.json'),str(ROOT/'stage/runs/t371_terminal_head_repair/episodes/initial_narrow070_dev_0.npz'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T372.json')}
    files|={v['source']['path'] for v in selected};p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    used=K.S.used(ROOT/'stage/runs');worst=14*p['full_bytes']+p['partial_bytes']+768*1024+128*1024
    assert worst<p['final_bytes'] and used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T373_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_max=p['final_bytes'],worst_payload=worst,global_full_reservation=used+p['final_bytes'],global_limit=p['global_bytes'],all_commands=29,max_physics=1120000,Adam=0))
    print(json.dumps(dict(state='REGISTERED',selected=14,native=0,Adam=0,full_global_reservation=used+p['final_bytes'])))


def check_forward(q,a,r):
    q.reset_noise();q.sample=False;h=None;assert np.array_equal(a['states'][1:],a['next_states'][:-1])
    assert {k:digest(v) for k,v in a.items()}==r['array_hashes']
    with torch.no_grad():
        for k,x in enumerate(a['states']):
            before=np.zeros(16) if h is None else h[0,0].numpy();assert np.array_equal(before,a['hidden_before'][k])
            action,_,h,_=q.step(torch.tensor(x[None]),h);assert np.array_equal(action[0].numpy(),a['actions'][k])
            for name,v in q.tape[-1].items():assert np.array_equal(v,a[name][k])
            assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);disc=K.R.GAMMA**n
            d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1]
            reward=-.000002*(1-disc)/(1-K.R.GAMMA)+.03*d0-disc*.03*d1*(not(c or g or t))
            if c or g or t:reward+=K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t))
            assert abs(reward-a['rewards'][k])<1e-12
    assert np.array_equal(h[0,0].numpy(),a['last_hidden_after']) and K.R.arrays_hash(a['final_native_integration'])==r['integration_sha256']


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.optim.Adam.step=forbid_step
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    q=NightActor(p,0.).eval().requires_grad_(False);q.load_learned(read_model(p['repair_parent']['path']));assert q.phash()==p['repair_parent']['policy_hash']
    for v in p['selected']:
        r=v['source'];a=arrays(r['path']);assert K.C.sha(r['path'])==r['sha256']
        assert sum(a['native_counts'].astype(int))==r['physics_steps'] and not a['end_flags'][:-1].any() and a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
    centre=next(r for r in read(ROOT/'stage/runs/t371_terminal_head_repair/cases.json') if r['role']=='initial' and r['job_id']=='narrow070_dev_0')
    check_forward(q,arrays(centre['path']),centre)
    folder=OUT/'preflight';folder.mkdir()
    with (folder/'factory.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        module,native,create,w,_=P.C.B.setup(folder/'native',p);job=next(v['job'] for v in p['selected'] if v['job']['id'].startswith('wide'))
        module.ACTIVE['circle_shift_x_m']=job.get('circle_shift_x_m',read(p['course_protocol'])['circle_shift_x_m']);env,obs,_=K.R.task_env(create,job)
        try:
            obs=K.C.perturb_anchor(env,obs,job);g=P.C.B.geometry(env,obs,w,job['case']['step_cap']);assert g['valid'] and env.step_count==0 and env.data.time==0. and env.max_wheel==5.
        finally:w.events.pop(id(env),None);env.close();w.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,Adam=0,gradient_calls=0,geometry_commands=1,frozen_parent_model_policy_exact=True,known_full_centre_forward_exact=True,selected_compact_hashes_exact=14,geometry=g))
    print(json.dumps(dict(state='PREFLIGHT_PASS',native=0,Adam=0,geometry=1)))


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=forbid_step
    start=time.time();deadline=read(OUT/'launch.json')['deadline_epoch'];ctx=mp.get_context('spawn');counts=ctx.Array('q',2);processes=[];pipes=[];rows=[];replays=[];items=[];commands=1;phase='STARTING';active=[];last=0.;state='FAILED'
    q=NightActor(p,0.).eval().requires_grad_(False);q.load_learned(read_model(p['repair_parent']['path']));initial=digest(q.learned())
    meter=P.C.ReservedBudget(ROOT/'stage/runs',OUT,p['final_bytes'],p['global_bytes'],p['min_free_bytes'])
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();meter.check();assert time.time()<deadline-5
        save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,commands=commands,physics_steps=sum(counts[:]),Adam=0,active=active))
    def receive(pipe):
        while not pipe.poll(.25):status()
        return pipe.recv()
    def jobs(role,reference=None):
        nonlocal commands,phase,active
        phase=role
        for offset in range(0,14,2):
            group=p['selected'][offset:offset+2];assert commands+len(group)<=29 and sum(counts[:])+sum(v['job']['case']['step_cap'] for v in group)<=p['max_physics']
            if role=='reconstruct':assert time.time()+60<deadline-p['final_reserve_seconds']
            assert time.time()+30<deadline;commands+=len(group);active=[v['job']['id'] for v in group];status(True)
            for pipe,v in zip(pipes,group):
                j=v['job'];pipe.send(dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='memory',rho=0.,learned=q.learned(),policy_hash=q.phash(),deadline_epoch=deadline-15,readback_model=p['repair_parent']['path'] if reference is not None else None,stochastic=False))
            batch=[receive(pipe) for pipe in pipes[:len(group)]];active=[]
            for v,item in zip(group,batch):
                assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays']
                if not r['complete']:
                    buf=io.BytesIO();np.savez_compressed(buf,**packed(a));blob=buf.getvalue();assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob);save(OUT/'partial_case.json',r);raise RuntimeError('Partial evidence preserved; no retry')
                source=v['source']
                exact=r['job_id']==source['job_id'] and r['success'] and not r['collision'] and not r['timeout'] and r['array_hashes']==source['array_hashes'] and r['integration_sha256']==source['integration_sha256']
                if not exact:
                    buf=io.BytesIO();np.savez_compressed(buf,**packed(a));blob=buf.getvalue();assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob);save(OUT/'partial_case.json',r);raise RuntimeError('Historical comparison mismatch; actual raw preserved, no retry')
                r.update(historical_success_reconstruction=True,new_independent_success=False,source_role=source['role'],source_id=source['id'])
                if reference is None:
                    buf=io.BytesIO();np.savez_compressed(buf,**packed(a));blob=buf.getvalue()
                    if len(blob)>p['full_bytes']:
                        assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob);save(OUT/'partial_case.json',r);raise RuntimeError('Full record exceeds256KiB; partial512KiB preserved, no budget expansion')
                    path=OUT/'episodes'/(r['job_id']+'.npz');assert not path.exists();path.write_bytes(blob)
                    r.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=True);rows.append(r);items.append(item);save(OUT/'cases.json',rows)
                else:
                    ref=reference[r['job_id']]
                    if not(a.keys()==ref['arrays'].keys() and all(np.array_equal(val,ref['arrays'][k]) for k,val in a.items())):
                        buf=io.BytesIO();np.savez_compressed(buf,**packed(a));blob=buf.getvalue();assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob);save(OUT/'partial_case.json',r);raise RuntimeError('Readback array mismatch preserved')
                    r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);replays.append(r);save(OUT/'replays.json',replays)
            status(True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T373 native deadline')));signal.alarm(max(1,int(deadline-time.time())-3))
    try:
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=night_worker,args=(b,k,p,counts));proc.start();b.close();processes.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[v.pid for v in processes],ready=ready))
        jobs('reconstruct');jobs('model_readback',{v['row']['job_id']:v for v in items});assert digest(q.learned())==initial;state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics_steps=sum(counts[:]),active=active))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError):pass
        for proc in processes:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(counts[:]),Adam=0,gradient_calls=0,primary_cases=len(rows),native_model_readbacks=len(replays),all_workers_stopped=all(not v.is_alive() for v in processes),frozen_parent_policy_hash=q.phash(),new_independent_successes=0,new_RL_updates=0,new_native_training_collection=0,research_candidate_not_promoted=True,old275=None,six54=None,fresh16=None,reload78=None,stable_original_six=False,entered7or8=False)
    save(OUT/'summary.json',s);save(OUT/'status.json',s);meter.check(force=True);meter.close();print(json.dumps(s),flush=True)


def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T373 audit300s')));signal.alarm(300)
    torch.set_num_threads(1);torch.optim.Adam.step=forbid_step;q=NightActor(p,0.).eval().requires_grad_(False);q.load_learned(read_model(p['repair_parent']['path']));assert q.phash()==p['repair_parent']['policy_hash']
    rows=read(OUT/'cases.json');replays=read(OUT/'replays.json');assert len(rows)==len(replays)==14;physics=0
    for r,v in zip(rows,p['selected']):
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);assert r['job_id']==v['source']['job_id'] and r['array_hashes']==v['source']['array_hashes']
        check_forward(q,a,r);assert r['success'] and not r['collision'] and not r['timeout'] and r['complete'] and not r['administrative_censor']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1,1] and not a['end_flags'][:,0].any();assert int(a['native_counts'].astype(int).sum())==r['physics_steps'];physics+=r['physics_steps']
    for r,ref in zip(replays,rows):
        assert r['job_id']==ref['job_id'] and r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256'] and r['all_arrays_hidden_history_integration_exact'] and not r['independent_sample'];physics+=r['physics_steps']
    assert physics==s['physics_steps']<=p['max_physics'] and s['commands']==29 and s['Adam']==s['gradient_calls']==s['new_independent_successes']==0 and s['elapsed_s']<=900
    assert not (OUT/'partial.npz').exists() and not (OUT/'failure.json').exists()
    result=dict(state='PASS',new_native=0,Adam=0,gradient_calls=0,all_fourteen_original_source_and_full_history_exact=True,model_readbacks_exact=14,new_independent_successes=0,commands=29,physics_steps=physics,policy_hash=q.phash(),source_model_sha256=K.C.sha(p['repair_parent']['path']))
    save(OUT/'audit.json',result);save(DOC/'T373_AUDIT.json',result);signal.alarm(0);print(json.dumps(result))


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();epoch=time.time()
    with (DOC/'T373_DRIVER.log').open('xb') as log:proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage factual policy history recovery','timeout','--signal=TERM','--kill-after=1s','1800s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,deadline_epoch=epoch+900,all_work_deadline_epoch=epoch+1800,source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')));print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


def driver():
    p=read(HERE/'protocol.json');verify(p);launch=read(OUT/'launch.json');active=None
    def call(args,seconds,log):
        nonlocal active
        with log.open('xb') as f:
            active=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),*args],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc=active.wait(timeout=seconds)
                if rc:raise subprocess.CalledProcessError(rc,str(args))
            except BaseException:
                try:os.killpg(active.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                if active.poll() is None:
                    try:active.wait(1)
                    except subprocess.TimeoutExpired:pass
                try:os.killpg(active.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                active.wait();raise
            finally:active=None
    def stop(*_):raise TimeoutError('T373 bounded finalizer interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(launch['all_work_deadline_epoch']-time.time())-3))
    try:
        save(DOC/'T373_DRIVER.json',dict(pid=os.getpid(),state='NATIVE'))
        call([str(HERE/'run.py'),'main'],min(900,launch['deadline_epoch']-time.time()),OUT/'main.log');s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
        for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
        save(DOC/'T373_DRIVER.json',dict(pid=os.getpid(),state='AUDIT'));call([str(HERE/'run.py'),'audit'],min(300,launch['all_work_deadline_epoch']-time.time()-30),OUT/'audit_console.log')
        assert read(OUT/'audit.json')['state']=='PASS';save(DOC/'T373_DRIVER.json',dict(pid=os.getpid(),state='RAW_BACKUP'))
        call([str(HERE/'transport.py'),'T373'],min(600,launch['all_work_deadline_epoch']-time.time()-20),DOC/'T373_BACKUP.log')
        assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T373.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        save(DOC/'T373_DRIVER.json',dict(pid=os.getpid(),state='COMPLETE_AUDIT_AND_RAW_BACKUP',summary_sha256=K.C.sha(OUT/'summary.json')))
    except BaseException:save(DOC/'T373_DRIVER_ERROR.json',dict(error=traceback.format_exc(),navigation_restarted=False));raise
    finally:signal.alarm(0)


if __name__=='__main__':globals()[sys.argv[1]]()

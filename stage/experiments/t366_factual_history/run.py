"""Exact factual history recovery, no learning or native search."""
from pathlib import Path
import sys,contextlib,ast
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t365_correction_aggregation'))
from t365_core import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t366_factual_history';DOC=ROOT/'stage/docs/narrow_20261105'


def forbid_step(*_args,**_kwargs):
    raise AssertionError('T366 forbids every optimizer step')


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def prepare():
    assert not OUT.exists()
    assert read(DOC/'T365_CLOSEOUT.json')['state']=='LOCAL_FINAL_EVIDENCE_AND_PROCESS_CLOSURE_VERIFIED'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T365.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(ROOT/'stage/runs/t365_correction_aggregation/protocol.json'));rows=read(ROOT/'stage/runs/t365_correction_aggregation/cases.json')
    jobs={j['id']:j for j in p['course_dev_jobs']+p['narrow_jobs']};selected=[]
    for role,ids,model in [('initial',[f'wide_dev_{k}' for k in range(5)],p['initial_model']),
                           ('final',['narrow070_dev_2','narrow070_dev_3'],read(ROOT/'stage/runs/t365_correction_aggregation/model.json')['path'])]:
        for name in ids:
            r=next(v for v in rows if v['role']==role and v['job_id']==name);assert r['success'] and r['complete'] and not r['full_arrays_saved']
            selected.append(dict(source=r,job=jobs[name],model=model,model_sha256=K.C.sha(model)))
    p.update(experiment='T366_FIXED_FACTUAL_SUCCESS_HISTORY_RECOVERY',out=str(OUT),selected=selected,wall_seconds=900,audit_seconds=300,all_work_seconds=1800,
        final_reserve_seconds=240,max_commands=16,max_physics=560000,final_bytes=6*1024**2,full_bytes=384*1024,partial_bytes=384*1024,
        contract=(DOC/'T366_PLAN.md').read_text(),max_live_Adam=0,max_shadow_Adam=0,preflight_discarded_Adam=0,audit_historical_discarded_Adam=0)
    files=set(p['input_hashes'])|{str(HERE/'run.py'),str(DOC/'T366_PLAN.md'),str(DOC/'T365_FINDINGS.md'),str(DOC/'T365_CLOSEOUT.json')}
    for v in selected:files|={v['source']['path'],v['model']}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)};used=K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T366_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_max=p['final_bytes'],global_full_reservation=used+p['final_bytes'],global_limit=p['global_bytes'],commands=16,max_physics=560000,Adam=0))


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.optim.Adam.step=forbid_step
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    folder=OUT/'preflight';folder.mkdir();geometry=[]
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        module,native,create,w,_=P.C.B.setup(folder/'native',p)
        try:
            for v in [p['selected'][0],p['selected'][5]]:
                j=v['job'];module.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',read(p['course_protocol'])['circle_shift_x_m']);env,obs,_=K.R.task_env(create,j)
                try:
                    obs=K.C.perturb_anchor(env,obs,j);g=P.C.B.geometry(env,obs,w,40000);assert g['valid'] and env.step_count==0 and env.data.time==0.;geometry.append(g)
                finally:w.events.pop(id(env),None);env.close()
        finally:w.close()
    q=NightActor(p,0.).requires_grad_(False)
    for v in p['selected']:
        q.load_learned(read_model(v['model']));assert q.phash()==v['source']['policy_hash']
        a=arrays(v['source']['path']);assert int(a['native_counts'].astype(int).sum())==v['source']['physics_steps']
    save(OUT/'preflight.json',dict(state='PASS',native=0,Adam=0,geometry_commands=2,frozen_model_policy_hashes_exact=True,geometry=geometry))


def diagnostics(p,items):
    centre=arrays(next(v['path'] for v in read(ROOT/'stage/runs/t365_correction_aggregation/cases.json') if v['role']=='final' and v['job_id']=='narrow070_dev_0'))
    q=NightActor(p,0.);q.load_learned(read_model(p['selected'][5]['model']));errors=[];divergence=[]
    with torch.no_grad():
        for it in items:
            a=it['arrays'];name=it['row']['job_id']
            if name.startswith('wide'):
                x=torch.tensor(a['states']);pred=q.sequence(x,q.base_actions(x));target=torch.tensor(a['locations']);errors.append(dict(id=name,rmse=(pred-target).square().mean(0).sqrt().tolist()))
            else:
                n=min(len(centre['states']),len(a['states']));c=centre['poses'][:n,:2]-centre['poses'][0,:2];b=a['poses'][:n,:2]-a['poses'][0,:2]
                xy=np.linalg.norm(c-b,axis=1);act=np.max(np.abs(centre['actions'][:n]-a['actions'][:n]),axis=1)
                first=lambda x:None if not np.any(x) else float(np.flatnonzero(x)[0]*.2)
                divergence.append(dict(id=name,first_translation_removed_position_gap_gt10cm_s=first(xy>.1),first_action_gap_gt_point1_s=first(act>.1),max_aligned_position_gap_m=float(xy.max())))
        # These are the same two historical support states used in T365, with T353 location targets.
        prior=NightActor(p,0.);_,_,supports=PREVIOUS.references(p,prior)
        older=[(q.sequence(g['x'],g['b'])-g['y']).square().mean(0).sqrt().tolist() for g in supports]
    input_pairs=[]
    for wide in items[:5]:
        for narrow in items[5:]:
            a,b=wide['arrays'],narrow['arrays'];n=min(len(a['states']),len(b['states']));ha=hashlib.sha256();hb=hashlib.sha256();conflicts=[]
            for k in range(n):
                ha.update(a['states'][k].tobytes());hb.update(b['states'][k].tobytes())
                if ha.digest()==hb.digest() and not np.array_equal(a['locations'][k],b['locations'][k]):conflicts.append(k)
            dif=np.abs(a['states'][0]-b['states'][0]);input_pairs.append(dict(wide=wide['row']['job_id'],narrow=narrow['row']['job_id'],initial120_equal=bool(np.array_equal(a['states'][0],b['states'][0])),initial120_max_difference=float(dif.max()),different_initial_columns=np.flatnonzero(dif).tolist(),exact_full_observation_prefix_different_target_steps=conflicts))
    return dict(initial_and_full_observation_prefix_comparisons=input_pairs,actual_five_wide_history_T365_location_rmse=errors,older_two_support_T365_location_rmse=older,
        centre_vs_successful_neighbours=divergence,descriptive_not_causal=True,synthetic_expert_labels=0,new_native=0,new_Adam=0)


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=forbid_step
    start=time.time();deadline=read(OUT/'launch.json')['deadline_epoch'];ctx=mp.get_context('spawn');counts=ctx.Array('q',2);procs=[];pipes=[];rows=[];reps=[];commands=2;phase='STARTING';state='FAILED';last=0.
    meter=P.C.ReservedBudget(ROOT/'stage/runs',OUT,p['final_bytes'],p['global_bytes'],p['min_free_bytes'])
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();meter.check();assert time.time()<deadline-5
        save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,commands=commands,physics_steps=sum(counts[:]),Adam=0,deadline_epoch=deadline))
    def receive(pipe):
        while not pipe.poll(.25):status()
        return pipe.recv()
    def run(role,refs=None):
        nonlocal phase,commands
        phase=role;result=[]
        for off in range(0,7,2):
            group=p['selected'][off:off+2];commands+=len(group);assert commands<=16 and sum(counts[:])+40000*len(group)<=p['max_physics'];status(True)
            for pipe,v in zip(pipes,group):
                q=NightActor(p,0.);q.load_learned(read_model(v['model']));j=v['job'];pipe.send(dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='memory',rho=0.,learned=q.learned(),policy_hash=q.phash(),deadline_epoch=deadline-10,stochastic=False,readback_model=v['model']))
            batch=[receive(pipe) for pipe in pipes[:len(group)]]
            for it,v in zip(batch,group):
                assert 'error' not in it,it.get('error');r,a=it['row'],it['arrays']
                if not(r['complete'] and r['success'] and r['array_hashes']==v['source']['array_hashes'] and r['integration_sha256']==v['source']['integration_sha256']):
                    blob=io.BytesIO();np.savez_compressed(blob,**packed(a));b=blob.getvalue();assert len(b)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(b);save(OUT/'partial_case.json',r);raise AssertionError('Factual reconstruction mismatch preserved; no retry')
                saved=arrays(v['source']['path']);assert all(np.array_equal(saved[k],a[k]) for k in ['actions','native_counts','end_flags'])
                if refs is None:
                    blob=io.BytesIO();np.savez_compressed(blob,**packed(a));b=blob.getvalue();assert len(b)<=p['full_bytes'];path=OUT/'episodes'/(r['job_id']+'.npz');path.write_bytes(b)
                    r.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=True,historical_reconstruction_not_new_independent_success=True,source_model=v['model'],source_model_sha256=v['model_sha256'])
                    rows.append(r);save(OUT/'cases.json',rows)
                else:
                    ref=refs[r['job_id']];assert a.keys()==ref['arrays'].keys() and all(np.array_equal(a[k],ref['arrays'][k]) for k in a)
                    r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);reps.append(r);save(OUT/'replays.json',reps)
                result.append(it)
            status(True)
        return result
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T366 absolute native deadline')));signal.alarm(max(1,int(deadline-time.time())-3))
    try:
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);pr=ctx.Process(target=night_worker,args=(b,k,p,counts));pr.start();b.close();procs.append(pr);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] for v in ready);save(OUT/'workers.json',dict(parent=os.getpid(),workers=[pr.pid for pr in procs],ready=ready))
        items=run('factual_recovery');assert time.time()<deadline-240;save(OUT/'diagnostics.json',diagnostics(p,items))
        run('saved_model_readback',{v['row']['job_id']:v for v in items});state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics_steps=sum(counts[:])))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError):pass
        for pr in procs:
            pr.join(3)
            if pr.is_alive():pr.kill();pr.join(3)
        signal.alarm(0)
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(counts[:]),Adam=0,full_historical_episodes=len(rows),exact_model_readbacks=len(reps),new_independent_navigation_samples=0,all_workers_stopped=all(not pr.is_alive() for pr in procs),six_gate=False)
    save(OUT/'summary.json',s);save(OUT/'status.json',s);meter.check(force=True);meter.close()


def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    torch.set_num_threads(1);torch.optim.Adam.step=forbid_step;items=[];physics=0
    for r,v in zip(read(OUT/'cases.json'),p['selected']):
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);assert {k:digest(x) for k,x in a.items()}==r['array_hashes']
        assert r['array_hashes']==v['source']['array_hashes'] and K.R.arrays_hash(a['final_native_integration'])==r['integration_sha256'];q=NightActor(p,0.);q.load_learned(read_model(v['model']));q.reset_noise();h=None
        with torch.no_grad():
            for k,x in enumerate(a['states']):
                before=np.zeros(16) if h is None else h[0,0].numpy();assert np.array_equal(before,a['hidden_before'][k]);action,_,h,_=q.step(torch.tensor(x[None]),h)
                assert np.array_equal(action[0].numpy(),a['actions'][k])
                for key,val in q.tape[-1].items():assert np.array_equal(val,a[key][k])
                assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
                nn=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);disc=K.R.GAMMA**nn;d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1]
                rw=-.000002*(1-disc)/(1-K.R.GAMMA)+.03*d0-disc*.03*d1*(not(c or g or t))
                if c or g or t:rw+=K.R.GAMMA**(nn-1)*(float(g)-float(c)-.5*float(t))
                assert abs(rw-a['rewards'][k])<1e-12
        assert np.array_equal(h[0,0].numpy(),a['last_hidden_after'])
        assert int(a['native_counts'].astype(int).sum())==r['physics_steps'];physics+=r['physics_steps'];items.append(dict(row=r,arrays=a))
    reps=read(OUT/'replays.json');assert len(reps)==7
    for rep,r in zip(reps,read(OUT/'cases.json')):assert rep['array_hashes']==r['array_hashes'] and rep['all_arrays_hidden_history_integration_exact'];physics+=rep['physics_steps']
    assert physics==s['physics_steps']<=p['max_physics'] and s['commands']==16 and s['Adam']==0 and s['elapsed_s']<=900
    assert diagnostics(p,items)==read(OUT/'diagnostics.json')
    result=dict(state='PASS',all_seven_source_policy_and_full_history_hashes_exact=True,zero_new_independent_successes=True,commands=16,physics_steps=physics,new_native=0,new_Adam=0,diagnostics=read(OUT/'diagnostics.json'))
    save(OUT/'audit.json',result);save(DOC/'T366_AUDIT.json',result)


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();epoch=time.time()
    with (DOC/'T366_DRIVER.log').open('xb') as log:proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage factual policy history recovery','timeout','--signal=TERM','--kill-after=1s','1800s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
    def stop(*_):raise TimeoutError('T366 bounded finalizer interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(launch['all_work_deadline_epoch']-time.time())-3))
    try:
        save(DOC/'T366_DRIVER.json',dict(pid=os.getpid(),state='NATIVE'))
        call([str(HERE/'run.py'),'main'],min(900,launch['deadline_epoch']-time.time()),OUT/'main.log');s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
        for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
        save(DOC/'T366_DRIVER.json',dict(pid=os.getpid(),state='AUDIT'));call([str(HERE/'run.py'),'audit'],min(300,launch['all_work_deadline_epoch']-time.time()-30),OUT/'audit_console.log')
        assert read(OUT/'audit.json')['state']=='PASS';save(DOC/'T366_DRIVER.json',dict(pid=os.getpid(),state='RAW_BACKUP'))
        call([str(ROOT/'stage/experiments/t260_distance_archive/small_parts.py'),'T366'],min(600,launch['all_work_deadline_epoch']-time.time()-20),DOC/'T366_BACKUP.log')
        assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T366.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        save(DOC/'T366_DRIVER.json',dict(pid=os.getpid(),state='COMPLETE_AUDIT_AND_RAW_BACKUP',summary_sha256=K.C.sha(OUT/'summary.json')))
    except BaseException:save(DOC/'T366_DRIVER_ERROR.json',dict(error=traceback.format_exc(),navigation_restarted=False));raise
    finally:signal.alarm(0)


if __name__=='__main__':globals()[sys.argv[1]]()

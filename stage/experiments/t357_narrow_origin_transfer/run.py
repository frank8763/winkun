"""Frozen recurrent navigation in an independently passable narrow scene."""
from pathlib import Path
import sys,os,copy,json,time,io,signal,subprocess,multiprocessing as mp,traceback,contextlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t353_offline_night_chain'))
from night_core import *
import night_core as NC
import mujoco as mj
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t357_narrow_origin_transfer';DOC=ROOT/'stage/docs/narrow_20261105'


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def prepare():
    assert not OUT.exists();p=copy.deepcopy(read(ROOT/'stage/runs/t353_temporal_exploration/protocol.json'))
    s=read(ROOT/'stage/runs/t353_temporal_exploration/summary.json');m=s['models']['iid']
    assert s['selected']=='iid' and K.C.sha(m['path'])==m['sha256']
    assert read(DOC/'T356_AUDIT.json')['local070_all3_pass']
    shift=read(ROOT/'stage/runs/t356_narrow_passage_contract/protocol.json')['shift_x_m'];jobs=[]
    for j in p['course_dev_jobs']:
        v=copy.deepcopy(j);v.update(id=j['id'].replace('wide_dev','narrow070_dev'),source_reset_id=j['id'].replace('wide_dev','narrow070_dev'),
            circle_shift_x_m=shift,expected_initial120_sha256=None,diagnostic_group='new070_course80');jobs.append(v)
    v=copy.deepcopy(p['course_dev_jobs'][0]);v.update(circle_shift_x_m=.6,diagnostic_group='historical117_course80');jobs.append(v)
    for j in p['original_jobs']:
        v=copy.deepcopy(j);v.update(diagnostic_group='original057_benchmark56');jobs.append(v)
    p.update(experiment='T357_FROZEN_ORIGINAL_START_NARROW_TRANSFER',out=str(OUT),initial_model=m['path'],jobs=jobs,
        wall_seconds=1200,audit_seconds=300,all_work_seconds=1800,final_bytes=8*1024**2,full_bytes=768*1024,partial_bytes=768*1024,
        geometry_commands=3,max_commands=19,max_physics=640000,Actor_updates=0,optimizer_calls=0,
        new_scene_shift_x_m=shift,original_gate_not_consumed=True,
        contract=(DOC/'T357_PLAN.md').read_text())
    inputs=set(p['input_hashes'])|{str(HERE/'run.py'),str(DOC/'T357_PLAN.md'),str(DOC/'T356_AUDIT.json'),str(DOC/'T356_FINDINGS.md'),
        str(ROOT/'stage/experiments/t353_offline_night_chain/night_core.py'),str(ROOT/'stage/runs/t353_temporal_exploration/summary.json'),m['path']}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(inputs)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes'];OUT.mkdir();(OUT/'episodes').mkdir()
    save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);print(json.dumps(dict(state='REGISTERED',native=0,Adam=0)))


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1)
    torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No T357 optimizer'))
    q=NightActor(p,0.).eval().requires_grad_(False);assert digest(q.learned())==read(ROOT/'stage/runs/t353_temporal_exploration/summary.json')['models']['iid']['hashes']['learned']
    geos=[];folder=OUT/'preflight';folder.mkdir()
    with (folder/'factory.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        module,native,create,w,_=P.C.B.setup(folder/'native',p);original=copy.deepcopy(module.ACTIVE)
        try:
            for j,gap in zip([p['jobs'][0],p['jobs'][5],p['jobs'][6]],[.70,1.17,.57]):
                module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m'])
                env,obs,_=K.R.task_env(create,j)
                try:
                    obs=K.C.perturb_anchor(env,obs,j);g=P.C.B.geometry(env,obs,w,j['case']['step_cap'])
                    c=mj.mj_name2id(env.model,mj.mjtObj.mjOBJ_GEOM,'real_circle_obs_9');wall=mj.mj_name2id(env.model,mj.mjtObj.mjOBJ_GEOM,'real_rect_obs_1')
                    actual=float(env.model.geom_pos[c,0]-env.model.geom_size[c,0]-env.model.geom_pos[wall,0]-env.model.geom_size[wall,0])
                    assert abs(actual-gap)<1e-6 and g['valid'] and env.data.time==0. and env.step_count==0
                    assert env.max_wheel==5. and np.array_equal(env.model.actuator_forcerange[[env.id_lr_wheel,env.id_rr_wheel]],np.array([[-120.,120.],[-120.,120.]]))
                    active=[k for k in range(env.model.ngeom) if int(env.model.geom_bodyid[k]) in env.robot_body_ids and env.model.geom_contype[k]]
                    assert sum(int(env.model.geom_type[k])==int(mj.mjtGeom.mjGEOM_CYLINDER) for k in active)>=4
                    x=torch.tensor([g['full120']],dtype=torch.float32)
                    with torch.no_grad():a,loc,h,_=q.step(x);seq=q.sequence(x,q.base_actions(x))
                    assert torch.equal(loc,seq) and all(v.grad is None for v in q.parameters())
                    geos.append(dict(id=j['id'],gap_m=actual,geometry=g,active_robot_colliders=active,location_sha256=digest(loc)))
                finally:w.events.pop(id(env),None);env.close()
        finally:w.close()
    save(OUT/'preflight.json',dict(state='PASS',geometry_commands=3,physics_steps=0,optimizer_calls=0,policy_hash=q.phash(),geometries=geos,
        full_reservation_bytes=p['final_bytes'],frozen_original_source_model=True))
    print(json.dumps(dict(state='PREFLIGHT_PASS',geometry=3,native=0,Adam=0)))


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (OUT/'main.log').open('xb') as f:proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage narrow original-start transfer',
        'timeout','--signal=TERM','--kill-after=1s','1200s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'main'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,deadline_epoch=epoch+1200,all_work_deadline_epoch=epoch+1800,
        source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No T357 optimizer'))
    start=time.time();deadline=read(OUT/'launch.json')['deadline_epoch'];ctx=mp.get_context('spawn');counts=ctx.Array('q',2)
    meter=P.C.ReservedBudget(ROOT/'stage/runs',OUT,p['final_bytes'],p['global_bytes'],p['min_free_bytes'])
    q=NightActor(p,0.).eval().requires_grad_(False);qhash=q.phash();processes=[];pipes=[];rows=[];reps=[];commands=3;active=[];last=0.;phase='STARTING';state='FAILED'
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();meter.check();assert time.time()<deadline-5
        save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,main_epoch=start,commands=commands,physics_steps=sum(counts[:]),active=active))
    def receive(pipe):
        while not pipe.poll(.25):status()
        return pipe.recv()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T357 deadline')));signal.alarm(max(1,int(deadline-time.time())-3))
    try:
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=night_worker,args=(b,k,p,counts));proc.start();b.close();processes.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[v.pid for v in processes],ready=ready))
        for offset in range(0,len(p['jobs']),2):
            group=p['jobs'][offset:offset+2];original=[]
            for replica in [False,True]:
                phase='EXACT_REPLAY' if replica else 'FROZEN_NAVIGATION';active=[j['id'] for j in group]
                assert commands+len(group)<=p['max_commands'] and sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
                assert time.time()+60<deadline;commands+=len(group);status(True)
                for pipe,j in zip(pipes,group):pipe.send(dict(j,id=('replay_' if replica else 'primary_')+j['id'],source_reset_id=j['id'],role=phase,arm='memory',rho=0.,
                    learned=q.learned(),policy_hash=qhash,deadline_epoch=deadline-15,readback_model=p['initial_model'] if replica else None,stochastic=False))
                items=[receive(pipe) for pipe in pipes[:len(group)]];active=[]
                for j,item in zip(group,items):
                    assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays'];assert r['policy_hash']==qhash
                    if not r['complete']:
                        b=io.BytesIO();np.savez_compressed(b,**packed(a));blob=b.getvalue();assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob)
                        save(OUT/'partial_case.json',r);raise RuntimeError('Partial preserved; no retry')
                    r['diagnostic_group']=j['diagnostic_group']
                    if replica:
                        ref=next(v for v in original if v['row']['job_id']==r['job_id'])
                        assert r['array_hashes']==ref['row']['array_hashes'] and r['integration_sha256']==ref['row']['integration_sha256']
                        assert a.keys()==ref['arrays'].keys() and all(np.array_equal(v,ref['arrays'][k]) for k,v in a.items())
                        r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);reps.append(r);save(OUT/'replays.json',reps)
                    else:
                        blob=io.BytesIO();np.savez_compressed(blob,**packed(a));data=blob.getvalue();assert len(data)<=p['full_bytes']
                        path=OUT/'episodes'/(j['id']+'.npz');assert not path.exists();path.write_bytes(data)
                        r.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=True);rows.append(r);original.append(item);save(OUT/'cases.json',rows)
                status(True)
        state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),active=active,phase=phase,commands=commands,physics_steps=sum(counts[:])))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError):pass
        for proc in processes:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
    result={}
    for name in ['new070_course80','historical117_course80','original057_benchmark56']:
        rr=[v for v in rows if v['diagnostic_group']==name];result[name]={k:sum(bool(v[k]) for v in rr) for k in ['success','collision','timeout','administrative_censor']};result[name]['executed']=len(rr)
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(counts[:]),result=result,
        primary_cases=len(rows),exact_replays=len(reps),Actor_updates=0,optimizer_calls=0,policy_hash=qhash,input_model_sha256=K.C.sha(p['initial_model']),
        all_workers_stopped=all(not v.is_alive() for v in processes),old275=None,six54=None,fresh16=None,reload78=None,stable_six=False,entered7or8=False,budget=meter.stats())
    save(OUT/'summary.json',s);save(OUT/'status.json',s);meter.check(force=True);meter.close();print(json.dumps(s),flush=True)


def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T357_AUDIT.json';assert not dest.exists() and s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T357 audit300s')));signal.alarm(300)
    torch.set_num_threads(1);torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No T357 audit optimizer'))
    rows=read(OUT/'cases.json');reps=read(OUT/'replays.json');assert len(rows)==len(reps)==8;physics=0
    q=NightActor(p,0.).eval().requires_grad_(False);assert q.phash()==s['policy_hash'];hist=read(ROOT/'stage/runs/t353_temporal_exploration/summary.json');history=[]
    oldrows=[json.loads(v) for v in (ROOT/'stage/runs/t353_temporal_exploration/cases.jsonl').read_text().splitlines()]
    for r,rep in zip(rows,reps):
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);assert {k:digest(v) for k,v in a.items()}==r['array_hashes']
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'];assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        c,g,t=map(bool,a['end_flags'][-1]);assert c==r['collision'] and (g and not c)==r['success'] and t==r['timeout'];assert not r['administrative_censor']
        q.reset_noise();q.sample=False;h=None;assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        for k,x in enumerate(a['states']):
            bh=np.zeros(16) if h is None else h[0,0].numpy();assert np.array_equal(bh,a['hidden_before'][k])
            with torch.no_grad():action,loc,h,_=q.step(torch.tensor(x[None]),h)
            assert np.array_equal(action[0].numpy(),a['actions'][k])
            for name,v in q.tape[-1].items():assert np.array_equal(v,a[name][k]),(r['id'],k,name)
            assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
            d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1]
            reward=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))
            if c or g or t:reward+=K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t))
            assert abs(reward-a['rewards'][k])<1e-12
        assert np.array_equal(h[0,0].numpy(),a['last_hidden_after']) and K.R.arrays_hash(a['final_native_integration'])==r['integration_sha256']
        assert rep['job_id']==r['job_id'] and rep['array_hashes']==r['array_hashes'] and rep['integration_sha256']==r['integration_sha256'] and rep['all_arrays_hidden_history_integration_exact']
        physics+=r['physics_steps']+rep['physics_steps']
        matches=[v['id'] for v in oldrows if v['role']=='final_iid' and v['job_id']==r['job_id'] and v['all_arrays_digest']==digest(r['array_hashes']) and v['integration_sha256']==r['integration_sha256']]
        if matches:history.append(dict(job=r['job_id'],historical_exact_matches=matches,independent_new=False))
    assert physics==s['physics_steps']<=p['max_physics'] and s['commands']==19 and s['elapsed_s']<=1200
    assert s['Actor_updates']==s['optimizer_calls']==0 and s['all_workers_stopped']
    save(dest,dict(state='PASS',native_added=0,optimizer_added=0,full120_density_hidden_history_reward_exact=True,source_model_data_SHA_exact=True,
        all8_native_regenerations_exact=True,historical_exact_matches=history,physics_steps=physics,commands=19,result=s['result'],stable_original_six=False))
    signal.alarm(0);print(json.dumps(dict(state='AUDIT_PASS',native=0,Adam=0)))


if __name__=='__main__':globals()[sys.argv[1]]()

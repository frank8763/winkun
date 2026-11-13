"""Conditional native policy-direction measurement; no optimizer or planner."""
from pathlib import Path
import contextlib,copy,fcntl,gzip,json,os,pickle,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t537_native_parameter_direction';DOC=ROOT/'stage/docs/day_20261113'
sys.path.insert(0,str(ROOT/'stage/experiments/t536_compact_native_restore'));import t536_runtime as R
F,G,A,E,P,np,torch=R.F,R.G,R.A,R.E,R.P,R.np,R.torch
read,save,sha,digest=R.read,R.save,R.sha,R.digest
def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
def verify(p,full=False):
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<p['absolute_all_stop'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    assert set(os.sched_getaffinity(0))<={14,15} and used(OUT)<=p['max_bytes']
    import shutil
    assert used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (p['inputs'] if full else p['protected_files']).items():assert sha(f)==h,f
def append(name,x):
    with (OUT/name).open('a') as h:h.write(json.dumps(x)+'\n')
def direction(p):
    old=P.checkpoint(p['direction_parent']['path'])['joint_state']['actor']
    new=P.checkpoint(p['direction_candidate']['path'])['joint_state']['actor']
    assert set(old)==set(new)
    q=A.frozen(read(p['actor_configuration']),p['direction_parent']['path'])
    names={k for k,v in q.named_parameters()};d={k:new[k]-old[k] for k in sorted(names)}
    assert all(torch.equal(old[k],new[k]) for k in old if k not in names),'Nonparameter buffer changed'
    assert digest(d)==p['direction_sha256']
    return d
def perturb(q,base,d,alpha):
    with torch.no_grad():
        for k,v in q.named_parameters():v.copy_(base[k]+float(alpha)*d[k])
    q.sample=False;q.rng=None;q.reset_noise()
def hidden(q,states):
    h=torch.zeros((1,1,64),dtype=torch.float64)
    with torch.no_grad():
        for x in states:
            n=q.normalized(torch.tensor(x)[None]);_,h=q.core(q.encoder(n)[None],h)
    return h
def metrics(payload):
    goal=payload['task_goal_xy'];before=payload['poses'][0,:2];after=payload['next_poses'][-1,:2]
    # Native final-goal distance and actual original discounted reward. Neither
    # metric implies this local direction solves a long obstacle detour.
    d0=float(np.linalg.norm(before-goal));d1=float(np.linalg.norm(after-goal));ret=0.
    for reward,n in zip(payload['rewards'][::-1],payload['native_counts'][::-1]):ret=float(reward)+R.E.N.K.R.GAMMA**int(n)*ret
    return dict(end_goal_distance_m=d1,progress_m=d0-d1,discounted_native_reward=ret,min_recorded_lidar_m=float(min(payload['states'][:,:48].min(),payload['next_states'][:,:48].min())*6.),contact=bool(payload['end_flags'][:,0].any()),goal=bool(payload['end_flags'][:,1].any()),timeout=bool(payload['end_flags'][:,2].any()),physics=int(payload['native_counts'].sum()),saturated_commands=int((np.abs(payload['actions'])==1.).sum()))
def analyze(p,rows):
    result=[]
    for window in p['windows']:
        group=[r for r in rows if r['id']==window['id']];first=[r for r in group if r['repeat']==0]
        assert len(group)==10 and len(first)==5
        for r in first:
            twin=next(z for z in group if z['repeat']==1 and z['alpha']==r['alpha'])
            assert r['array_hashes']==twin['array_hashes'] and r['metrics']==twin['metrics'] and r['policy_hash']==twin['policy_hash']
        by={r['alpha']:r for r in first};base=by[0.]['metrics'];derivatives=[]
        for name in ['end_goal_distance_m','discounted_native_reward','min_recorded_lidar_m']:
            eps=p['epsilon'];coarse=(by[eps]['metrics'][name]-by[-eps]['metrics'][name])/(2*eps);fine=(by[eps/2]['metrics'][name]-by[-eps/2]['metrics'][name])/eps
            effect=max(abs(by[eps]['metrics'][name]-by[-eps]['metrics'][name]),abs(by[eps/2]['metrics'][name]-by[-eps/2]['metrics'][name]))
            resolved=effect>p['effect_floor'];stable=resolved and coarse*fine>0 and abs(coarse-fine)<=p['relative_scale_tolerance']*max(abs(coarse),abs(fine))
            derivatives.append(dict(metric=name,coarse=coarse,fine=fine,max_antithetic_effect=effect,resolved=resolved,scale_stable=stable,unresolved_NOT_pass=not resolved))
        primary=derivatives[0]
        result.append(dict(id=window['id'],base_metrics=base,derivatives=derivatives,short_primary_direction=('toward_T527_reduces_distance' if primary['fine']<0 else 'toward_T527_increases_distance') if primary['scale_stable'] else 'UNRESOLVED_OR_SCALE_UNSTABLE',contact_in_any_branch=any(r['metrics']['contact'] for r in first),clipped_in_any_branch=any(r['metrics']['saturated_commands'] for r in first),exact_repeat=True))
    resolved=all(w['derivatives'][0]['scale_stable'] and not w['contact_in_any_branch'] for w in result)
    return dict(reason='NATIVE_LOCAL_DIRECTION_MEASURED' if resolved else 'NATIVE_LOCAL_DIRECTION_PARTLY_UNRESOLVED_OR_NONSMOOTH',windows=result,all_primary_local_scale_checks=resolved,Actor_Adam=0,Value_Adam=0,policy_gradient_updates=0,new_navigation_result=None,whole_episode_gradient_NOT_measured=True,conditional_fixed_history_prefix=True,local_effect_NOT_long_horizon_success=True)

def main():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.N.offline();E.set_out(OUT);E.L.install_stop_handlers()
    F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify
    config=read(p['actor_configuration']);d=direction(p);counts=0;commands=0;start=time.time();deadline=min(start+p['main_seconds'],p['absolute_native_stop']);rows=[];state='FAILED';error=None;s=w=None;last=[0.]
    def guard(physical=False):
        E.L.check_stop();assert time.time()<deadline-5 and counts<p['max_physics']
        if time.time()-last[0]>1.:verify(p);last[0]=time.time()
    def charge():
        nonlocal counts
        counts+=1;assert counts<=p['max_physics']
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        folder=OUT/'native';folder.mkdir()
        with (OUT/'native_console.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            module,native,create,w,reward=F.field_setup(folder,p);commands+=1;original=copy.deepcopy(module.ACTIVE);E.L.install_stop_handlers()
            for wi,window in enumerate(p['windows']):
                verify(p,True);source=window['row'];a=E.N.S.arrays(source['path']);assert all(digest(v)==source['array_hashes'][k] for k,v in a.items())
                snap=pickle.loads(gzip.decompress(Path(window['snapshot_path']).read_bytes()));first=window['first'];assert snap['steps']==first*100 and not snap['rows'] and not snap['qstate'][2]
                q=A.frozen(config,source['request']['model']);base=copy.deepcopy(q.state_dict());job=source['request']['job']
                module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=job.get('circle_shift_x_m',original['circle_shift_x_m'])
                s=F.FieldSession(job,native,create,w,guard,charge,q);commands+=1;E.L.install_stop_handlers()
                data={}
                for repeat in [0,1]:
                    for ai,alpha in enumerate(p['alphas']):
                        s.restore(snap);commands+=1;perturb(q,base,d,alpha);s.hidden=hidden(q,a['states'][:first])
                        if alpha==0.:assert np.array_equal(s.hidden.numpy(),snap['hidden'].numpy()) and q.phash()==source['policy_hash']
                        h0=s.hidden.numpy().copy();before=counts
                        for k in range(p['holds']):s.step()
                        payload=R.continuation(s,0);assert len(payload['states'])==p['holds'];assert counts-before==500
                        if alpha==0.:
                            for key in ['states','next_states','actions','native_counts','poses','next_poses','rewards','hidden_before','locations']:
                                assert np.array_equal(payload[key],a[key][first:first+p['holds']]),key
                            with np.load(window['reference_suffix_path']) as reference:
                                assert all(np.array_equal(v,reference['restored__'+k]) for k,v in payload.items()),'Baseline differs from T536 exact restored suffix'
                        maxdelta=float(np.max(np.abs(payload['actions'][0]-a['actions'][first])))
                        assert maxdelta<=p['max_first_command_change']
                        key=f'r{repeat}a{ai}';data.update({key+'__'+k:v for k,v in payload.items()});m=metrics(payload)
                        row=dict(id=window['id'],repeat=repeat,alpha=alpha,policy_hash=q.phash(),hidden_initial_sha256=digest(h0),first_command_max_change=maxdelta,metrics=m,array_hashes={k:digest(v) for k,v in payload.items()},payload_prefix=key,payload_path=str(OUT/f'window_{wi}.npz'))
                        rows.append(row);append('cases.jsonl',row)
                path=OUT/f'window_{wi}.npz';E.N.write_npz(path,data,p['window_bytes']);save(OUT/f'window_{wi}.json',dict(id=window['id'],payload_sha256=sha(path),rows=rows[-10:]))
                s.close();s=None;save(OUT/'status.json',dict(state='WINDOW_COMPLETE',window=wi,physics=counts,commands=commands,pid=os.getpid(),epoch=time.time()))
        assert counts==p['max_physics'] and commands==p['max_commands'];result=analyze(p,rows);state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();result=None;save(OUT/'failure.json',dict(error=error,physics=counts,commands=commands,raw_preserved=True,no_restart=True))
    finally:
        if s:s.close()
        if w:w.close()
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=counts,commands=commands,elapsed_s=time.time()-start,error=error,result=result));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE',error
    print('T537_NATIVE_DIRECTION_COMPLETE',flush=True)

def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    d=direction(p);config=read(p['actor_configuration']);rows=[]
    for wi,window in enumerate(p['windows']):
        m=read(OUT/f'window_{wi}.json');path=OUT/f'window_{wi}.npz';assert sha(path)==m['payload_sha256'];a=E.N.S.arrays(window['row']['path'])
        q=A.frozen(config,window['row']['request']['model']);base=copy.deepcopy(q.state_dict())
        with np.load(path) as z:
            for row in m['rows']:
                payload={k.removeprefix(row['payload_prefix']+'__'):z[k] for k in z.files if k.startswith(row['payload_prefix']+'__')}
                assert {k:digest(v) for k,v in payload.items()}==row['array_hashes'] and metrics(payload)==row['metrics']
                perturb(q,base,d,row['alpha']);h=hidden(q,a['states'][:window['first']]);assert digest(h.numpy())==row['hidden_initial_sha256'] and q.phash()==row['policy_hash']
                with torch.no_grad():
                    for k,x in enumerate(payload['states']):
                        assert np.array_equal(h.numpy().reshape(64),payload['hidden_before'][k].reshape(64))
                        action,loc,h,_=q.step(torch.tensor(x[None]),h)
                        assert np.array_equal(action[0].numpy(),payload['actions'][k]) and np.array_equal(loc[0].numpy(),payload['locations'][k])
                assert np.array_equal(h[0,0].numpy(),payload['last_hidden_after']);rows.append(row)
    assert s['result']==analyze(p,rows) and s['physics']==sum(r['metrics']['physics'] for r in rows)==p['max_physics'] and s['commands']==p['max_commands']
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,all_arrays_policies_conditional_hidden_metrics_and_exact_repeats_rebuilt=True,physics=s['physics'],commands=s['commands'],navigation_test=None,policy_update=0));print('T537_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

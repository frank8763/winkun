"""Bounded original/native relabel witnesses, then zero-update support screen."""
from pathlib import Path
import sys,os,json,time,io,contextlib,signal,traceback,shutil,collections,builtins,symtable
from t295_contract import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t295_hindsight_support';DOC=ROOT/'stage/docs/distance_20261102'
sys.path.insert(0,str(ROOT/'stage/experiments/t278_fixed_goal_starts'))
from t278_guard import Budget
sys.path.insert(0,str(ROOT/'stage/experiments/t294_paired_action_value'))
sys.path.insert(0,str(ROOT/'stage/experiments/t280_local_cost'))
from t294_native import LimitedLog

def save(f,v):Path(f).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify(p):
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert S.used(OUT)<=p['final_bytes'] and S.used(ROOT/'stage/runs')<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

class Witness:
    def __init__(self,Base):
        self.Base=Base;self.original=Base._history_frame;self.events={}
        def hook(env,lidar,d,a):
            f=self.original(env,lidar,d,a)
            self.events.setdefault(id(env),collections.deque(maxlen=48)).append((f,np.asarray(env._pose()[:3],dtype=np.float64),env.goal_xy.copy(),float(env.data.time)))
            return f
        Base._history_frame=hook
    def get(self,env):
        events=list(self.events.get(id(env),[]));result=[]
        for f in env.history:
            matches=[r for r in events if r[0] is f]
            if not matches:
                matches=[r for r in events if np.array_equal(r[0],f)]
                assert matches,'Missing actual factory/history generation witness; no invented pose'
                assert all(np.array_equal(r[1],matches[0][1]) and np.array_equal(r[2],matches[0][2]) and r[3]==matches[0][3] for r in matches),'Ambiguous copied history pose'
            result.append(matches[-1])
        assert len(result)==4
        return tuple(np.asarray([r[k] for r in result]) for k in [1,2,3])
    def alternate(self,env,goal,initial):
        hp,_,ht=initial;env.goal_xy=np.asarray(goal,dtype=np.float64).copy();env.prev_goal_dist=fields(env._pose(),goal)[1]
        frames=[];self.events[id(env)]=collections.deque(maxlen=48)
        for f,p,t in zip(env.history,hp,ht):
            f=f.copy();_,d,a=fields(p,goal);f[8]=np.clip(d/10,0,2);f[9]=np.clip(a/math.pi,-1,1);frames.append(f)
            self.events[id(env)].append((f,p.copy(),env.goal_xy.copy(),float(t)))
        env.history=collections.deque(frames,maxlen=4)
        return np.asarray(env._observation(update_history=False),dtype=np.float32)
    def close(self):self.Base._history_frame=self.original

def reward(d0,d1,n,contact,goal,timeout):
    disc=R.GAMMA**n;terminal=contact or goal or timeout;event=float(goal)-float(contact)-.5*float(timeout)
    return -.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0.)

def replay(source,p,native,create,witness,meter,status,alternate=None):
    old=load(source['path']);env=None;steps=0;rows=[];hp=[];hg=[];ht=[];first=None;partial=None;term=trunc=False;info={};error=None
    goal=np.array(source['auxiliary_goal_xy'],dtype=np.float64);cap=source['job']['case']['step_cap'];initial=None
    try:
        env,obs,_=R.task_env(create,source['job']);obs=C.perturb_anchor(env,obs,source['job']);scene=C.model_hash(env)
        initial=witness.get(env);assert not env._collision();original_goal=env.goal_xy.copy()
        if alternate is not None:
            assert all(np.array_equal(a,b) for a,b in zip(initial,alternate['initial']))
            obs=witness.alternate(env,goal,initial)
        active_goal=env.goal_xy.copy();previous=np.zeros(2,dtype=np.float32);pn=0
        for k,action in enumerate(old['actions']):
            assert time.time()<meter.deadline-p['final_reserve_seconds'];meter.hold();x=L.packet(env,obs,steps,cap,previous,pn);pose=np.asarray(env._pose()[:3]);meta=witness.get(env)
            identity=rebuild(x,pose,meta[0],active_goal,meta[1]);assert float(np.max(np.abs(identity-x)))<=2e-6
            if alternate is None:
                assert np.array_equal(x,old['states'][k]) and np.array_equal(pose,old['poses'][k]);hp.append(meta[0]);hg.append(meta[1]);ht.append(meta[2])
            else:
                expected=rebuild(old['states'][k],pose,alternate['hp'][k],goal)
                assert np.array_equal(x,expected),'Full120 relabel differs from native alternate-goal observation'
            d0=float(env._goal_state()[1]);before=steps
            for _ in range(min(100,cap-steps)):
                assert time.time()<meter.deadline-p['final_reserve_seconds']
                obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;meter.physics+=1
                if alternate is None and first is None and not info['collision'] and np.linalg.norm(np.asarray(env._pose()[:2])-goal)<=.2:
                    first=steps;n=steps-before;tm=witness.get(env)
                    partial=dict(state=L.packet(env,obs,steps,cap,action,n),pose=np.asarray(env._pose()[:3]),hp=tm[0],hg=tm[1],ht=tm[2],integration_sha256=R.arrays_hash(R.integration(env)))
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);nxt=L.packet(env,obs,steps,cap,action,n);npose=np.asarray(env._pose()[:3]);nm=witness.get(env)
            rw=reward(d0,d1,n,info['collision'],info['reach_goal'],trunc)
            rows.append((x,action,n,nxt,pose,npose,[info['collision'],info['reach_goal'],trunc],rw))
            if alternate is None:
                assert n==old['native_counts'][k] and np.array_equal(nxt,old['next_states'][k]) and np.array_equal(npose,old['next_poses'][k]) and rw==old['environment_rewards'][k]
                if term or trunc:
                    hp.append(nm[0]);hg.append(nm[1]);ht.append(nm[2]);break
            elif term or trunc:break
            previous=action.copy();pn=n
        integration=R.arrays_hash(R.integration(env));assert scene==C.model_hash(env)
        if alternate is None:
            assert steps==source['physics_steps'] and integration==source['integration_sha256'] and (term or trunc)
            assert len(hp)==len(old['states'])+1
        else:
            if alternate['first'] is not None:
                assert steps==alternate['first'] and info['reach_goal'] and not info['collision']
                assert integration==alternate['partial']['integration_sha256']
            else:assert steps==source['physics_steps'] and not info['reach_goal']
        arrays={name:np.asarray([r[j] for r in rows]) for j,name in enumerate(['states','actions','native_counts','next_states','poses','next_poses','end_flags','environment_rewards'])}
        arrays['native_counts']=arrays['native_counts'].astype(np.uint8);arrays['end_flags']=arrays['end_flags'].astype(np.uint8)
        row=dict(id=source['id'],mode='alternate' if alternate is not None else 'original',physics_steps=steps,decisions=len(rows),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),complete=bool(term or trunc),censored=not bool(term or trunc),integration_sha256=integration,array_hashes={k:S.digest(v) for k,v in arrays.items()},scene_sha256=scene)
        if alternate is None:
            assert all(S.digest(arrays[k])==source['array_hashes'][k] for k in ['states','actions','native_counts','next_states','poses','next_poses','end_flags','environment_rewards'])
        return dict(arrays=arrays,row=row,hp=np.asarray(hp),hg=np.asarray(hg),ht=np.asarray(ht),initial=initial,first=first,partial=partial)
    except BaseException:
        mode='alternate' if alternate is not None else 'original'
        evidence={name:np.asarray([r[j] for r in rows]) for j,name in enumerate(['states','actions','native_counts','next_states','poses','next_poses','end_flags','environment_rewards'])}
        evidence.update(history_poses=np.asarray(hp),history_goals=np.asarray(hg),history_times=np.asarray(ht))
        if env is not None:
            evidence['interruption_integration']=R.integration(env)
            evidence['interruption_pose']=np.asarray(env._pose())
        buf=io.BytesIO();np.savez_compressed(buf,**evidence);b=buf.getvalue();assert len(b)<=p['partial_bytes'];meter.charge_payload(len(b))
        f=OUT/('partial_'+source['id']+'_'+mode+'.npz');f.write_bytes(b)
        save(f.with_suffix('.json'),dict(id=source['id'],mode=mode,physics_steps=steps,completed_decisions=len(rows),path=str(f),sha256=C.sha(f),traceback=traceback.format_exc(),administrative_or_contract_interruption_not_navigation_failure=True))
        raise
    finally:
        if env is not None:
            witness.events.pop(id(env),None);env.close()

def support(source,a,q):
    old=load(source['path']);m,lp=density(q,old['states'],old['actions']);assert np.array_equal(m,old['means']) and np.array_equal(lp,old['logp'])
    original_identity_logratio_max=float(np.max(np.abs(lp-old['logp'])));assert original_identity_logratio_max==0
    z=a['arrays'];n=len(z['states']);result=dict(id=source['id'],source_reset_id=source['source_reset_id'],original_density_identity_max_error=original_identity_logratio_max,newgoal_density_invertible=False,success=a['row']['success'],collision=a['row']['collision'],timeout=a['row']['timeout'],censored=a['row']['censored'])
    try:
        m2,lp2=density(q,z['states'],z['actions']);delta=lp2-old['logp'][:n];prefix=delta.cumsum();assert np.isfinite(prefix).all()
        z['means']=m2;z['newgoal_logp']=lp2;z['behavior_logp']=old['logp'][:n];z['prefix_log_ratio']=prefix
        result.update(newgoal_density_invertible=True,terminal_log_weight=float(prefix[-1]),prefix_log_min=float(prefix.min()),prefix_log_max=float(prefix.max()),max_abs_conditional_log_ratio=float(np.abs(delta).max()),auxiliary_environment_G0=float(S.monte_carlo(z['environment_rewards'],z['native_counts'])[0]))
    except AssertionError as error:
        result.update(density_rejection=str(error),no_density_clipping_or_replacement=True)
    return result

def screen(p,items):
    groups=[]
    for name in p['reset_ids']:
        rows=[r for r in items if r['source_reset_id']==name];assert len(rows)==4
        ok=all(r['newgoal_density_invertible'] and not r['censored'] for r in rows)
        d=weights([r['terminal_log_weight'] for r in rows]) if ok else None
        passed=ok and d['ESS']>=p['ESS_min'] and d['max_weight']<=p['max_normalized_weight'] and max(abs(v) for v in d['log_weights'])<=p['max_abs_terminal_log_weight']
        groups.append(dict(reset_id=name,support=bool(passed),weights=d,eligible_complete=sum(not r['censored'] for r in rows),safe_auxiliary_goal=sum(r['success'] and not r['collision'] for r in rows)))
    return dict(passed=all(g['support'] for g in groups),groups=groups)

def preflight():
    assert not any((OUT/n).exists() for n in ['status.json','preflight.json']);p=json.loads((HERE/'protocol.json').read_text());verify(p);torch.set_num_threads(1)
    q=policy(json.loads(Path(p['source_protocol']).read_text())).eval().requires_grad_(False);assert S.digest(q.state_dict())==p['policy_hash']
    for s in p['sources']:
        a=load(s['path']);assert a['states'].shape[1]==120 and s['policy_hash']==p['policy_hash'];m,lp=density(q,a['states'],a['actions']);assert np.array_equal(m,a['means']) and np.array_equal(lp,a['logp'])
        assert np.linalg.norm(a['poses'][0,:2]-s['auxiliary_goal_xy'])>.2
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes']
    assert len(p['sources'])==8 and p['max_commands']==16 and p['max_physics']==448000
    assert 8*p['episode_bytes']+p['metadata_bytes']<=p['final_bytes']
    save(OUT/'preflight.json',dict(state='PASS',native_steps=0,optimizer_calls=0,Actor_updates=0,all8_original_AR_densities_bit_exact=True,synthetic=enumeration(),full_capacity_reserved_bytes=p['final_bytes']))
    print('T295 preflight PASS zero native/optimizer')

def main():
    assert not (OUT/'status.json').exists();p=json.loads((HERE/'protocol.json').read_text());start=time.time();meter=None;witness=None;log=None;raw=[];items=[];state='COMPLETE';gate=None
    def status(phase):save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=time.time(),absolute_deadline_epoch=start+p['wall_seconds'],commands=0 if meter is None else meter.commands,physics_steps=0 if meter is None else meter.physics,Actor_updates=0,optimizer_calls=0))
    def alarm(signum,frame):raise TimeoutError('T295 absolute wall budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds'])
    try:
        verify(p);meter=Budget(ROOT,OUT,p,start_epoch=start);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        q=policy(json.loads(Path(p['source_protocol']).read_text())).eval().requires_grad_(False);assert S.digest(q.state_dict())==p['policy_hash'];save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));status('STARTED');(OUT/'native').mkdir();log=LimitedLog(OUT/'native.log')
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):native,create=S.native_setup(OUT/'native')
        from training.e2e_phase10_lidar_v1.phase10_scene_e2e_env import Phase10SceneE2ELidarEnv as Base
        witness=Witness(Base)
        for source in p['sources']:
            meter.begin(source['job']['case']['step_cap']);status('ORIGINAL_'+source['id'])
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):original=replay(source,p,native,create,witness,meter,status)
            # Save original witness immediately, retaining evidence if alternate fails.
            payload=dict(history_poses=original['hp'],history_goals=original['hg'],history_times=original['ht'],first_hit=np.array([-1 if original['first'] is None else original['first']],dtype=np.int64))
            for k,v in (original['partial'] or {}).items():
                if k!='integration_sha256':payload['first_hit_'+k]=v
            f=OUT/(source['id']+'.npz');buf=io.BytesIO();np.savez_compressed(buf,**payload);b=buf.getvalue();assert len(b)<=p['episode_bytes'];f.write_bytes(b)
            raw.append(dict(source_id=source['id'],original=original['row'],path=str(f),sha256=C.sha(f),first_hit=original['first']));save(OUT/'cases.json',raw)
            meter.begin(source['job']['case']['step_cap']);status('ALTERNATE_'+source['id'])
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):alt=replay(source,p,native,create,witness,meter,status,original)
            d=support(source,alt,q);items.append(d);save(OUT/'support.json',items)
            for k,v in alt['arrays'].items():payload['alternate_'+k]=v
            buf=io.BytesIO();np.savez_compressed(buf,**payload);b=buf.getvalue();assert len(b)<=p['episode_bytes'];meter.charge_payload(len(b));f.write_bytes(b)
            raw[-1].update(alternate=alt['row'],sha256=C.sha(f),payload_array_hashes={k:S.digest(v) for k,v in payload.items()});save(OUT/'cases.json',raw);meter.checkpoint()
        assert len(raw)==len(items)==8 and S.digest(q.state_dict())==p['policy_hash'];gate=screen(p,items);verify(p)
    except BaseException:
        state='FAILED';save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=0 if meter is None else meter.physics,commands=0 if meter is None else meter.commands))
    finally:
        signal.alarm(0)
        if witness is not None:witness.close()
        if log is not None:log.close()
        if meter is not None:meter.close()
    result=dict(state=state,support_screen=gate,commands=0 if meter is None else meter.commands,physics_steps=0 if meter is None else meter.physics,elapsed_s=time.time()-start,Actor_updates=0,optimizer_calls=0,critic_updates=0,all_workers_stopped=True,own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),six_metre_gate_evaluated=False,seven_or_eight_started=False,policy_hash=p['policy_hash'],limitation=p['scientific_scope'])
    save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result),flush=True)

def audit():
    dest=DOC/'T295_AUDIT.json';assert not dest.exists();p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text());verify(p);assert s['state']=='COMPLETE';torch.set_num_threads(1)
    for pid in [json.loads((OUT/'workers.json').read_text())['parent'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live/reused process',pid))
    q=policy(json.loads(Path(p['source_protocol']).read_text())).eval().requires_grad_(False);rows=json.loads((OUT/'cases.json').read_text());details=[];total=0
    for source,r in zip(p['sources'],rows):
        assert C.sha(r['path'])==r['sha256'];z=load(r['path']);assert {k:S.digest(v) for k,v in z.items()}==r['payload_array_hashes'];old=load(source['path']);a={k.removeprefix('alternate_'):v for k,v in z.items() if k.startswith('alternate_')}
        points=np.concatenate([old['states'],old['next_states'][-1:]]);poses=np.concatenate([old['poses'],old['next_poses'][-1:]])
        for x,pose,hp,hg in zip(points,poses,z['history_poses'],z['history_goals']):assert np.max(np.abs(rebuild(x,pose,hp,source['goal_xy'],hg)-x))<=2e-6
        for k,x in enumerate(a['states']):assert np.array_equal(x,rebuild(old['states'][k],old['poses'][k],z['history_poses'][k],source['auxiliary_goal_xy']))
        first=int(z['first_hit'][0]);n=len(a['states'])
        for k in range(n):
            if k==n-1 and first>0:
                expected=rebuild(z['first_hit_state'],z['first_hit_pose'],z['first_hit_hp'],source['auxiliary_goal_xy']);assert np.array_equal(a['next_states'][k],expected)
            else:assert np.array_equal(a['next_states'][k],rebuild(old['next_states'][k],old['next_poses'][k],z['history_poses'][k+1],source['auxiliary_goal_xy']))
        assert all(S.digest(a[k])==r['alternate']['array_hashes'][k] for k in r['alternate']['array_hashes'])
        for k in range(n):
            d0=fields(a['poses'][k],source['auxiliary_goal_xy'])[1];d1=fields(a['next_poses'][k],source['auxiliary_goal_xy'])[1];c,g,t=map(bool,a['end_flags'][k]);rw=reward(d0,d1,int(a['native_counts'][k]),c,g,t);assert rw==a['environment_rewards'][k]
        item=dict(arrays={k:v.copy() for k,v in a.items()},row=r['alternate']);d=support(source,item,q);details.append(d)
        for k in ['means','newgoal_logp','behavior_logp','prefix_log_ratio']:
            if d['newgoal_density_invertible']:assert np.array_equal(a[k],item['arrays'][k])
        total+=r['original']['physics_steps']+r['alternate']['physics_steps']
    assert details==json.loads((OUT/'support.json').read_text());assert screen(p,details)==s['support_screen'];assert total==s['physics_steps']<=p['max_physics'] and s['commands']==16
    assert S.digest(q.state_dict())==p['policy_hash'];verify(p)
    save(dest,dict(state='PASS',native_added=0,optimizer_calls=0,Actor_updates=0,all8_original_history_current_goal_identity=True,all8_native_alternate_full120_bit_exact=True,all13_goal107_nongoal_fields_verified=True,all_AR_original_new_likelihoods_recomputed=True,first_hit_short_holds_rewards_and_terminal_states_exact=True,all_original_processes_absent=True,total_commands=16,total_physics=total,screen=s['support_screen'],synthetic=enumeration(),summary_sha256=C.sha(OUT/'summary.json')));print('T295 independent audit PASS zero native/optimizer')

if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:preflight()
    elif sys.argv[1:]==['--audit']:audit()
    else:assert not sys.argv[1:];main()

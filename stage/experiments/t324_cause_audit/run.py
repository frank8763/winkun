"""Finite, frozen, zero-native cause audit of existing Stage evidence.

Input interventions query a network on recorded states. They are not physical
counterfactual outcomes, expert labels, training data or navigation successes.
"""
from pathlib import Path
import copy, fcntl, gzip, hashlib, io, json, os, shutil, signal, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t324_cause_audit';DOC=ROOT/'stage/docs/distance_20261103'
sys.path.insert(0,str(ROOT/'stage/experiments/t319_imperfect_demo_head'))
import t319_core as H
np,torch,K=H.np,H.torch,H.K

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    f=Path(path);b=f.read_bytes()
    return json.loads(gzip.decompress(b) if f.suffix=='.gz' else b)
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def used():return sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
def arrays(path):
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    if 'last_next_state' in a:a['next_states']=np.concatenate([a['states'][1:],a.pop('last_next_state')])
    return a
def deny(*a,**k):raise AssertionError('T324 forbids native setup and optimizer steps')
def snapshots(path):return K.C.read_snapshot(path)

def prepare():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    p0=read(ROOT/'stage/experiments/t322_full_network_steering/protocol.json')
    data={
        'five_LEFT':'t288_five_metre_presentation/left_full120.npz',
        'five_RIGHT':'t288_five_metre_presentation/right_full120.npz',
        'six_LEFT':'t256_distance_frontier/episodes/base6_LEFT_11.npz',
        'six_RIGHT':'t256_distance_frontier/episodes/base6_RIGHT_11.npz',
        'human01':'t317_human_demonstration/attempt_01.npz',
        'human02':'t317_human_demonstration/attempt_02.npz',
        'completion01':'t323_human_state_completion/completion_01.npz',
        'completion02':'t323_human_state_completion/completion_02.npz'}
    data={k:str(ROOT/'stage/runs'/v) for k,v in data.items()}
    snaps={}
    for side,seed in [('LEFT',2029653601),('RIGHT',2029654802)]:
        snaps['five_'+side]=str(ROOT/f'stage/runs/t207_parameter_pg/tasks/original_5m_{seed}_{side}.npz')
        snaps['six_'+side]=str(ROOT/f'stage/runs/t256_distance_frontier/tasks/base6_{side}.npz')
    files=set(data.values())|set(snaps.values())|{p0['checkpoint'],p0['initial_base'],str(ROOT/'stage/runs/t322_full_network_steering/final_checkpoint.pt.gz')}
    # Lock original policy/observation implementations as well as new code.
    files|={f for f in p0['input_hashes'] if Path(f).suffix=='.py'}
    files|={str(f) for f in HERE.glob('*.py')}
    files|={str(ROOT/v) for v in ['stage/runs/t288_five_metre_presentation/recordings.json','stage/runs/t256_distance_frontier/distance.jsonl.gz','stage/runs/t323_human_state_completion/cases.json','stage/docs/distance_20261103/T324_PLAN.md']}
    p=dict(experiment='T324_FIXED_FROZEN_CAUSE_AUDIT',data=data,snapshots=snaps,
        policy_protocol=str(ROOT/'stage/experiments/t322_full_network_steering/protocol.json'),
        trained_checkpoint=str(ROOT/'stage/runs/t322_full_network_steering/final_checkpoint.pt.gz'),
        initial_policy_hash=p0['policy_hash'],input_hashes={f:sha(f) for f in sorted(files)},
        wall_seconds=300,audit_seconds=300,final_bytes=2*1024**2,
        global_bytes=1024**3,min_free_bytes=40*1024**3,
        native_commands=0,physics_steps=0,optimizer_calls=0,autograd_calls=0,Actor_updates=0,
        fixed_inputs=8,clock_column=112,goal_columns=[48,49,50,51,52,72,73,84,85,96,97,108,109],
        intervention_prefix_seconds=40.,five_cap=23334,six_cap=28000,
        scope='Descriptive audit only, no causal navigation claims or automatic training. All eight fixed trajectories retained; no seed/timing/grid selection.')
    assert used()+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();write(HERE/'protocol.json',p);write(OUT/'protocol.json',p)
    print('T324 registered zero native/optimizer/gradient')

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert sha(f)==h,f
    assert used()<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def models(p):
    pp=read(p['policy_protocol']);q=K.policy(pp).eval().requires_grad_(False)
    assert K.S.digest(q.state_dict())==p['initial_policy_hash']
    trained=copy.deepcopy(q);trained.load_state_dict(K.T.L.read(p['trained_checkpoint'])['policy'])
    trained.eval().requires_grad_(False)
    return q,trained

def preflight():
    assert not (OUT/'preflight.json').exists();p=read(OUT/'protocol.json');verify(p)
    torch.set_grad_enabled(False);torch.set_num_threads(1)
    q,trained=models(p);assert all(not v.requires_grad for m in [q,trained] for v in m.parameters())
    for tag in ['five_LEFT','five_RIGHT','six_LEFT','six_RIGHT']:
        a=arrays(p['data'][tag]);assert a['states'].shape[1]==120
        assert np.array_equal(predict(q,a['states'][:1]),a['actions'][:1]),tag
    write(OUT/'preflight.json',dict(state='PASS',native_commands=0,physics_steps=0,optimizer_calls=0,autograd_calls=0,
        frozen_models_loaded_and_four_actual_initial_actions_exact=True,protocol_sha256=sha(OUT/'protocol.json')))
    print('T324 preflight PASS zero native/optimizer/gradient')

@torch.no_grad()
def predict(q,x):return np.stack([q.step(torch.tensor(r[None]))[0][0].numpy() for r in x])

def metrics(a,goal):
    count=a['native_counts'].astype(np.int64);begin=np.r_[0,np.cumsum(count)[:-1]]*.002
    d0=np.linalg.norm(a['poses'][:,:2]-goal,axis=1);d1=np.linalg.norm(a['next_poses'][:,:2]-goal,axis=1)
    flags=a['end_flags'][-1].astype(bool);N=int(count.sum());G=float(np.sum(a['rewards']*K.R.GAMMA**np.r_[0,np.cumsum(count)[:-1]]))
    event=float(flags[1])-float(flags[0])-.5*float(flags[2])
    closed=.03*d0[0]-.000002*(1-K.R.GAMMA**N)/(1-K.R.GAMMA)+K.R.GAMMA**(N-1)*event
    assert abs(G-closed)<1e-11 and not a['end_flags'][:-1].any()
    rw=[]
    for k,n in enumerate(count):
        c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**int(n)
        rw.append(-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0[k]-z*.03*d1[k]*(not(c or g or t))+(K.R.GAMMA**(int(n)-1)*(float(g)-float(c)-.5*float(t)) if c or g or t else 0))
    assert np.max(np.abs(np.array(rw)-a['rewards']))<1e-11
    periods=[]
    for lo,hi in [(0,10),(10,30),(30,56)]:
        mask=(begin>=lo)&(begin<hi)
        if not mask.any():continue
        w=count[mask]/count[mask].sum()
        periods.append(dict(window=[lo,hi],seconds=float(count[mask].sum()*.002),
            mean_drive=float(w@a['actions'][mask,0]),mean_abs_steer=float(w@np.abs(a['actions'][mask,1])),
            chord_sum_m=float(np.linalg.norm(a['next_poses'][mask,:2]-a['poses'][mask,:2],axis=1).sum()),
            radial_progress_m=float((d0[mask]-d1[mask]).sum())))
    return dict(decisions=len(count),physics_steps=N,duration_s=N*.002,initial_distance_m=float(d0[0]),final_distance_m=float(d1[-1]),
        sampled_endpoint_min_m=float(min(d0.min(),d1.min())),continuous_minimum_not_measured=True,
        GOAL=bool(flags[1] and not flags[0]),CONTACT=bool(flags[0]),TIMEOUT=bool(flags[2]),
        discounted_return=G,closed_return=closed,reward_max_error=float(np.max(np.abs(np.array(rw)-a['rewards']))),periods=periods)

def analyze(p):
    K.S.native_setup=deny;torch.optim.Adam.step=deny;torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.set_grad_enabled(False)
    q,trained=models(p);before=[K.S.digest(v.state_dict()) for v in [q,trained]]
    a={k:arrays(v) for k,v in p['data'].items()};sn={k:snapshots(v) for k,v in p['snapshots'].items()}
    goals={k:np.asarray(s['attrs']['goal_xy']) for k,s in sn.items()}
    validation=read(ROOT/'stage/runs/t288_five_metre_presentation/recordings.json')
    for r in validation:
        tag='five_'+r['side'];assert sha(p['data'][tag])==r['native']['payload']['sha256']
        assert all(K.S.digest(a[tag][k])==h for k,h in r['native']['array_hashes'].items())
    rows=[json.loads(line) for line in gzip.decompress((ROOT/'stage/runs/t256_distance_frontier/distance.jsonl.gz').read_bytes()).decode().splitlines()]
    for side in ['LEFT','RIGHT']:
        r=next(r for r in rows if r['id']==f'base6_{side}_11');assert sha(p['data']['six_'+side])==r['sha256']
        assert all(K.S.digest(a['six_'+side][k])==h for k,h in r['array_hashes'].items())
    predictions={};trajectory={};initial=[];human=[];interventions=[]
    for tag,item in a.items():
        assert item['states'].shape[1]==120 and np.array_equal(item['states'][1:],item['next_states'][:-1])
        goal=goals[tag] if tag in goals else goals['six_LEFT']
        trajectory[tag]=metrics(item,goal)
        steps=np.r_[0,np.cumsum(item['native_counts'].astype(np.int64))[:-1]]
        cap=23334 if tag.startswith('five') else 28000
        assert np.max(np.abs(item['states'][:,112]-(1-steps/cap)))<1e-7
        pred=predict(q,item['states']);predictions[tag+'_safe']=pred
        predictions[tag+'_trained']=predict(trained,item['states'])
        if tag.startswith(('five','six')):assert np.array_equal(pred,item['actions']),(tag,float(np.max(np.abs(pred-item['actions']))))
        if tag.startswith('completion'):assert np.array_equal(pred[200:],item['actions'][200:])
    for side in ['LEFT','RIGHT']:
        five,six=a['five_'+side],a['six_'+side];diff=np.flatnonzero(five['states'][0]!=six['states'][0])
        keep=np.setdiff1d(np.arange(120),p['goal_columns'])
        assert np.array_equal(sn['five_'+side]['physics'],sn['six_'+side]['physics'])
        assert sn['five_'+side]['model_hash']==sn['six_'+side]['model_hash']
        assert np.array_equal(five['states'][0,keep],six['states'][0,keep])
        initial.append(dict(side=side,native_initial_state_exact=True,scene_hash_equal=True,
            differing_columns=diff.tolist(),non_goal107_exact=True,initial_clock_both=float(five['states'][0,112]),
            actual_five_action=five['actions'][0].tolist(),actual_six_action=six['actions'][0].tolist(),
            action_delta=(six['actions'][0]-five['actions'][0]).tolist(),
            interpretation='At this initial state, only declared goal fields differ; cap lengths do not differ in the initial normalized clock. This isolates the neural initial action response, not full physical route causality.'))
    for suffix in ['01','02']:
        tag='human'+suffix;item=a[tag];x=item['states'];y=item['actions'];changed=y[:,1]!=x[:,-2]
        def loss(pred):
            return {n:dict(rows=int(mask.sum()),steer_MSE=float(np.mean((pred[mask,1]-y[mask,1])**2)),drive_MSE=float(np.mean((pred[mask,0]-y[mask,0])**2)))
                for n,mask in [('ALL',np.ones(len(x),bool)),('CHANGE',changed),('HOLD',~changed)]}
        copy_pred=np.column_stack([x[:,-3],x[:,-2]])
        human.append(dict(id=tag,changed_indices=np.flatnonzero(changed).tolist(),safe=loss(predictions[tag+'_safe']),
            trained=loss(predictions[tag+'_trained']),copy_previous=loss(copy_pred),
            boundary40s=dict(actual_human=y[200].tolist(),safe=predictions[tag+'_safe'][200].tolist(),trained=predictions[tag+'_trained'][200].tolist()),
            caution='Both human trajectories TIMEOUT. Copy baseline is only a diagnostic, not a deployed policy; lower fitted error does not imply safe navigation.'))
        # Fixed 2x2 INPUT query, with true source history and previous actions.
        n=201;source=x[:n];hp=item['history_poses'][:n];hg=item['history_goals'][:n]
        g6=goals['six_LEFT'];g5=goals['five_LEFT'];alt=[]
        for k in range(n):
            history_goal=hg[k].copy();active=np.linalg.norm(history_goal-g6,axis=1)<1e-10
            history_goal[active]=g5 # retain actual factory goal frames
            alt.append(K.H.rebuild(source[k],item['poses'][k],hp[k],g5,history_goal))
        alt=np.array(alt);oldclock=(1-np.arange(n)*100/23334).astype(np.float32)
        clock=source.copy();clock[:,112]=oldclock;both=alt.copy();both[:,112]=oldclock
        variants={'actual6':predictions[tag+'_safe'][:n],'goal5':predict(q,alt),'clock5':predict(q,clock),'both5':predict(q,both)}
        for name,v in variants.items():predictions[tag+'_input_'+name]=v
        effects={name:dict(mean_abs_action_delta=np.mean(np.abs(v-variants['actual6']),axis=0).tolist(),
            max_abs_action_delta=np.max(np.abs(v-variants['actual6']),axis=0).tolist()) for name,v in variants.items() if name!='actual6'}
        interaction=variants['both5']-variants['goal5']-variants['clock5']+variants['actual6']
        interventions.append(dict(id=tag,rows=n,effects=effects,mean_abs_interaction=np.abs(interaction).mean(0).tolist(),
            off_policy_input_queries_only=True,physical_outcome_unknown=True,not_new_navigation_or_demonstration=True))
    after=[K.S.digest(v.state_dict()) for v in [q,trained]];assert before==after
    report=dict(scope='Fixed eight existing trajectories; frozen descriptive audit, no new navigation',
        initial_comparisons=initial,trajectory=trajectory,human_prediction=human,input_queries=interventions,
        model_hashes=before,models_unchanged=True,native_commands=0,physics_steps=0,optimizer_calls=0,autograd_calls=0,Actor_updates=0,
        all_original_rewards_and_clock_equations_verified=True,all_safe_policy_recorded_actions_bit_exact=True,
        no_causal_claim_from_input_queries=True,no_unseen_navigation_test_consumed=True)
    return report,predictions

def main():
    assert not (OUT/'status.json').exists();p=read(OUT/'protocol.json');verify(p);start=time.time()
    lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T324300s')));signal.alarm(p['wall_seconds'])
    write(OUT/'status.json',dict(state='READ_ONLY_ANALYSIS',pid=os.getpid(),main_epoch=start,deadline_epoch=start+p['wall_seconds']))
    write(OUT/'workers.json',dict(parent=os.getpid(),workers=[]))
    try:
        report,pred=analyze(p);buf=io.BytesIO();np.savez_compressed(buf,**pred);blob=buf.getvalue();assert len(blob)<512*1024
        (OUT/'frozen_predictions.npz').write_bytes(blob);write(OUT/'analysis.json',report);verify(p)
        summary=dict(state='COMPLETE',elapsed_s=time.time()-start,pid=os.getpid(),main_epoch=start,
            native_commands=0,physics_steps=0,optimizer_calls=0,autograd_calls=0,Actor_updates=0,
            all_workers_stopped=True,worker_processes_spawned=0,analysis_sha256=sha(OUT/'analysis.json'),prediction_sha256=sha(OUT/'frozen_predictions.npz'))
        assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<p['final_bytes']
        write(OUT/'summary.json',summary);write(OUT/'status.json',summary);print(json.dumps(summary))
    except BaseException:
        write(OUT/'failure.json',dict(error=traceback.format_exc(),native_commands=0,optimizer_calls=0))
        write(OUT/'summary.json',dict(state='FAILED',all_workers_stopped=True,pid=os.getpid(),native_commands=0,physics_steps=0,optimizer_calls=0,autograd_calls=0,Actor_updates=0))
        raise
    finally:signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def audit():
    dest=DOC/'T324_AUDIT.json';assert not dest.exists();p=read(OUT/'protocol.json');verify(p)
    s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc/'+str(s['pid'])).exists()
    assert not Path('/proc/'+str(read(OUT/'launch.json')['pid'])).exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T324 audit300s')));signal.alarm(p['audit_seconds'])
    recomputed,pred=analyze(p);assert recomputed==read(OUT/'analysis.json')
    saved=arrays(OUT/'frozen_predictions.npz');assert saved.keys()==pred.keys() and all(np.array_equal(saved[k],pred[k]) for k in pred)
    verify(p);write(dest,dict(state='PASS',all_frozen_predictions_metrics_raw_rewards_clock_and_source_hashes_exact=True,
        native_added=0,optimizer_calls=0,autograd_calls=0,Actor_updates=0,all_processes_absent=True,
        analysis_sha256=sha(OUT/'analysis.json'),prediction_sha256=sha(OUT/'frozen_predictions.npz')))
    signal.alarm(0);print('T324 independent zero-native recomputation PASS')

if __name__=='__main__':globals()[sys.argv[1]]()

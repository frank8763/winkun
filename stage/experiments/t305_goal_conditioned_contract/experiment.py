from gc_contract import *
import argparse,copy,fcntl,gzip,io,json,os,shutil,signal,subprocess,time,traceback
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t305_goal_conditioned_contract';DOC=ROOT/'stage/docs/distance_20261102'


def read(path):return json.loads(Path(path).read_text())
def resource(p):
    assert S.used(OUT)<p['final_bytes'] and S.used(ROOT/'stage/runs')<p['global_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    resource(p)


def prepare():
    assert not OUT.exists()
    source=ROOT/'stage/runs/t295_hindsight_support/protocol.json';p295=read(source);cases=ROOT/'stage/runs/t295_hindsight_support/cases.json';rows=read(cases)
    inputs={str(source):C.sha(source),str(cases):C.sha(cases)}
    sources=[]
    for source in p295['sources']:
        row=next(r for r in rows if r['source_id']==source['id']);f=Path(row['path']);assert C.sha(f)==row['sha256']
        original=H.resolve(source['path']);assert C.sha(original)==source['sha256']
        sources.append(dict(id=source['id'],witness_path=str(f),witness_sha256=row['sha256'],original_path=str(original),original_sha256=source['sha256'],goal=source['auxiliary_goal_xy']))
        inputs[str(f)]=row['sha256'];inputs[str(original)]=source['sha256']
    base=ROOT/'stage/runs/t267_causal_credit_course/protocol.json';b=read(base)
    for f in [base,Path(b['initial_base']),Path(b['checkpoint']),ROOT/'stage/docs/distance_20261102/T295_AUDIT.json',
              ROOT/'stage/docs/distance_20261102/T305_PLAN.md',*HERE.glob('*.py'),
              Path(H.__file__),Path(D.__file__),ROOT/'stage/experiments/t238_ar_density_recovery/copula.py']:
        inputs[str(f)]=C.sha(f)
    p=dict(experiment='T305',sources=sources,policy_protocol=str(base),input_hashes=inputs,
        wall_seconds=600,audit_seconds=600,final_bytes=1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        max_commands=0,max_physics=0,optimizer_calls=0,Actor_updates=0,
        contract='Engineering contract only. Existing8nativevalidated T295 goal/history/firsthit witnesses reused; zero newnative/Adam/Actorlearning/criticfit. Verify full13goal107nongoal reconstruction, actual historical poses, nativealternate nextstates/firsthitshortlast hold; exact originalAR density and currentprevious-mean conditional reparam derivatives. Implement finitephysicaldiscount decision-endpoint occupancy with safeGOALabsorption/collisionpriority/taskcapcemetery/administrativecensorunknown; not2mscontinuousoccupancy orsafe firsthitprobability orunbiasedcontinuingCRL theorem. Per-anchorSE2 queries explicitpairwise insteadofmixingbodyframes. Sourcebehavior120 goalcontext distinctfromreconstructedquery120Actor; transfergoal/continuation offpolicy discrepancy is a scientific limitation, not silently solved. SyntheticCritic InfoNCE+.01LSE, unknowncensor neitherpositive nornegative, nativeactiongradient/Actorreachablegradient andfrozencritic/no update/serialization contracts only; randomcriticneveracceptedforActor. ConditionalhistoryAR previousmean keptdifferentiable, oldreplay logpnever newgoalonpolicy. SingleActor120hiddenreset unchanged, no map/planner/waypoint/runtime teacher/router/Phase. FullRL requires ownseparatebudget/prereg/source andtruevaluecontrolgate. No6m/7m/8m promotion.')
    assert S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes']
    OUT.mkdir()
    for path in [HERE/'protocol.json',OUT/'protocol.json']:C.save(path,p)
    print('T305 registered engineering only; zero native/optimizer')


def calculate(p,guard):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=H.policy(read(p['policy_protocol']));before=S.digest(q.state_dict());details=[];sample=[];rebuild_count=0;identity=0.
    for source in p['sources']:
        guard()
        with np.load(source['witness_path']) as z:w={k:z[k].copy() for k in z.files}
        with np.load(source['original_path']) as z:a={k:z[k].copy() for k in z.files}
        n=len(w['alternate_states']);goal=source['goal']
        for k in range(n):
            x=H.rebuild(a['states'][k],a['poses'][k],w['history_poses'][k],goal)
            assert np.array_equal(x,w['alternate_states'][k]);rebuild_count+=1
            original=H.rebuild(a['states'][k],a['poses'][k],w['history_poses'][k],w['history_goals'][k][0],w['history_goals'][k])
            identity=max(identity,float(np.abs(original-a['states'][k]).max()))
            if k==n-1:
                nxt=H.rebuild(w['first_hit_state'],w['first_hit_pose'],w['first_hit_hp'],goal)
            else:nxt=H.rebuild(a['next_states'][k],a['next_poses'][k],w['history_poses'][k+1],goal)
            assert np.array_equal(nxt,w['alternate_next_states'][k]);rebuild_count+=1
        assert int(w['alternate_native_counts'][-1])<100
        assert not w['alternate_end_flags'][:-1].any() and np.array_equal(w['alternate_end_flags'][-1],[0,1,0])
        means,logp=H.density(q,a['states'],a['actions']);assert np.array_equal(means,a['means']) and np.array_equal(logp,a['logp'])
        # Different query goals do not license reusing these old behavior logp.
        indices=np.linspace(1,n-1,4).astype(int)
        for k in indices:
            sample.append((a['states'][k],w['alternate_states'][k],w['alternate_states'][k-1],a['actions'][k-1],relative(a['poses'][k],np.asarray(goal)),a['poses'][k],np.asarray(goal)))
        law=future_law(a['native_counts'],a['end_flags'],0,28000,True)
        details.append(dict(id=source['id'],native_rebuilt_rows=n*2,original_density_bit_exact=True,
            original_terminal_types=[r['type'] for r in law[-2:]],law_sum=sum(r['mass'] for r in law),
            original_physics=int(a['native_counts'].astype(np.int64).sum()),native_alternate_last_hold=int(w['alternate_native_counts'][-1])))
    assert identity<=2e-6 and len(sample)==32
    # Independent closed forms: safe absorption, collision priority, true taskcap,
    # and unknown collector censor are distinct outcomes, never invented failure.
    g=.91
    success=future_law([2,3],[[0,0,0],[0,1,0]],0,5,True,g)
    collision=future_law([2,3],[[0,0,0],[1,1,0]],0,5,True,g)
    timeout=future_law([2,3],[[0,0,0],[0,0,1]],0,5,True,g)
    censored=future_law([2,3],[[0,0,0],[0,0,0]],0,9,False,g)
    assert abs(success[-1]['mass']-g**5)<1e-14 and success[-1]['type']=='coordinate_absorbed_goal'
    assert abs(collision[-1]['mass']-g**2)<1e-14 and collision[-1]['type']=='collision'
    assert abs(timeout[-1]['mass']-g**5)<1e-14 and timeout[-1]['type']=='task_timeout'
    assert abs(censored[-1]['mass']-g**5)<1e-14 and censored[-1]['type']=='collector_censored_unknown'
    rng=np.random.default_rng(2026110210)
    selected=[draw(success,float(u)) for u in rng.random(32)]
    # SE2 transformation preserves all relative query coordinates, including
    # row-specific conversion of negatives; no absolute map enters the network.
    pose=np.array([.3,-.4,.6]);goals=np.array([[1.,2.],[-.7,.1]])
    theta=.8;rot=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
    shift=np.array([2.,-1.]);newpose=np.r_[pose[:2]@rot.T+shift,pose[2]+theta]
    err=float(np.abs(relative(pose,goals)-relative(newpose,goals@rot.T+shift)).max());assert err<1e-12
    # Same conditional bounded sampler, retaining previous-mean derivatives.
    m=torch.tensor([[.2,-.1]],dtype=torch.float64,requires_grad=True);pm=torch.tensor([[.1,-.2]],dtype=torch.float64,requires_grad=True)
    pa=torch.tensor([[.12,-.19]],dtype=torch.float64);eps=torch.tensor([[.4,-.5]],dtype=torch.float64);reset=torch.tensor([False])
    ac,_=conditional_action(m,pm,pa,reset,eps);weights=torch.tensor([[.3,-.7]],dtype=torch.float64)
    gm,gp=torch.autograd.grad((ac*weights).sum(),[m,pm]);fd=[];delta=1e-6
    for name,v,grad in [('current',m,gm),('previous',pm,gp)]:
        for k in range(2):
            plus=v.detach().clone();minus=plus.clone();plus[0,k]+=delta;minus[0,k]-=delta
            a1,_=conditional_action(plus if name=='current' else m.detach(),plus if name=='previous' else pm.detach(),pa,reset,eps)
            a0,_=conditional_action(minus if name=='current' else m.detach(),minus if name=='previous' else pm.detach(),pa,reset,eps)
            fd.append(abs(float(((a1-a0)*weights).sum()/(2*delta))-float(grad[0,k])))
    assert max(fd)<1e-5 and float(gp.abs().max())>1e-3
    # Random network here is only for interface/gradient/serialization checks.
    torch.manual_seed(2026110210);critic=Critic().eval();ch=S.digest(critic.state_dict())
    bh=torch.tensor(np.stack([r[0] for r in sample]));qs=torch.tensor(np.stack([r[1] for r in sample]));ps=torch.tensor(np.stack([r[2] for r in sample]));pas=torch.tensor(np.stack([r[3] for r in sample]));bg=torch.tensor(np.stack([r[4] for r in sample]),dtype=torch.float32)
    matrix=query_matrix(np.stack([r[5] for r in sample]),np.stack([r[6] for r in sample]))
    assert np.allclose(matrix[np.arange(32),np.arange(32)],bg.numpy(),rtol=0,atol=1e-6)
    body=torch.tensor(matrix,dtype=torch.float32);types=torch.zeros((32,32),dtype=torch.long);types[:,-1]=3
    logits=critic.logits(bh,torch.zeros(32,2),pas,body,types)
    valid=torch.ones(32,dtype=torch.bool);valid[-1]=False;loss=contrastive_objective(logits,valid)
    grad=torch.autograd.grad(loss,tuple(critic.parameters()));assert all(torch.isfinite(v).all() for v in grad)
    # Unknown censor removal is exactly the corresponding submatrix loss.
    expected=torch.nn.functional.cross_entropy(logits[:31,:31],torch.arange(31))+.01*torch.logsumexp(logits[:31,:31],1).square().mean()
    assert torch.equal(loss,expected)
    critic.requires_grad_(False)
    innovation=torch.tensor(rng.normal(size=(32,2)),dtype=torch.float64)
    objective,new_action=proposed_actor_objective(q,critic,bh,qs,ps,pas,torch.zeros(32,dtype=torch.bool),innovation,bg)
    params=[v for v in q.parameters() if v.requires_grad]
    actorgrad=torch.autograd.grad(objective,params,allow_unused=True)
    flat=torch.cat([torch.zeros_like(v).reshape(-1) if grad is None else grad.reshape(-1) for v,grad in zip(params,actorgrad)])
    assert torch.isfinite(flat).all() and float(flat.norm())>0 and bool((new_action.abs()<1).all())
    assert before==S.digest(q.state_dict()) and ch==S.digest(critic.state_dict())
    buf=io.BytesIO();torch.save(critic.state_dict(),buf);loaded=Critic();loaded.load_state_dict(torch.load(io.BytesIO(buf.getvalue()),map_location='cpu',weights_only=False));assert S.digest(loaded.state_dict())==ch
    result=dict(state='ENGINEERING_CONTRACTS_PASS',native_steps=0,optimizer_calls=0,Actor_updates=0,
        real_source_trajectories=8,full120_rows_reconstructed=rebuild_count,original_history_identity_max_error=identity,
        original_AR_all_bit_exact=True,synthetic_terminal_laws=dict(goal=success,collision_priority=collision,true_task_timeout=timeout,collector_censor_unknown=censored),
        SE2_body_query_max_error=err,conditional_AR_action_current_previous_mean_FD_max_error=max(fd),
        previous_mean_gradient_norm=float(gp.norm()),future_sample_trace=selected,
        contrastive_loss_with_unknown_censor_mask=float(loss.detach()),random_critic_parameters=sum(v.numel() for v in critic.parameters()),
        Actor_gradient_norm_with_random_critic=float(flat.norm()),Actor_gradient_SHA=S.digest(flat),
        Actor_original_hash=before,Actor_final_hash=S.digest(q.state_dict()),random_critic_hash=ch,critic_roundtrip_exact=True,
        details=details,semantic_warning=p['contract'],
        scientific_control_gate=False,critic_trained=False,online_collection_started=False,automatic_next_training=False)
    evidence=dict(conditional_action=ac.detach().numpy(),current_gradient=gm.numpy(),previous_gradient=gp.numpy(),
        sampled_actor_actions=new_action.detach().numpy(),random_critic_logits=logits.detach().numpy())
    return result,evidence


def preflight():
    p=read(HERE/'protocol.json');assert not any((OUT/x).exists() for x in ['status.json','launch.json','preflight.json']);verify(p)
    C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,optimizer_calls=0,all_inputs_SHA_match=True,
        reserved_global_bytes=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes'],protocol_sha256=C.sha(HERE/'protocol.json')))
    print('T305 zero-native/optimizer preregistration preflight PASS')


def run():
    p=read(HERE/'protocol.json');assert not (OUT/'status.json').exists();start=time.time()
    lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outside=S.used(ROOT/'stage/runs')-S.used(OUT)
    def guard():
        assert time.time()<start+p['wall_seconds']-5;resource(p)
        assert S.used(ROOT/'stage/runs')-S.used(OUT)==outside
    def alarm(*_):raise TimeoutError('T305 absolute engineering budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds']-2)
    state='FAILED';C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));C.save(OUT/'status.json',dict(state='STARTED',pid=os.getpid(),updated_epoch=time.time(),deadline_epoch=start+p['wall_seconds']))
    try:
        verify(p);r,a=calculate(p,guard);C.save(OUT/'contracts.json',r)
        buf=io.BytesIO();np.savez_compressed(buf,**a);b=buf.getvalue();assert len(b)<128*1024;(OUT/'evidence.npz').write_bytes(b);verify(p);state='COMPLETE'
    except BaseException:C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),native_steps=0,optimizer_calls=0))
    finally:signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    s=dict(state=state,pid=os.getpid(),native_steps=0,physics_steps=0,commands=0,optimizer_calls=0,Actor_updates=0,
        all_workers_stopped=True,elapsed_s=time.time()-start,main_epoch=start,deadline_epoch=start+p['wall_seconds'],
        scientific_control_gate=False,automatic_next_training=False,own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'))
    C.save(OUT/'summary.json',s);C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)


def audit():
    dest=DOC/'T305_AUDIT.json';assert not dest.exists();p=read(HERE/'protocol.json');s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Live/reused',pid))
    start=time.time();verify(p)
    def guard():assert time.time()<start+p['audit_seconds'];resource(p)
    r,a=calculate(p,guard);assert r==read(OUT/'contracts.json')
    with np.load(OUT/'evidence.npz') as z:
        assert set(z.files)==set(a) and all(np.array_equal(z[k],v) for k,v in a.items())
    verify(p);C.save(dest,dict(state='PASS',native_steps=0,optimizer_calls=0,Actor_updates=0,
        all_original_processes_absent=True,all_contracts_and_gradient_arrays_recomputed_exact=True,
        control_or_navigation_gate_passed=False,summary_sha256=C.sha(OUT/'summary.json'),elapsed_s=time.time()-start))
    print('T305 independent exact contract audit PASS; scientific control gate remains untested')


def launch():
    p=read(HERE/'protocol.json');assert not any((OUT/x).exists() for x in ['status.json','launch.json','summary.json']);assert read(OUT/'preflight.json')['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (OUT/'run.log').open('xb') as log:proc=subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=1s',str(p['wall_seconds'])+'s',str(ROOT/'run_stage.sh'),str(Path(__file__)),'run'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    C.save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_commit=receipt['remote_commit'],wall_seconds=p['wall_seconds'],single_launch=True))
    print(json.dumps(dict(pid=proc.pid)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','preflight','launch','run','audit']);args=parser.parse_args();globals()[args.mode]()

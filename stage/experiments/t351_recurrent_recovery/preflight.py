from t351_core import *
import multiprocessing as mp, contextlib, signal


def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T351 preflight1200s')));signal.alarm(1200)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();q=Actor(p);x=torch.tensor(arrays(p['preflight_sequence'])['states'])
    q.sample=True;q.rng=np.random.default_rng(p['seed']);q.tape=[];actions=[];h=None;locs=[]
    with torch.no_grad():
        for row in x:
            a,loc,h,_=q.step(row[None],h);actions.append(a[0].numpy());locs.append(loc[0])
        serial=torch.stack(locs);seq=q.sequence(x,q.base_actions(x));sequence_error=float((serial-seq).abs().max());assert sequence_error<1e-10
        assert float((q.sequence(x[:25],q.base_actions(x[:25]))-seq[:25]).abs().max())<1e-12
    ctx=mp.get_context('spawn');aa,bb=ctx.Pipe();counts=ctx.Array('q',2);os.environ['STAGE_PROCESS_CPU']='1'
    child=ctx.Process(target=worker,args=(bb,0,p,counts,True));child.start();bb.close()
    aa.send(dict(learned=q.learned(),states=x[:25].numpy(),seed=p['seed']));assert aa.poll(60);got=aa.recv();child.join(10);assert child.exitcode==0 and not child.is_alive()
    assert np.array_equal(got['actions'],np.asarray(actions[:25])) and digest(got['tape'])==digest(q.tape[:25]) and got['policy_hash']==q.phash()
    aa.close();os.environ['STAGE_PROCESS_CPU']='0'
    # Synthetic conditional histories test gradient/shadow mechanics only. They
    # are not valid new navigation trajectories or reported training samples.
    human=arrays(p['preflight_sequence']);items=[]
    for j in range(4):
        a=copy.deepcopy(human);qq=Actor(p).requires_grad_(False);qq.sample=True;qq.rng=np.random.default_rng(p['seed']+j);hh=None
        with torch.no_grad():
            for row in a['states']:_,_,hh,_=qq.step(torch.tensor(row[None]),hh)
        a.update({name:np.asarray([v[name] for v in qq.tape]) for name in qq.tape[0]});a['rewards']=a['rewards']*(1-.1*j)
        items.append(dict(row=dict(stochastic=True),arrays=a))
    old,_=C.dataset(dict(p,sequence_sources=[]),q,lambda:None)
    refs=(old,[old[j] for j in range(20)],[dict(x=x,b=q.base_actions(x),y=torch.tensor(human['actions'],dtype=torch.float64)) for _ in range(2)])
    pp=dict(p,epochs=2);live=Learner(p);shadow=Learner(p);before=live.state();imm=digest(before);shadow.load(before)
    a=update(pp,live,items,refs,0,lambda:None,lambda r:None);b=update(pp,shadow,items,refs,0,lambda:None,lambda r:None)
    assert a==b and live.hashes()==shadow.hashes() and digest(before)==imm
    assert sum(float(v.grad.abs().sum()) for v in live.actor.core.parameters())>0
    mature=live.state();oldhash=digest(mature);clone=Learner(p);clone.load(mature)
    with torch.no_grad():next(v for state in clone.opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)).add_(1.)
    assert digest(mature)==oldhash;clone.load(mature);assert clone.hashes()==live.hashes()
    with torch.no_grad():next(clone.actor.core.parameters()).add_(1.)
    clone.load(mature);assert clone.hashes()==live.hashes() and digest(mature)==oldhash
    # Both boundary and interior scores agree with finite differences.
    mu=torch.tensor([[-1.25,-.4],[.3,1.25]],dtype=torch.float64,requires_grad=True)
    raw=torch.tensor([[-2.,.2],[.7,2.]],dtype=torch.float64);atom=torch.tensor([[-1,0],[0,1]],dtype=torch.int8)
    grad=torch.autograd.grad(logp(mu,raw,atom,p).sum(),mu)[0];eps=1e-6
    fd=torch.empty_like(mu)
    for i in range(2):
        for j in range(2):
            plus=mu.detach().clone();minus=plus.clone();plus[i,j]+=eps;minus[i,j]-=eps
            fd[i,j]=(logp(plus,raw,atom,p).sum()-logp(minus,raw,atom,p).sum())/(2*eps)
    assert float((fd-grad).abs().max())<1e-6
    folder=OUT/'preflight';folder.mkdir();geometry={}
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        _,native,create,w,_=C.B.setup(folder/'native',p)
        try:
            for job in [*p['course_dev_jobs'],*p['original_jobs']]:
                env,obs,_=K.R.task_env(create,job)
                try:
                    obs=K.C.perturb_anchor(env,obs,job);geometry[job['id']]=C.B.geometry(env,obs,w,job['case']['step_cap']);assert env.step_count==0 and env.max_wheel==5.
                finally:w.events.pop(id(env),None);env.close()
        finally:w.close()
    save(OUT/'geometry.json',geometry)
    payload=copy.deepcopy(items[0]['arrays']);n=len(payload['states']);sizes={}
    for count,limit in [(400,p['full_bytes']),(234,p['old_full_bytes'])]:
        cap={}
        for name,v in payload.items():
            if name in ['initial_native_integration','final_native_integration','last_hidden_after']:cap[name]=v
            elif len(v)==n:cap[name]=np.zeros((count,*v.shape[1:]),dtype=v.dtype)
            elif len(v)==n+1:cap[name]=np.zeros((count+1,*v.shape[1:]),dtype=v.dtype)
            else:raise AssertionError(name)
        buf=io.BytesIO();np.savez(buf,**packed(cap));sizes[str(count)]=len(buf.getvalue());assert sizes[str(count)]+1024<=limit,(count,sizes[str(count)],limit)
    assert not any(counts[:]);save(OUT/'preflight.json',dict(state='PASS',geometry_commands=7,physics_steps=0,discarded_Adam=4,
        synthetic_histories_not_navigation=True,actual_spawn_draws_density_exact=True,sequence_step_location_max_error=sequence_error,
        causal_prefix=True,full_current_sequence_gradient_nonzero=True,mature_clone_reject_interrupt_exact=True,
        boundary_interior_score_finite_difference_max_error=float((fd-grad).abs().max()),uncompressed_bytes=sizes,
        raw_sigma_dtype='float64',elapsed_s=time.time()-start))
    print(json.dumps(dict(state='PREFLIGHT_PASS',geometry=7,native=0,discarded_Adam=4,sizes=sizes)))


if __name__=='__main__':main()

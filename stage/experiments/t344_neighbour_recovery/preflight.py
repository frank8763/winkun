from t344_core import *
import multiprocessing as mp

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    obj=Learner(p);initial=obj.state();x=arrays(p['demo_sources'][-1]['path'])['states'];x=torch.tensor(x[np.linspace(0,len(x)-1,16,dtype=np.int64)])
    shared=None;ctx=mp.get_context('spawn');a,b=ctx.Pipe();shared=ctx.Array('q',4)
    os.environ['STAGE_PROCESS_CPU']='1';proc=ctx.Process(target=worker,args=(b,0,p,shared,True));proc.start();b.close()
    z=np.random.default_rng(20261104344).normal(size=(len(x),2))
    try:
        a.send(dict(states=x.numpy(),z=z,policy_np=K.S.state_np(obj.actor)));assert a.poll(90);out=a.recv()
        with torch.no_grad():expected=obj.actor.draw(x,torch.tensor(z),False)
        assert out['policy_hash']==digest(obj.actor.state_dict()) and all(np.array_equal(v.numpy(),out['outputs'][k]) for k,v in expected.items())
    finally:proc.join(10);assert not proc.is_alive();a.close();os.environ['STAGE_PROCESS_CPU']='0'
    # Synthetic four-episode proposal tests exercise the real mixed-boundary
    # likelihood and support loss, not physical success or new navigation data.
    batch=[]
    for k in range(4):
        xx=x[k*4:k*4+4]
        with torch.no_grad():draw=obj.actor.draw(xx,torch.tensor(z[k*4:k*4+4]),False)
        flags=np.zeros((4,3),np.uint8);flags[-1,1 if k==0 else 0]=1
        counts=np.array([100,100,100,100],np.uint8);nextx=np.concatenate([xx.numpy()[1:],xx.numpy()[-1:]])
        aa=dict(states=xx.numpy(),next_states=nextx,rewards=np.array([-.01,.01,.01,1. if k==0 else -1.]),native_counts=counts,end_flags=flags,**{n:v.numpy() for n,v in draw.items() if n!='actions'})
        aa['actions']=draw['actions'].numpy();batch.append(dict(arrays=aa,row=dict(id='synthetic'+str(k),success=k==0,collision=k!=0,complete=True)))
    support=dict(arrays=dict(states=x.numpy(),actions=expected['actions'].numpy()),row=dict(id='synthetic_support',success=True,collision=False,complete=True))
    trace=[];models=[]
    for _ in range(2):
        v=Learner(p);v.load(initial);groups=targets(p,v.actor,support);trace.append(T.supported_update(p,v,batch,groups,0,lambda:None,lambda _:None,steps=2));models.append(v)
    assert trace[0]==trace[1] and models[0].hashes()==models[1].hashes()
    models[0].load(initial);assert models[0].hashes()=={k:digest(v) for k,v in initial.items()}
    try:
        with torch.no_grad():next(models[0].actor.correction.parameters()).add_(.2)
        raise InterruptedError('synthetic interruption')
    except InterruptedError:models[0].load(initial)
    assert models[0].hashes()=={k:digest(v) for k,v in initial.items()}
    folder=OUT/'preflight';folder.mkdir();geo={}
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        m,native,create,witness,reward=setup(folder/'native',p)
        try:
            torch.optim.Adam.step=lambda *_a,**_kw:(_ for _ in ()).throw(AssertionError('Zero geometry optimizer'))
            for j in p['course_dev_jobs']:
                env,obs,_=K.R.task_env(create,j)
                try:
                    obs=K.C.perturb_anchor(env,obs,j);g=geometry(env,obs,witness,40000);assert g['valid'] and g['max_wheel']==5.
                    geo[j['id']]=g
                finally:witness.events.pop(id(env),None);env.close()
        finally:witness.close()
    save(OUT/'geometry.json',geo)
    base=geo['wide_dev_0'];assert base['initial120_sha256']=='381401926732929781744bf135aa14f2eb4738a4ec99149d294c62fe192f08c1'
    save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,geometry_commands=5,discarded_synthetic_Adam=4,real_spawned_worker_draw_exact=True,
        no_synthetic_success_evidence=True,full_history_actual_origin_verified=True,whole_Actor_Adam_shadow_reject_interrupt_exact=True,
        protocol_sha256=K.C.sha(HERE/'protocol.json'),geometry_sha256=K.C.sha(OUT/'geometry.json'),synthetic_trace_sha256=digest(trace[0])))
    print('T344 preflight PASS0native/5geometry/4discardedAdam')

if __name__=='__main__':main()

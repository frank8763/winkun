from t340_core import *
from t340_worker import worker
import multiprocessing as mp

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=Learner(p,'capg');groups=demos(p,q.actor)
    assert [len(v['x']) for v in groups]==[320,168,76]
    initial=q.state();frozen=digest(q.actor.fixed.state_dict());x=torch.cat([g['x'] for g in groups])
    z=torch.tensor(np.random.default_rng(p['seed']).normal(size=(len(x),2)))
    with torch.no_grad():draw=q.actor.draw(x,z)
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']='1'
    proc=ctx.Process(target=worker,args=(child,0,p,None,True));proc.start();child.close();os.environ['STAGE_PROCESS_CPU']='0'
    parent.send(dict(arm='capg',policy_np=K.S.state_np(q.actor),x=x.numpy(),z=z.numpy()));assert parent.poll(90)
    answer=parent.recv();proc.join(10);assert proc.exitcode==0 and not proc.is_alive()
    assert answer['policy_hash']==digest(q.actor.state_dict()) and all(np.array_equal(v.numpy(),answer['outputs'][name]) for name,v in draw.items())
    q.opt=torch.optim.Adam(q.actor.trainable(),lr=p['warm_lr']);shadow=Learner(p,'capg');shadow.load(q.state());traces=[];events=[]
    def event(kind,replica,row):
        events.append(dict(kind=kind,replica=replica,**row));save(OUT/'preflight_optimizer_calls.json',events)
    for label,obj in [('live',q),('shadow',shadow)]:
        traces.append(warmup(p,obj.actor,obj.opt,demos(p,obj.actor),lambda:None,lambda row:event('warm',label,row),steps=1))
    assert traces[0]==traces[1] and q.hashes()==shadow.hashes()
    assert frozen==digest(q.actor.fixed.state_dict())
    # Synthetic density-consistent transitions test only: original states and
    # rewards are reused, draws are recomputed under the cloned warmed policy.
    # They are explicitly NOT navigation data and never enter main training.
    source=arrays(p['source_success_row']['path']);fake=[]
    for j in range(4):
        a=copy.deepcopy(source)
        with torch.no_grad():d=q.actor.draw(torch.tensor(a['states']),torch.tensor(np.random.default_rng(p['seed']+j).normal(size=(len(a['states']),2))))
        a.update({name:v.numpy() for name,v in d.items()});fake.append(dict(row=dict(id='synthetic'+str(j),complete=True,success=True,collision=False),arrays=a))
    q.opt=torch.optim.Adam(q.actor.trainable(),lr=p['actor_lr']);shadow=Learner(p,'capg');shadow.load(q.state());traces=[]
    for label,obj in [('live',q),('shadow',shadow)]:
        traces.append(supported_update(p,obj,fake,demos(p,obj.actor),0,lambda:None,lambda row:event('supported',label,row),steps=1))
    assert traces[0]==traces[1] and q.hashes()==shadow.hashes()
    blob=io.BytesIO();torch.save(q.state(),blob);size=len(gzip.compress(blob.getvalue(),mtime=0));assert size<p['model_bytes']
    q.load(initial);assert digest(q.state())==digest(initial)
    try:warmup(p,q.actor,q.opt,demos(p,q.actor),lambda:(_ for _ in ()).throw(InterruptedError('Pre-step interrupt')),lambda _r:None,steps=1)
    except InterruptedError:pass
    else:raise AssertionError('Interrupt not exercised')
    assert digest(q.state())==digest(initial)
    folder=OUT/'preflight';folder.mkdir();s=witness=None
    spec=importlib.util.spec_from_file_location('stage340_zero_native',ROOT/'stage/experiments/t331_policy_prefix_retry/recorder.py')
    M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M);course=read(p['course_protocol']);M.install(course)
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            native,create,witness,reward=M.setup(folder/'native')
            s=M.R.Session(course,native,create,witness,reward,lambda:None,lambda:(_ for _ in ()).throw(AssertionError('Zero physics')))
            assert np.array_equal(s.initial120,arrays(p['reference'])['states'][0]) and s.env.max_steps==40000
        finally:
            if s is not None:s.close()
            if witness is not None:witness.close()
    assert read(ROOT/'stage/runs/t339_success_to_control/preflight.json')['raw_episode_byte_bound']<p['training_episode_bytes']
    save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,geometry_commands=1,synthetic_discarded_Adam_calls=4,
        actual_worker_564_demo_states_all_terms_exact=True,demo_rows=[320,168,76],policy_prefix_human_labels=0,
        warm_and_supported_updates_exact_shadow=True,rollback_and_before_step_interrupt_exact=True,frozen_parent_unchanged=True,
        synthetic_model_bytes=size,raw_episode_byte_bound=read(ROOT/'stage/runs/t339_success_to_control/preflight.json')['raw_episode_byte_bound'],
        not_scientific_success_or_critic_validation=True,protocol_sha256=K.C.sha(HERE/'protocol.json')))
    verify(p);print('T340 preflight PASS0physics4discardedAdam realworker/warm/support/shadow/rollback')

if __name__=='__main__':
    try:main()
    except BaseException:
        save(OUT/'preflight_failure.json',dict(error=traceback.format_exc(),physics_steps=0));raise

"""Offline new-format/recipe prerequisites; live replay is mandatory before learning."""
from support import *
import copy
import multiprocessing as mp
def main():
    assert not (OUT/'contracts.json').exists()
    plan=json.loads((HERE/'protocol.json').read_text());torch.set_num_threads(1)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    state=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];p=policy(state);q=load_np(R.Policy(),state_np(p))
    assert digest(p.state_dict())==digest(q.state_dict())
    with np.load(plan['retention_targets']) as z:
        x=torch.tensor(z['states']);y=torch.tensor(z['actions'])
    with torch.no_grad():error=float((p(x)-y).abs().max())
    assert error==0
    # Storage encoding round trip is checked before a new navigation rollout.
    b=io.BytesIO();torch.save({'policy':p.state_dict()},b);data=gzip.compress(b.getvalue(),compresslevel=6,mtime=0)
    restored=torch.load(io.BytesIO(gzip.decompress(data)),map_location='cpu',weights_only=False)['policy']
    assert digest(restored)==digest(p.state_dict()) and len(data)<2200*1024
    live=torch.optim.Adam([t for t in p.parameters() if t.requires_grad],lr=plan['actor_lr'])
    shadow=copy.deepcopy(p);shadow_opt=torch.optim.Adam([t for t in shadow.parameters() if t.requires_grad],lr=plan['actor_lr'])
    assert digest(live.state_dict())==digest(shadow_opt.state_dict())
    # Native imports must succeed in the actual spawn process, before any
    # rollout, model export or optimizer step is permitted.
    ctx=mp.get_context('spawn');workers=[];pipes=[];ready=[]
    try:
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1)
            w=ctx.Process(target=worker,args=(b,i,time.time()+60));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0'
        deadline=time.monotonic()+45
        for pipe in pipes:
            assert pipe.poll(max(0,deadline-time.monotonic())),'Native startup timeout'
            message=pipe.recv();assert message.get('ready'),message;ready.append(message)
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        assert all(not w.is_alive() for w in workers)
    assert all(w.exitcode==0 for w in workers)
    # No diagnostic optimizer calls here; full state/optimizer/bonus reconstruction
    # and exact native observation replay are enforced at each actual batch.
    result={'state':'PASS','protocol_sha256':C.sha(HERE/'protocol.json'),'physics_steps':0,'optimizer_calls':0,
            'parent_action_max_error':error,'numpy_policy_hash_equal':True,'compressed_checkpoint_content_equal':True,
            'initial_compressed_policy_bytes':len(data),'live_and_shadow_initial_optimizer_equal':True,
            'live_replay_and_training_recipe_gates_mandatory':True}
    result['four_spawned_native_workers_ready']=ready
    result['all_preflight_workers_exited_cleanly']=True
    C.save(OUT/'contracts.json',result);print(json.dumps(result))
if __name__=='__main__':main()

from actuator_support import *
import multiprocessing as mp,fcntl,shutil,copy
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicateT249'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();wall=time.time()+plan['wall_seconds'];physics=commands=0;workers=[];pipes=[];extra=[];offset=[0];rows=[];reason='COMPLETE';dataset=None
    def guard(n=0):assert time.time()<wall and physics<=plan['max_physics'] and commands<=plan['max_commands'] and S.used(OUT)+n<plan['final_bytes'] and S.used(ROOT/'stage/runs')+n<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    def receive(p):
        while not p.poll(.5):assert time.time()<wall,'Deadline'
        r=p.recv();assert 'error' not in r,r;return r
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(plan['seed']);state=read(plan['checkpoint']);old=S.policy(state).eval().requires_grad_(False);p=policy(state)
        with np.load(plan['base_states']) as z:base=z['states'].copy();baseoff=z['offsets'].copy();ids=z['case_ids'].copy()
        rng=np.random.default_rng(plan['seed']);x=torch.tensor(base[::7]);e=torch.tensor(rng.uniform(-1,1,(len(x),4)),dtype=torch.float32);xx=torch.cat([x,e],-1)
        with torch.no_grad():modeerr=float((old(x)-p(xx)).abs().max());rawerr=float((C.ParentPolicy.forward(old,x)-C.ParentPolicy.forward(p,xx)).abs().max())
        assert modeerr==rawerr==0
        raw=C.ParentPolicy.forward(p,xx);loss=(raw*torch.tensor(rng.normal(size=raw.shape),dtype=torch.float32)).mean();loss.backward();ng=float(p.body.net[0].proprio.weight.grad.norm());bg=float(p.body.net[0].base.weight.grad.norm());assert ng>0 and bg>0 and all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in p.parameters())
        variant=copy.deepcopy(p)
        with torch.no_grad():variant.body.net[0].proprio.weight.normal_(0,.001);sensor_response=float((variant(torch.cat([x,e],-1))-variant(torch.cat([x,-e],-1))).abs().max());assert sensor_response>1e-7
        C.save(OUT/'contracts.json',dict(state='PASS',initial_mode_max_error=modeerr,initial_preclip_max_error=rawerr,added_parameters=2400,encoder_branch_gradient_norm=ng,base_gradient_norm=bg,nonzero_copy_sensor_response_max=sensor_response,optimizer_calls=0,physics_steps=0))
        ps=S.state_np(p);ph=S.digest(ps);ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';assert all(receive(p)['ready'] for p in pipes);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers]))
        for off in range(0,211,4):
            batch=plan['jobs'][off:off+4];guard();assert commands+len(batch)<=211 and physics+sum(j['case']['step_cap'] for j in batch)<=plan['max_physics']
            for pipe,j in zip(pipes,batch):pipe.send(dict({k:v for k,v in j.items() if k!='expected'},policy_np=ps,policy_hash=ph))
            rr=[receive(pipe) for pipe in pipes[:len(batch)]];commands+=len(rr);physics+=sum(i['row']['physics_steps'] for i in rr)
            for j,item in zip(batch,rr):
                r=item['row'];a=item['arrays'];expected=j['expected'];assert r['base_array_hashes']==expected['array_hashes'] and r['integration_sha256']==expected['integration_sha256'];k=len(rows);assert str(ids[k])==j['id'] and np.array_equal(a['states'][:,:113],base[baseoff[k]:baseoff[k+1]])
                extra.append(a['states'][:,113:]);offset.append(offset[-1]+len(a['states']));row=dict(id=j['id'],complete=r['complete'],success=r['success'],collision=r['collision'],physics_steps=r['physics_steps'],all_base_arrays_and_integration_exact=True,encoder_sha256=S.digest(extra[-1]));rows.append(row)
                with (OUT/'reconstruction.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            C.save(OUT/'status.json',dict(state='ZERO_BRANCH_PHYSICS_IDENTITY',cases=len(rows),commands=commands,physics_steps=physics,optimizer_calls=0,elapsed_s=time.monotonic()-start))
        assert len(rows)==211 and np.array_equal(offset,baseoff)
        buf=io.BytesIO();np.savez_compressed(buf,encoders=np.concatenate(extra),offsets=np.asarray(offset),case_ids=ids);b=buf.getvalue();guard(len(b));path=OUT/'actual_steering_context.npz';path.write_bytes(b)
        with np.load(path) as z:assert S.digest(z['encoders'])==S.digest(np.concatenate(extra))
        dataset=dict(path=str(path),sha256=C.sha(path),bytes=len(b),rows=offset[-1],cases=len(rows),encoder_abs_max=np.abs(np.concatenate(extra)).max(0).tolist(),base_states_sha256=C.sha(plan['base_states']))
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight_physics_upper_bound=4*23334))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    r=dict(state='PREFLIGHT_PASS' if reason=='COMPLETE' else 'FAILED',cases=len(rows),dataset=dataset,physics_steps=physics,commands=commands,optimizer_calls=0,all_workers_stopped=all(not w.is_alive() for w in workers),navigation_learning_claim=False,new_independent_tests=0,elapsed_s=time.monotonic()-start,own_bytes=S.used(OUT),runs_MiB=S.used(ROOT/'stage/runs')/1024**2);C.save(OUT/'summary.json',r);C.save(OUT/'status.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()

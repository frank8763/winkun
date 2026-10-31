import os
os.sched_setaffinity(0,{0})
import multiprocessing as mp,fcntl,shutil
from runtime import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t203_native_exploration'
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists()
    plan=json.loads((HERE/'protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert C.sha(p)==h,p
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p=Policy().eval().requires_grad_(False);p.load_state_dict(torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'])
    path=OUT/'parent_heads.npz';save_heads(path,p);deadline=time.time()+plan['wall_seconds'];ctx=mp.get_context('spawn');pipes=[];workers=[];spent=0;rows=[]
    C.save(OUT/'status.json',{'state':'PREFLIGHT','optimizer_updates':0})
    def receive(pipe):
        assert pipe.poll(max(.1,min(90,deadline-time.time())));r=pipe.recv();assert 'error' not in r,r;return r
    def batch(jobs):
        nonlocal spent
        result=[]
        for off in range(0,len(jobs),8):
            sub=jobs[off:off+8];assert spent+len(sub)*23334<=plan['max_physics']
            for pipe,j in zip(pipes,sub):pipe.send(('rollout',j))
            rr=[receive(pipe) for pipe in pipes[:len(sub)]];spent+=sum(r['physics_steps'] for r in rr);result+=rr
        return result
    try:
        # Independent normalization and score-integral checks, including edge means.
        grid=torch.linspace(-1,1,20001,dtype=torch.float64);checks=[]
        for raw in [-3.,-1.2,0.,1.2,3.]:
            v=torch.tensor(raw,dtype=torch.float64,requires_grad=True);m=center(v);sigma=SIGMA[0];z=torch.special.ndtr((1-m)/sigma)-torch.special.ndtr((-1-m)/sigma);density=torch.exp(-.5*((grid-m)/sigma).square())/(sigma*math.sqrt(2*math.pi)*z);mass=torch.trapezoid(density,grid);g=torch.autograd.grad(mass,v)[0]
            assert abs(float(mass)-1)<1e-7 and abs(float(g))<1e-6;checks.append({'raw':raw,'mass':float(mass.detach()),'integrated_score':float(g)})
        C.save(OUT/'distribution_checks.json',{'state':'PASS','checks':checks,'optimizer_updates':0,'physics_steps':0})
        for i in range(8):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,str(OUT),plan['checkpoint'],deadline));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';C.save(OUT/'workers.json',{'parent':os.getpid(),'pids':[w.pid for w in workers]})
        tasks=[]
        for i,delta in enumerate([[0,0,0]]+plan['deltas']):
            pipes[0].send(('make_task',{'id':f'near_{i}','case':plan['case'],'near_snapshot':plan['near_snapshot'],'delta':delta}));r=receive(pipes[0]);assert r['valid'];tasks.append(r)
        def job(i,mode):return {'id':f'{i}_{mode}','task':tasks[i]['id'],'case':plan['case'],'exact_snapshot':tasks[i]['snapshot'],'heads':str(path),'seed':plan['seed']+i,'mode':mode,'initial_pose':tasks[i]['pose'],'role':'preflight'}
        source=batch([job(0,'deterministic')]);rows+=source;assert source[0]['complete'] and source[0]['success'] and not source[0]['collision'],'Source gate failed'
        rows+=batch([job(i,mode) for i in range(1,6) for mode in ['deterministic','legacy_gaussian','physical_gaussian']])
        assert all(r['complete'] for r in rows)
        for r in rows:
            with (OUT/'episodes.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        # Prove recoverability before compacting only the two witnesses.
        replay=[]
        for mode in ['legacy_gaussian','physical_gaussian']:
            r=next(r for r in rows if r['mode']==mode);assert spent+r['physics_steps']<=plan['max_physics'];pipes[0].send(('replay',r));v=receive(pipes[0]);spent+=v['physics_steps'];assert v['verified'];replay.append(v)
        counts={};coverage={}
        for mode in ['deterministic','legacy_gaussian','physical_gaussian']:
            subset=[r for r in rows if r['mode']==mode and r['task']!='near_0'];counts[mode]={'n':len(subset),'GOAL':sum(r['success'] and not r['collision'] for r in subset),'CONTACT':sum(r['collision'] for r in subset),'TIMEOUT':sum(r['timeout'] for r in subset)}
            allactions=[];clipactions=[]
            for r in subset:
                with np.load(r['path']) as z:states=torch.tensor(z['states']);a=z['actions'].copy()
                with torch.no_grad():mask=p.latent_mean(states)[:,0].numpy()>=7.2476
                allactions.append(a);clipactions.append(a[mask])
            a=np.concatenate(allactions);b=np.concatenate(clipactions)
            coverage[mode]={'decisions':len(a),'clip_decisions':len(b),'clip_throttle_below_0_9_fraction':float(np.mean(b[:,0]<.9)) if len(b) else None,'throttle_quantiles':np.quantile(a[:,0],[0,.1,.5,.9,1]).tolist()}
        gate=counts['physical_gaussian']['GOAL']>=3 and coverage['physical_gaussian']['clip_decisions']>0 and coverage['physical_gaussian']['clip_throttle_below_0_9_fraction']>=.05
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<plan['max_bytes']
        result={'state':'COMPLETE','training_gate_passed':gate,'episodes':len(rows),'counts':counts,'coverage':coverage,'replay_checks':replay,'physics_steps':spent,'training_physics_steps':0,'optimizer_updates':0,'automatic_promotion':False,'scope':plan['contract']}
        C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result,indent=2),flush=True)
    except BaseException:
        C.save(OUT/'partial_episodes.json',rows)
        C.save(OUT/'failure.json',{'state':'FAILED','physics_steps':spent,'completed_episodes':len(rows),'optimizer_updates':0,'traceback':traceback.format_exc()});raise
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.terminate();w.join(3)
if __name__=='__main__':main()

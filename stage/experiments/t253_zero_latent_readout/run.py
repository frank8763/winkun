from readout_support import *
import multiprocessing as mp,fcntl,shutil
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicateT253'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();wall=time.time()+plan['wall_seconds'];commands=physics=0;workers=[];pipes=[];screen=[];dev=[];hold=[];reason='COMPLETE';control_exact=False;screen_exact=False;overlap_exact=False
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    def guard(pending=0):assert time.time()<wall and commands<=plan['max_commands'] and physics<=plan['max_physics'] and S.used(OUT)+pending<plan['final_bytes'] and S.used(ROOT/'stage/runs')+pending<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    def receive(p):
        while not p.poll(.5):assert time.time()<wall,'Deadline'
        r=p.recv();assert 'error' not in r,r;return r
    def evaluate(jobs,prefix,readout):
        nonlocal commands,physics
        rr=[]
        for off in range(0,len(jobs),4):
            jj=[dict(j,id=prefix+'_'+j['id'],source_reset_id=j['id'],hold_steps=100,readout=readout,policy_np=ps,policy_hash=ph) for j in jobs[off:off+4]];guard();assert commands+len(jj)<=plan['max_commands'] and physics+sum(j['case']['step_cap'] for j in jj)<=plan['max_physics']
            for pipe,j in zip(pipes,jj):pipe.send(j)
            batch=[receive(p) for p in pipes[:len(jj)]];commands+=len(batch);physics+=sum(i['row']['physics_steps'] for i in batch)
            for item in batch:
                a=item['arrays'];compact={k:v for k,v in a.items() if k not in ['states','next_states']};buf=io.BytesIO();np.savez_compressed(buf,**compact);b=buf.getvalue();assert len(b)<64*1024;guard(len(b));path=OUT/(item['row']['id']+'.npz');assert not path.exists();path.write_bytes(b);r=item['row'];r.update(path=str(path),sha256=C.sha(path));rr.append(r)
                with (OUT/(prefix+'.jsonl')).open('a') as f:f.write(json.dumps(r)+'\n')
            C.save(OUT/'status.json',dict(state=prefix,commands=commands,physics_steps=physics,cases=len(rr),optimizer_calls=0,elapsed_s=time.monotonic()-start));assert all(i['row']['complete'] for i in batch)
        return rr
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        import math
        grid=np.asarray([[-1.7,-1.7],[-1.1,-1.1],[-1.,-1.],[0.,0.],[.95,.95],[1.,1.],[1.1,1.1],[1.7,1.7]])
        sigma=np.asarray(AR.SIGMA)
        def independent_median(mean,sd):
            sign=1 if mean>=0 else -1;mean=abs(mean);cdf=lambda a:.5*math.erfc(-(a-mean)/sd/math.sqrt(2));target=.5*(cdf(-1)+cdf(1));assert target>0
            low,high=-1.,1.
            for _ in range(100):
                mid=(low+high)/2
                if cdf(mid)<target:low=mid
                else:high=mid
            return sign*(low+high)/2
        reference=np.asarray([[independent_median(m,s) for m,s in zip(row,sigma)] for row in grid]);actual=AR.sample(torch.tensor(grid),torch.zeros_like(torch.tensor(grid))).numpy();error=float(np.abs(actual-reference).max());assert error<1e-7
        C.save(OUT/'readout_math.json',dict(state='PASS',independent_erfc_bisection_median_max_error=error,physical_steps=0,optimizer_calls=0,additional_dependencies_installed=False,policy_noise_zero=True,readout='sameinverseCDFasRL,stationarylatent0,notclippedmode'))
        state=read(plan['checkpoint']);ph=S.digest(state);ps={k:v.numpy().copy() for k,v in state.items()};ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';assert all(receive(p)['ready'] for p in pipes);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers]))
        prior=list(map(json.loads,(ROOT/'stage/runs/t239_ar_policy_gradient/development.jsonl').read_text().splitlines()));control=evaluate(plan['control_jobs'],'control_mode','clipped_mode')
        for r in control:
            old=next(q for q in prior if q['job']['source_reset_id']==r['source_reset_id']);assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
        control_exact=True;screen=evaluate(plan['screen_jobs'],'screen_zero','zero_latent');repeat=evaluate(plan['screen_jobs'],'repeat_zero','zero_latent')
        for a,b in zip(screen,repeat):assert a['array_hashes']==b['array_hashes'] and a['integration_sha256']==b['integration_sha256']
        screen_exact=True
        if all(safe(r) for r in screen):
            dev=evaluate(plan['development_jobs'],'development_zero','zero_latent')
            for r in screen:
                q=next(q for q in dev if q['source_reset_id']==r['source_reset_id']);assert r['array_hashes']==q['array_hashes'] and r['integration_sha256']==q['integration_sha256']
            overlap_exact=True
            if all(safe(r) for r in dev):hold=evaluate(plan['heldout_jobs'],'heldout_zero','zero_latent')
        else:reason='ZERO_LATENT_SCREEN_FAILED_NO_FULL_DEVELOPMENT_OR_HELDOUT'
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight_upper_physics=4*23334))
    finally:
        for p in pipes:
            try:p.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    gate=control_exact and screen_exact and overlap_exact and len(dev)==211 and all(safe(r) for r in dev) and len(hold)==64 and all(safe(r) for r in hold)
    result=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,clipped_mode_control_exact=control_exact,zero_latent_screen_n=len(screen),zero_latent_screen_safe=sum(safe(r) for r in screen),zero_latent_screen_failures=[r['source_reset_id'] for r in screen if not safe(r)],screen_independent_replay_exact=screen_exact,development_n=len(dev),development_safe=sum(safe(r) for r in dev),heldout_n=len(hold),heldout_safe=sum(safe(r) for r in hold),known_invalid_starts=5,stable_valid_local_3m_4m_5m_gate=gate,checkpoint=plan['checkpoint'],policy_hash=locals().get('ph'),hold_steps=100,readout="zero_latent",physics_steps=physics,commands=commands,optimizer_calls=0,all_workers_stopped=all(not w.is_alive() for w in workers),elapsed_s=time.monotonic()-start,own_bytes=S.used(OUT),runs_MiB=S.used(ROOT/'stage/runs')/1024**2);C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()

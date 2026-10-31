from pathlib import Path
import sys,os,json,time,io,gzip,fcntl,shutil,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t247_recovery_branches'
sys.path.insert(0,str(ROOT/'stage/experiments/t229_valid_start_acceptance'))
import valid_support as V
S,C,R,np,torch=V.S,V.C,V.R,V.np,V.torch
def read(path):return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)['policy']
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicateT247'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();wall=time.time()+plan['wall_seconds'];physics=commands=0;rows=[];reason='COMPLETE';gate=False;env=None
    def guard(pending=0):assert time.time()<wall and commands<=plan['max_commands'] and physics<=plan['max_physics'] and S.used(OUT)+pending<plan['final_bytes'] and S.used(ROOT/'stage/runs')+pending<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    def savearrays(name,a):
        assert np.array_equal(a['states'][1:],a['next_states'][:-1]);c={k:v for k,v in a.items() if k!='next_states'};c['final_next_state']=a['next_states'][-1];b=io.BytesIO();np.savez_compressed(b,**c);d=b.getvalue();assert len(d)<192*1024;guard(len(d));p=OUT/(name+'.npz');assert not p.exists();p.write_bytes(d);return dict(path=str(p),sha256=C.sha(p))
    def simulate(p,label,begin=0,capture=False):
        nonlocal env,physics,commands
        guard();assert commands+1<=plan['max_commands'] and physics+plan['job']['case']['step_cap']-begin<=plan['max_physics'];commands+=1
        env,obs,_=R.task_env(create,plan['job']);obs=C.perturb_anchor(env,obs,plan['job']);cap=plan['job']['case']['step_cap']
        if begin:
            snap=C.read_snapshot(OUT/f'anchor_{begin}.npz');obs=C.restore(env,snap);assert np.array_equal(R.integration(env),snap['physics']) and env.step_count==begin
        scene=C.model_hash(env);goal=env.goal_xy.copy();steps=begin;term=trunc=False;info={};rec=[]
        while not(term or trunc) and steps<cap:
            guard()
            if capture and steps in plan['start_steps']:
                path=OUT/f'anchor_{steps}.npz';assert not path.exists();C.write_snapshot(path,C.capture(env,obs));assert path.stat().st_size<64*1024
            state=C.observation(obs,1-steps/cap);pose=np.asarray(env._pose()[:3]);d0=float(env._goal_state()[1]);before=steps
            with torch.no_grad():action=p(torch.tensor(state[None])).numpy()[0].astype(np.float32)
            for _ in range(min(100,cap-steps)):
                obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;physics+=1
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);discount=R.GAMMA**n;terminal=term or trunc;event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc);reward=-.000002*(1-discount)/(1-R.GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            rec.append((state,action,np.zeros(2,dtype=np.float32),0.,n,C.observation(obs,1-steps/cap),pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward))
        names=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards'];a={k:np.asarray([r[i] for r in rec]) for i,k in enumerate(names)};a['native_counts']=a['native_counts'].astype(np.uint8);a['end_flags']=a['end_flags'].astype(np.uint8);assert C.model_hash(env)==scene and np.array_equal(goal,env.goal_xy)
        row=dict(id=label,start_step=begin,complete=bool(term or trunc),success=bool(info['reach_goal']),collision=bool(info['collision']),timeout=bool(trunc),physics_steps=steps-begin,absolute_end_step=steps,remaining_m=d1,array_hashes={k:S.digest(v) for k,v in a.items()},integration_sha256=R.arrays_hash(R.integration(env)));env.close();env=None;rows.append(row)
        with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        C.save(OUT/'status.json',dict(state='FROZEN_RECOVERY_DIAGNOSTIC',physics_steps=physics,commands=commands,cases=len(rows),optimizer_calls=0,elapsed_s=time.monotonic()-start));return row,a
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);student=S.policy(read(plan['checkpoint'])).eval().requires_grad_(False);recover=S.policy(read(plan['recovery_checkpoint'])).eval().requires_grad_(False);native,create=S.native_setup(OUT/'native')
        first,a=simulate(student,'source_first',capture=True);assert first['array_hashes']==plan['expected_source']['array_hashes'] and first['integration_sha256']==plan['expected_source']['integration_sha256'];savearrays('source_T239_failure24',a)
        second,b=simulate(student,'source_second');assert first['array_hashes']==second['array_hashes'] and first['integration_sha256']==second['integration_sha256'] and all(np.array_equal(v,b[k]) for k,v in a.items())
        recovery=[]
        for begin in plan['start_steps']:
            control,ca=simulate(student,f'control_{begin}',begin);assert control['integration_sha256']==first['integration_sha256'] and all(np.array_equal(v,a[k][begin//100:]) for k,v in ca.items())
            one,ra=simulate(recover,f'recovery_{begin}_first',begin);two,rb=simulate(recover,f'recovery_{begin}_second',begin);assert one['array_hashes']==two['array_hashes'] and one['integration_sha256']==two['integration_sha256'] and all(np.array_equal(v,rb[k]) for k,v in ra.items());one.update(savearrays(f'recovery_{begin}',ra));recovery.append(one)
        gate=all(r['complete'] and r['success'] and not r['collision'] for r in recovery) and len(rows)==14
        # Labels queried on actual student states, up to latest declared branch.
        x=torch.tensor(a['states'][:plan['start_steps'][-1]//100+1]);
        with torch.no_grad():labels=C.ParentPolicy.forward(recover,x).numpy()
        out=io.BytesIO();np.savez_compressed(out,states=x.numpy(),preclip_targets=labels);data=out.getvalue();guard(len(data));(OUT/'student_induced_correction_queries.npz').write_bytes(data)
        C.save(OUT/'recovery_gate.json',dict(recovery_source_gate=gate,recovery_cases=recovery,all_controls_exact=True,query_labels_are_not_per_state_safety_certificates=True))
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands))
    finally:
        if env is not None:env.close()
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    r=dict(state=reason,physics_steps=physics,commands=commands,optimizer_calls=0,recovery_source_gate=gate,recovery_results=[r for r in rows if r['id'].startswith('recovery_')],all_native_envs_closed=True,all_workers_stopped=True,end_to_end_deployment_success_claim=False,elapsed_s=time.monotonic()-start,own_bytes=S.used(OUT),runs_MiB=S.used(ROOT/'stage/runs')/1024**2);C.save(OUT/'summary.json',r);C.save(OUT/'status.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()

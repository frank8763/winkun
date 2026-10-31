"""Exact frozen failure reproduction and offline closed-loop error audit."""
from pathlib import Path
import sys,os,json,time,io,gzip,fcntl,shutil,traceback,multiprocessing as mp
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t236_closed_loop_audit'
sys.path.insert(0,str(ROOT/'stage/experiments/t229_valid_start_acceptance'))
import valid_support as V
S,C,R,np,torch=V.S,V.C,V.R,V.np,V.torch
def read(path):return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)['policy']
def worker(pipe,index,deadline):V.OUT=OUT;V.worker(pipe,index,deadline)
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicateT236'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();wall=time.time()+plan['wall_seconds'];physics=commands=0;workers=[];pipes=[];results=[];reason='COMPLETE'
    def save(name,obj):C.save(OUT/name,obj)
    def receive(p):
        while not p.poll(.5):assert time.time()<wall,'Deadline'
        r=p.recv();assert 'error' not in r,r;return r
    def guard(pending=0):
        assert time.time()<wall and commands<=plan['max_commands'] and physics<=plan['max_physics']
        assert S.used(OUT)+pending<plan['final_bytes'] and S.used(ROOT/'stage/runs')+pending<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);state=read(plan['checkpoint']);ps={k:v.numpy().copy() for k,v in state.items()};ph=S.digest(state);student=S.policy(state).eval().requires_grad_(False);other=S.policy(read(plan['complementary_checkpoint'])).eval().requires_grad_(False)
        ctx=mp.get_context('spawn')
        for i in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';assert all(receive(p)['ready'] for p in pipes);save('workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers]))
        jobs=[dict(j,policy_np=ps,policy_hash=ph) for j in plan['jobs']]
        save('status.json',dict(state='FROZEN_REPRODUCTION',physics_steps=0,optimizer_calls=0))
        for p,j in zip(pipes,jobs):p.send(('evaluate',j))
        first=[receive(p) for p in pipes];commands+=2;physics+=sum(r['row']['physics_steps'] for r in first)
        for j,item in zip(jobs,first):
            a=item['arrays'];assert np.array_equal(a['states'][1:],a['next_states'][:-1]);b={k:v for k,v in a.items() if k!='next_states'};b['final_next_state']=a['next_states'][-1]
            buf=io.BytesIO();np.savez_compressed(buf,**b);data=buf.getvalue();assert len(data)<192*1024;guard(len(data));path=OUT/(j['id']+'.npz');path.write_bytes(data)
            prior=next(r for r in plan['prior_failures'] if r['id']==j['id']);assert item['row']['array_hashes']==prior['array_hashes'] and item['row']['integration_sha256']==prior['integration_sha256']
            source=next(r for r in plan['successful_sources'] if r['source_reset_id']==j['id'])
            with np.load(source['path']) as z:t={k:z[k].copy() for k in z.files}
            with torch.no_grad():
                sx=torch.tensor(a['states']);tx=torch.tensor(t['states']);sd=(student(sx)-other(sx)).numpy();td=(student(tx)-other(tx)).numpy()
            n=min(len(a['poses']),len(t['poses']));pos=np.linalg.norm(a['poses'][:n,:2]-t['poses'][:n,:2],axis=1);turn=np.abs(np.arctan2(np.sin(a['poses'][:n,2]-t['poses'][:n,2]),np.cos(a['poses'][:n,2]-t['poses'][:n,2])))
            first_cm=np.flatnonzero(pos>.01);first_5cm=np.flatnonzero(pos>.05)
            metrics=dict(id=j['id'],trace=str(path),sha256=C.sha(path),complete=item['row']['complete'],collision=item['row']['collision'],physics_steps=item['row']['physics_steps'],first_1cm_divergence_s=float(first_cm[0]*.2) if len(first_cm) else None,first_5cm_divergence_s=float(first_5cm[0]*.2) if len(first_5cm) else None,position_difference_before_student_contact_m=float(pos[-1]),yaw_difference_before_student_contact_deg=float(np.degrees(turn[-1])),student_states_action_RMSE=float(np.sqrt((sd**2).mean())),source_states_action_RMSE=float(np.sqrt((td**2).mean())),student_states_action_max=float(np.abs(sd).max()),source_states_action_max=float(np.abs(td).max()),student_collision_s=.002*item['row']['physics_steps'],source_GOAL_s=.002*source['physics_steps'],original_failure_exact=True)
            curve=[dict(t_s=.2*k,pose_delta_m=float(pos[k]),yaw_delta_deg=float(np.degrees(turn[k])),student_action=a['actions'][k].tolist(),same_student_state_action_delta=sd[k].tolist(),source_state_action_delta=td[k].tolist()) for k in range(n)]
            save(j['id']+'_error_curve.json',curve);results.append(metrics)
        for p,j in zip(pipes,jobs):p.send(('evaluate',j))
        second=[receive(p) for p in pipes];commands+=2;physics+=sum(r['row']['physics_steps'] for r in second)
        for a,b in zip(first,second):assert a['row']==b['row'] and all(np.array_equal(v,b['arrays'][k]) for k,v in a['arrays'].items())
        guard()
    except BaseException:reason='FAILED';save('failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands))
    finally:
        for p in pipes:
            try:p.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    summary=dict(state=reason,results=results,physics_steps=physics,commands=commands,optimizer_calls=0,all_workers_stopped=all(not w.is_alive() for w in workers),causal_attribution_proven=False,elapsed_s=time.monotonic()-start,runs_MiB=S.used(ROOT/'stage/runs')/1024**2)
    save('summary.json',summary);save('status.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

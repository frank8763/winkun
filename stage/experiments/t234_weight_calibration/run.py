from calibration_support import *
import multiprocessing as mp,fcntl,shutil
class Stop(Exception):pass
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicateT234'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();deadline=start+plan['wall_seconds'];wall=time.time()+plan['wall_seconds'];physics=commands=0;workers=[];pipes=[];calibration=[];rows=[];hold=[];reason=None;selected=None;export=None;ph=None;ps=None;alpha=None
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    def status(state):C.save(OUT/'status.json',dict(state=state,physics_steps=physics,commands=commands,alpha=alpha,selected_alpha=selected,development_cases=len(rows),heldout_cases=len(hold),optimizer_calls=0,elapsed_s=time.monotonic()-start))
    def guard(n=0,count=0,pending=0,reserve=0):
        if time.monotonic()+reserve>=deadline:raise Stop('WALL_RESERVE')
        if commands+n>plan['max_commands'] or physics+count>plan['max_physics']:raise Stop('COUNT_RESERVE')
        if used(OUT)+pending>=plan['final_bytes'] or used(ROOT/'stage/runs')+pending>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<40*1024**3:raise Stop('STORAGE_RESERVE')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.monotonic()>deadline:raise Stop('RESPONSE_DEADLINE')
        r=pipe.recv();assert 'error' not in r,r;return r
    def append(name,obj):
        with (OUT/name).open('a') as f:f.write(json.dumps(obj)+'\n')
    def evaluate(jobs,name):
        nonlocal physics,commands
        result=[]
        for off in range(0,len(jobs),4):
            batch=jobs[off:off+4];guard(len(batch),sum(j['case']['step_cap'] for j in batch),16384)
            for pipe,j in zip(pipes,batch):pipe.send(('evaluate',dict(j,policy_np=ps,policy_hash=ph,arm='frozenT234',seed=0,mode='deterministic')))
            replies=[receive(p) for p in pipes[:len(batch)]];commands+=len(replies);physics+=sum(i['row']['physics_steps'] for i in replies)
            for j,item in zip(batch,replies):
                r=item['row'];r.update(case=j['case'],alpha=alpha,delta_x_m=j['delta_x_m'],delta_y_m=j['delta_y_m'],delta_yaw_deg=j['delta_yaw_deg']);append(name,r);result.append(r)
                if name=='development.jsonl':rows.append(r)
                elif name=='heldout.jsonl':hold.append(r)
            status('FROZEN_'+name)
            if not all(i['row']['complete'] for i in replies):raise Stop('CENSORED_CASE')
        return result
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);a=read(plan['checkpoint']);b=read(plan['old_checkpoint']);assert set(a)==set(b) and all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype for k in a)
        assert digest(interpolate(a,b,0))==digest(a) and digest(interpolate(a,b,1))==digest(b)
        x=np.load(ROOT/'stage/runs/t196_left_exploration/retention_targets.npz')['states'];x=torch.tensor(x[:512]);p0=S.policy(a).eval();p1=S.policy(b).eval()
        with torch.no_grad():assert torch.equal(S.policy(interpolate(a,b,0))(x),p0(x)) and torch.equal(S.policy(interpolate(a,b,1))(x),p1(x))
        C.save(OUT/'contracts.json',dict(state='PASS',alpha0_exact_state_and_actions=True,alpha1_exact_state_and_actions=True,common_parameter_count=sum(v.numel() for v in a.values()),optimizer_calls=0,physics_steps=0))
        ctx=mp.get_context('spawn')
        for i in range(4):
            pa,pb=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(pb,i,wall));w.start();pb.close();pipes.append(pa);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(p) for p in pipes];assert all(r.get('ready') for r in ready);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers]))
        for alpha in plan['alphas']:
            guard(pending=2500000,reserve=plan['final_reserve_seconds']);state=interpolate(a,b,alpha);assert all(torch.isfinite(v).all() for v in state.values());ph=digest(state);ps={k:v.numpy().copy() for k,v in state.items()}
            critical=evaluate(plan['calibration_critical_jobs'],'calibration.jsonl');other=[]
            if all(safe(r) for r in critical):other=evaluate(plan['calibration_other_jobs'],'calibration.jsonl')
            passed=len(other)==len(plan['calibration_other_jobs']) and all(safe(r) for r in critical+other)
            rec=dict(alpha=alpha,policy_hash=ph,critical_n=len(critical),critical_safe=sum(safe(r) for r in critical),other_n=len(other),other_safe=sum(safe(r) for r in other),passed=passed);calibration.append(rec);C.save(OUT/'calibration_summary.json',calibration)
            if passed:selected=alpha;break
        if selected is None:reason='FIXED_ALPHA_GRID_FAILED'
        else:
            export=S.save_gzip(OUT/'selected_Actor.pt.gz',dict(policy=state,alpha=selected,source='T234boundedposttrainingweightcalibration',protocol_sha256=C.sha(HERE/'protocol.json'),optimizer_calls=0),lambda n:guard(pending=n))
            saved=read(export['path']);assert digest(saved)==ph;ps={k:v.numpy().copy() for k,v in saved.items()};C.save(OUT/'selection.json',dict(alpha=selected,policy_hash=ph,export=export,rule='smallestpredeclaredalpha passingallcalibrationjobs; no furtherselectionafterfinaleval'))
            evaluate(plan['development_jobs'],'development.jsonl')
            if len(rows)==211 and all(safe(r) for r in rows):evaluate(plan['heldout_jobs'],'heldout.jsonl');reason='DEVELOPMENT_AND_HELDOUT_COMPLETE'
            else:reason='FINAL_DEVELOPMENT_FAILED_HELDOUT_NOT_RUN'
    except Stop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight_upper_physics=4*23334))
    finally:
        for p in pipes:
            try:p.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        assert all(not w.is_alive() for w in workers)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    scores={}
    for i in range(8):
        rr=[r for r in rows if r['group']==f'task{i}'];hh=[r for r in hold if r['group']==f'task{i}'];scores[f'task{i}']=dict(development_n=len(rr),development_safe=sum(safe(r) for r in rr),CONTACT=sum(r['collision'] for r in rr),timeout=sum(r['timeout'] for r in rr),heldout_n=len(hh),heldout_safe=sum(safe(r) for r in hh))
    centers=[r for r in rows if r['delta_x_m']==r['delta_y_m']==r['delta_yaw_deg']==0.];gate=selected is not None and len(rows)==211 and all(safe(r) for r in rows) and len(hold)==64 and all(safe(r) for r in hold) and len(centers)==8 and all(safe(r) for r in centers)
    final_by_id={r['id']:r for r in rows};selected_calibration=[r for r in map(json.loads,(OUT/'calibration.jsonl').read_text().splitlines()) if selected is not None and r['alpha']==selected] if (OUT/'calibration.jsonl').exists() else []
    overlap_exact=(len(selected_calibration)==len(plan['calibration_critical_jobs'])+len(plan['calibration_other_jobs']) and all(r['id'] in final_by_id and r['physics_steps']==final_by_id[r['id']]['physics_steps'] and r['array_hashes']==final_by_id[r['id']]['array_hashes'] and r['integration_sha256']==final_by_id[r['id']]['integration_sha256'] for r in selected_calibration))
    gate=gate and overlap_exact
    summary=dict(calibration_final_overlap_exact=overlap_exact,state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,calibration=calibration,selected_alpha=selected,policy_hash=ph,export=export,scores=scores,original_centers_n=len(centers),original_centers_safe=sum(safe(r) for r in centers),development_n=len(rows),development_safe=sum(safe(r) for r in rows),development_invalid_starts_predeclared=5,heldout_n=len(hold),heldout_safe=sum(safe(r) for r in hold),optimizer_calls=0,physics_steps=physics,commands=commands,stable_valid_local_3m_4m_5m_gate=gate,original_T228_strict_gate=False,multimap_generalization_claim=False,automatic_promotion=False,all_workers_stopped=True,elapsed_s=time.monotonic()-start,own_bytes=used(OUT),runs_MiB=used(ROOT/'stage/runs')/1024**2)
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)
    assert physics<=plan['max_physics'] and commands<=plan['max_commands'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<plan['global_bytes']
if __name__=='__main__':main()

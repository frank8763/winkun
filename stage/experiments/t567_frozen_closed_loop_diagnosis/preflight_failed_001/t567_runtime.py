"""One bounded smooth consolidation, qualified autonomous panel, then probes."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t567_frozen_closed_loop_diagnosis';DOC=ROOT/'stage/docs/day_20261115/t567'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'stage/experiments/t562_full_history_representation'));import t562_actor as L
sys.path.insert(0,str(ROOT/'stage/experiments/t555_adam_direction_control'));import t555_runtime as Old
Base,F,G,E,P,np,torch=Old.Base,Old.F,Old.G,Old.E,Old.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest

def used(f):return sum(x.stat().st_size for x in Path(f).rglob('*') if x.is_file() and not x.is_symlink())
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
def configuration():
    root=read(OUT/'protocol.json');p=read(root['actor_configuration']);p.update(root);p['out']=str(OUT);return p
def setup():
    Old.OUT=OUT;Old.HERE=HERE;Old.verify=verify;Old.setup(OUT);Base.ACTIVE=OUT;P.frozen=L.frozen;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def worker(pipe,index,p,counts):
    G.T.Session=F.FieldSession;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=Old.Prior.NORMAL;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(1):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready);save(OUT/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
        except BaseException:self.close();raise
def diagnose(p,r,a):
    baseline=p['baseline_rows'][r['job_id']];ref=E.N.S.arrays(baseline['path']);assert digest(a['initial_native_integration'])==baseline['array_hashes']['initial_native_integration'];goal=a['task_goal_xy'];d=np.linalg.norm(a['poses'][:,:2]-goal,axis=1);dn=np.linalg.norm(a['next_poses'][:,:2]-goal,axis=1);dt=a['native_counts']*.002;clock=np.r_[0,np.cumsum(dt)[:-1]];speed=np.linalg.norm(a['next_poses'][:,:2]-a['poses'][:,:2],axis=1)/dt;n=min(len(a['poses']),len(ref['poses']));dx=np.linalg.norm(a['poses'][:n,:2]-ref['poses'][:n,:2],axis=1);dyaw=np.abs(np.arctan2(np.sin(a['poses'][:n,2]-ref['poses'][:n,2]),np.cos(a['poses'][:n,2]-ref['poses'][:n,2])));flags=(dx>.03)|(dyaw>np.deg2rad(3));first=int(np.flatnonzero(flags)[0]) if flags.any() else None;desired=np.arctan2(goal[1]-a['poses'][:,1],goal[0]-a['poses'][:,0]);heading=np.arctan2(np.sin(desired-a['poses'][:,2]),np.cos(desired-a['poses'][:,2]));nearest=float(min(d.min(),dn.min()));remaining=float(r['remaining_m']);label='GOAL' if r['success'] and not r['collision'] else 'CONTACT' if r['collision'] else 'TIMEOUT_AFTER_NEAR_APPROACH' if r['timeout'] and nearest<.5 and remaining-nearest>.3 else 'TIMEOUT_WITH_REMAINING_DISTANCE' if r['timeout'] else 'INCOMPLETE_OR_INVALID'
    return dict(id=r['job_id'],model_policy_hash=r['policy_hash'],success=bool(r['success']),contact=bool(r['collision']),timeout=bool(r['timeout']),remaining_m=remaining,minimum_recorded_endpoint_distance_m=nearest,seconds=float(dt.sum()),descriptive_outcome=label,first_3cm_or3deg_deviation_decision=first,first_deviation_seconds=float(clock[first]) if first is not None else None,first_deviation_translation_m=float(dx[first]) if first is not None else None,first_deviation_yaw_deg=float(np.rad2deg(dyaw[first])) if first is not None else None,mean_recorded_speed_mps=float(np.sum(speed*dt)/dt.sum()),slow_below_003mps_fraction=float(np.sum(dt[speed<.03])/dt.sum()),drive_mean=float(np.sum(a['actions'][:,0]*dt)/dt.sum()),steer_saturated_abs099_fraction=float(np.sum(dt[np.abs(a['actions'][:,1])>.99])/dt.sum()),heading_error_final_deg=float(np.rad2deg(heading[-1])),baseline_safe=True,initial_native_exact=True,not_unique_causal_diagnosis=True)

def main():
    p=configuration();setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None;diagnostics=[]
    try:
        rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for j in p['jobs']:
            values=rt.request([j],p['model'],'FROZEN_REJECTED_STUDENT_RESEARCH_DIAGNOSIS',full=True);a,r=values[0];F.M.audit_full_history(p,r,a);item=diagnose(p,r,a);diagnostics.append(item);save(OUT/'diagnostics.json',diagnostics);rt.request([j],p['model'],'EXACT_FROZEN_RESEARCH_DIAGNOSIS',full=True,reference=[r])
        result=dict(model=p['model'],diagnostics=diagnostics,new_optimizer=0,new_RL=0,promotion=False,old275=None,six54=None,eight54=None,field24=None,fresh16=None,reload78=None,scope='Only3consumedfailuretargets,frozenresearchnotqualificationortraining;old8/32notexecuted');state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit():
    p=configuration();setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps)==3;by={r['path']:r for r in rows};di=[]
    for r in rows:
        a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert not a['ownership'].any() and not a['stochastic'].any();F.M.audit_full_history(p,r,a);di.append(diagnose(p,r,a))
    for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
    assert di==s['result']['diagnostics']==read(OUT/'diagnostics.json');assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'];assert s['commands']==10<=p['max_commands'];assert sha(p['model']['path'])==p['model']['sha256']
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,all3full_history_and3exact_native=True,physics=s['physics'],commands=s['commands'],model_unchanged=True,no_promotion=True));print('T567_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

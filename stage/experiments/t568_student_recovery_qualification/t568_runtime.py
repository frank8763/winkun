"""One bounded smooth consolidation, qualified autonomous panel, then probes."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t568_student_recovery_qualification';DOC=ROOT/'stage/docs/day_20261115/t568'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'stage/experiments/t562_full_history_representation'));import t562_actor as L
sys.path.insert(0,str(ROOT/'stage/experiments/t555_adam_direction_control'));import t555_runtime as Old
Base,F,G,E,P,np,torch=Old.Base,Old.F,Old.G,Old.E,Old.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest
sys.path.insert(0,str(ROOT/'stage/experiments/t544_terminal_feedback_repair'));import t544_runtime as Teacher

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
    G.T.Session=Teacher.RecoverySession;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=Teacher.payload;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(1):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(len(self.children))];assert all(v['state']=='READY' for v in self.ready);save(OUT/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
        except BaseException:self.close();raise
    def receive(self,index):
        value=super().receive(index)
        if isinstance(value,tuple):value[1].update(teacher=True,teacher_first=self.p['teacher_first'],teacher_contract='TRAINING_ONLY_REACTIVE_GOAL_FEEDBACK_VALIDATION',training_mixture_NOT_autonomous=True)
        return value

def main():
    p=configuration();setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None
    try:
        baseline=E.N.S.arrays(p['baseline_row']['path']);rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);values=rt.request([p['job']],p['model'],'ACTUAL_STUDENT_PREFIX_THEN_TRAINING_ONLY_FEEDBACK',full=True);a,r=values[0];Teacher.recovery_audit(p,r,a,baseline);rt.request([p['job']],p['model'],'EXACT_STUDENT_PREFIX_AND_FEEDBACK',full=True,reference=[r]);eligible=bool(r['success'] and not r['collision']);result=dict(model=p['model'],teacher_qualified=eligible,teacher_outcome=dict(success=r['success'],contact=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],seconds=r['physics']*.002),fixed_prefix=p['fixed_prefix'],new_optimizer=0,new_RL=0,new_student_model=None,student_autonomous_test=None,promotion=False,old275=None,six54=None,eight54=None,field24=None,fresh16=None,reload78=None,scope='Teacher-assisted recovery qualification only, not autonomous GOAL or RL gain');state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit():
    p=configuration();setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps)==1;r=rows[0];a=E.N.S.arrays(r['path']);baseline=E.N.S.arrays(p['baseline_row']['path']);assert sha(r['path'])==r['sha256'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items());Teacher.recovery_audit(p,r,a,baseline);z=reps[0];assert z['exact'] and z['array_hashes']==r['array_hashes'] and z['integration_sha256']==r['integration_sha256'];assert s['physics']==r['physics']+z['physics']<=p['max_physics'];assert s['commands']==6<=p['max_commands'];assert s['result']['teacher_qualified']==bool(r['success'] and not r['collision']);assert sha(p['model']['path'])==p['model']['sha256']
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,actual_prefix_all_arrays_and_boundary_exact=True,all_current_hidden_density_reward_history_and_teacher_actions_verified=True,teacher_native_replay_exact=True,physics=s['physics'],commands=s['commands'],teacher_qualified=s['result']['teacher_qualified'],no_autonomous_student_improvement_claim=True));print('T568_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

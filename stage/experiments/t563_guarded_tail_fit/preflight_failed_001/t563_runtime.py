"""One bounded smooth consolidation, qualified autonomous panel, then probes."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t563_guarded_tail_fit';DOC=ROOT/'stage/docs/day_20261115/t563'
sys.path.insert(0,str(HERE));import t563_actor as L
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
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready);save(OUT/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
        except BaseException:self.close();raise
def assess(p,rows):
    assert [r['job_id'] for r in rows]==[j['id'] for j in p['evaluation_jobs']];ids=sorted(r['job_id'] for r in rows if r['success'] and not r['collision']);old=G.R.M.tally(rows[:8]);field=G.R.M.tally(rows[8:]);lost=sorted(set(p['protected_success_ids'])-set(ids))
    return dict(old8=old,field24=field,success_ids=ids,lost_success_ids=lost,retained23=not lost and len(ids)>=23,not_global_promotion=True)
def event(who):return lambda stage,k:Base.append('optimizer_calls.jsonl',[who,stage,k])

def main():
    p=configuration();setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);result=None;error=None;state='FAILED';obj=None
    try:
        rt.role='SMOOTH_OFFLINE_DISTILLATION';data=L.dataset(p);obj=L.make(p);initial_hash=digest(L.state(obj));u=L.fit(obj,data,p,lambda:rt.guard(True),event('live'));shadow=L.make(p);v=L.fit(shadow,data,p,lambda:rt.guard(True),event('shadow'))
        assert digest(u)==digest(v) and digest(L.state(obj))==digest(L.state(shadow));E.N.write_npz(OUT/'fit_trace.npz',dict(trace=u.pop('trace')),64*1024);v.pop('trace')
        model=L.export(obj,p,OUT/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==obj[0].phash();result=dict(reason='OFFLINE_FIT_GATE_FAILED_NO_NATIVE',fit=u,model=model,initial_joint_hash=initial_hash,live_Adam=u['steps'],shadow_Adam=v['steps'],new_RL=0,check=None,perturbation=None,old275=None,six54=None,eight54=None,fresh16=None,reload78=None,retained_research_student=False)
        if u['fit_gate']:
            rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
            short=rt.request(p['probe_jobs'],model,'SHORT_NEW_STUDENT_ENGINEERING',full=True);rt.request(p['probe_jobs'],model,'EXACT_SHORT_STUDENT',full=True,reference=[r for a,r in short])
            for a,r in short:F.M.audit_full_history(p,r,a)
            vals=rt.request(p['evaluation_jobs'],model,'PURE_SMOOTH_STUDENT32');rr=[r for a,r in vals];rt.request(p['evaluation_jobs'],model,'EXACT_SMOOTH_STUDENT32',reference=rr);check=assess(p,rr);result.update(check=check,reason='AUTONOMOUS_PANEL_RETAINED' if check['retained23'] else 'PANEL_REGRESSION_REJECTED')
            result['retained_research_student']=check['retained23'];result['reason']='FINITE_SUPPORT_TRANSFER_QUALIFIED_NOT_GLOBAL_GATE' if check['retained23'] else 'PANEL_REGRESSION_REJECTED'
        state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        else:rt.commands=0
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit():
    p=configuration();setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';r=s['result'];data=L.dataset(p);obj=L.make(p);assert digest(L.state(obj))==r['initial_joint_hash'];events=[];u=L.fit(obj,data,p,lambda:None,lambda a,b:events.append([a,b]));trace=u.pop('trace');assert u==r['fit'] and np.array_equal(trace,np.load(OUT/'fit_trace.npz')['trace']);assert digest(L.state(obj))==r['model']['joint_hash'];assert sha(r['model']['path'])==r['model']['sha256'] and L.frozen(p,r['model']['path']).phash()==obj[0].phash()
    assert G.rows(OUT,'optimizer_calls.jsonl')==[[who,a,b] for who in ['live','shadow'] for a,b in events];rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');by={x['path']:x for x in rows}
    for row in rows:
        a=E.N.S.arrays(row['path']);assert sha(row['path'])==row['sha256'] and int(a['native_counts'].sum())==row['physics'];assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        if row['full']:F.M.audit_full_history(p,row,a)
    for row in reps:assert row['exact'] and row['array_hashes']==by[row['reference_path']]['array_hashes'] and row['integration_sha256']==by[row['reference_path']]['integration_sha256']
    if r['fit']['fit_gate']:
        assert assess(p,rows[2:34])==r['check']
        assert len(rows)==len(reps)==34
    else:assert not rows and not reps
    assert s['physics']==sum(x['physics'] for x in rows+reps)<=p['max_physics'];assert s['commands']==(len(rows)+len(reps)+4 if rows else 0)<=p['max_commands']
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Adam=u['steps'],fit_and_all_Adam_rebuilt=True,all_saved_full_histories_rebuilt=True,all_native_arrays_replayed_exact=True,physics=s['physics'],commands=s['commands'],not_global_stability_or_generalization=True));print('T563_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

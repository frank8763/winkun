"""One bounded smooth consolidation, qualified autonomous panel, then probes."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t569_frozen_complete_panel';DOC=ROOT/'stage/docs/day_20261115/t569'
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
            self.ready=[self.receive(i) for i in range(len(self.children))];assert all(v['state']=='READY' for v in self.ready);save(OUT/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
        except BaseException:self.close();raise
def combine(p,rows):
    allrows=rows+p['imported_rows'];by={r['job_id']:r for r in allrows};assert len(by)==len(allrows)==32 and all(r['policy_hash']==p['model']['policy_hash'] and r['loaded_model_sha256']==p['model']['sha256'] for r in allrows);ordered=[by[j['id']] for j in p['evaluation_jobs']];ids=sorted(r['job_id'] for r in ordered if r['success'] and not r['collision']);prior=set(p['prior23_success_ids']);return dict(old8=G.R.M.tally(ordered[:8]),field24=G.R.M.tally(ordered[8:]),field_qualified_success_ids=sorted(r['job_id'] for r in ordered[8:] if r['field_detail']['field_qualified_success']),success_ids=ids,lost_vs_T55023=sorted(prior-set(ids)),gained_vs_T55023=sorted(set(ids)-prior),same_model_native29_new3_imported=True,new_unique_autonomous=29,imported_unique_autonomous=3,new_exact_replays=0,imported_exact_replays=3,all32_promotion_qualified=False,old275=None,six54=None,eight54=None,fresh16=None,reload78=None)

def main():
    p=configuration();setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None
    try:
        rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for j in p['jobs']:
            a,r=rt.request([j],p['model'],'FROZEN_STUDENT_MISSING29_DEVELOPMENT',full=False)[0];F.M.audit_full_history(p,r,a);Base.append('inline_full_history_checks.jsonl',dict(id=j['id'],source_sha256=r['sha256'],policy_hash=r['policy_hash'],complete_actual_arrays_checked_in_RAM=True,not_independent_offline_full_history_audit=True))
        result=combine(p,rt.rows);save(OUT/'combined_development32.json',dict(model=p['model'],ordered_rows=[{r['job_id']:r for r in rt.rows+p['imported_rows']}[j['id']] for j in p['evaluation_jobs']],result=result));state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit():
    p=configuration();setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(OUT);assert len(rows)==29 and not G.rows(OUT,'replays.jsonl')
    for r in rows:
        a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert not r['full'];assert a['end_flags'][-1].tolist()==[int(r['collision']),int(r['success']),int(r['timeout'])]
    assert len(G.rows(OUT,'inline_full_history_checks.jsonl'))==29
    for r in p['imported_rows']:
        a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items());F.M.audit_full_history(p,r,a)
    assert combine(p,rows)==s['result']==read(OUT/'combined_development32.json')['result'];assert s['physics']==sum(r['physics'] for r in rows)<=p['max_physics'];assert s['commands']==33<=p['max_commands'];assert sha(p['model']['path'])==p['model']['sha256'];save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,all29_compact_saved_actions_counts_flags_verified=True,new29_all_arrays_recorded_hash_only_not_independent_fullhistory=True,main29_RAM_full_history_checked=True,imported3_fullhistories_rebuilt=True,physics=s['physics'],commands=s['commands'],model_unchanged=True,no_promotion=True));print('T569_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

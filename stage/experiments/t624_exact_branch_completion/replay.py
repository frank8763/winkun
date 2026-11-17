from pathlib import Path
import sys,os,time,fcntl,traceback,copy
import t624_native as V
import t623_runtime as M
sys.path.insert(0,str(V.ROOT/'stage/experiments/t623r_branch_contract_recovery'))
import recover as C
ROOT=V.ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t624_exact_branch_completion';DOC=ROOT/'stage/docs/day_20261117/t624'
L,torch,np=M.L,M.torch,M.np
read,save,sha=L.read,L.save,L.sha
verify=M.verify
def plan():
 p=L.read(HERE/'protocol.json');c=L.read(p['actor_configuration']);c.update(p);c.update(out=str(OUT),child_bytes=p['max_bytes']);return c
def setup(p):
 M.OUT=OUT;M.HERE=HERE;M.DOC=DOC;M.setup();V.restore_parent(p);M.verify(p,True)
def main():
 p=plan();setup(p);M.R.R.E.N.offline();M.R.R.E.L.install_stop_handlers();rt=M.R.R.Base.Runtime(p);rt.commands=3;rt.counts=M.N.G.R.M.mp.get_context('spawn').RawArray('q',4);original=M.N.worker;M.N.worker=V.worker;error=None
 try:
  rt.pool=M.N.Pool(dict(p,native_stop_epoch=rt.deadline,phase='student'),rt.counts,rt.guard);workers=L.read(OUT/'worker_generations.json')[0]['workers'];actual=[sorted(os.sched_getaffinity(i)) for i in workers];assert actual==[[8],[10],[12]],actual;L.save(OUT/'ACTUAL_AFFINITY.json',dict(worker_pids=workers,actual=actual,parent=sorted(os.sched_getaffinity(0))))
  refs=L.read(p['reference_rows']);vv=M.execute(rt,p,refs)
  for a,r in vv:C.check(p,a,r,M.R.R.E.N.S.arrays(p['prefix_source']))
 except BaseException:error=traceback.format_exc();L.save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=3
  L.save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',error=error,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,new_optimizer=0));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert error is None,error
def audit():
 p=plan();setup(p);M.R.R.G.stopped(OUT);s=L.read(OUT/'summary.json');assert s['state']=='COMPLETE';refs=L.read(p['reference_rows']);reps=M.R.R.G.rows(OUT,'replays.jsonl');assert len(reps)==3 and s['commands']==9 and s['physics']==sum(r['physics'] for r in reps)<=p['max_physics']
 source=M.R.R.E.N.S.arrays(p['prefix_source'])
 for old,r in zip(refs,reps):
  assert L.sha(old['path'])==old['sha256'] and r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'];a=M.R.R.E.N.S.arrays(old['path']);C.check(p,a,old,source)
 L.save(OUT/'audit.json',dict(state='PASS',three_original_native_branch_exact_replays=True,all_array_and_native_integration_hashes_exact=True,actual_worker_affinity_verified=True,new_native_during_audit=0,new_optimizer=0,original_T623_producer_remains_FAILED=True))
if __name__=='__main__':globals()[sys.argv[1]]()

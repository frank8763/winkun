from pathlib import Path
import sys,time,os,fcntl,traceback
import t625_native as New
import t623_runtime as M
L,torch,np=M.L,M.torch,M.np
ROOT=New.ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t625_target_continuation_calibration';DOC=ROOT/'stage/docs/day_20261117/t625'
read,save,sha=L.read,L.save,L.sha
def plan():
 p=read(HERE/'protocol.json');c=read(p['actor_configuration']);c.update(p);c.update(out=str(OUT/'collect'),child_bytes=p['max_bytes']);return c
def setup(p):
 M.OUT=OUT/'collect';M.HERE=HERE;M.DOC=DOC;M.setup();New.V.restore_parent(p)
 M.R.R.Base.verify=lambda *args:verify(p)
def verify(p):
 assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json') and time.time()<p['absolute_all_stop']
 for f,h in p['inputs'].items():assert sha(f)==h,f
def main():
 p=plan();setup(p);verify(p);M.R.R.E.N.offline();M.R.R.E.L.install_stop_handlers();rt=M.R.R.Base.Runtime(p);rt.commands=3;rt.counts=New.N.G.R.M.mp.get_context('spawn').RawArray('q',4);New.N.worker=New.worker;error=None;rows=[]
 try:
  rt.pool=New.Pool(dict(p,native_stop_epoch=rt.deadline,phase='student'),rt.counts,rt.guard);workers=read(M.OUT/'worker_generations.json')[0]['workers'];actual=[sorted(os.sched_getaffinity(i)) for i in workers];assert actual==[[8],[10],[12]];save(M.OUT/'ACTUAL_AFFINITY.json',dict(workers=workers,actual=actual))
  vv=M.execute(rt,p);New.inspect(p,vv);rows=[r for a,r in vv];save(M.OUT/'ROWS.json',rows);M.execute(rt,p,rows)
 except BaseException:error=traceback.format_exc();save(M.OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=3
  save(M.OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',error=error,rows=rows,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,new_optimizer=0));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert error is None,error
def audit():
 p=plan();setup(p);verify(p);M.R.R.G.stopped(M.OUT);s=read(M.OUT/'summary.json');assert s['state']=='COMPLETE';rows=M.R.R.G.rows(M.OUT);reps=M.R.R.G.rows(M.OUT,'replays.jsonl');assert len(rows)==len(reps)==3 and s['commands']==12 and s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics']
 for old,r in zip(rows,reps):
  assert sha(old['path'])==old['sha256'] and r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'];a=M.R.R.E.N.S.arrays(old['path']);assert all(L.digest(v)==old['array_hashes'][key] for key,v in a.items());New.inspect(p,[(a,old)])
 save(M.OUT/'audit.json',dict(state='PASS',full_arrays_and_integrations_and_all_actor_memory_rewards_reconstructed=True,three_original_and_three_exact=True,new_optimizer=0,new_native=0))
if __name__=='__main__':globals()[sys.argv[1]]()

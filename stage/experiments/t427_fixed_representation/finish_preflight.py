"""Complete only interrupted zero-native worker evidence; no optimizer work."""
from t427_core import *
def main():
 p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();assert read(OUT/'clone_comparison.json')['state']=='PASS_NEW_ONE_STEP_CROSS_PROCESS_CONTRACT';geometry=read(OUT/'eight_geometry.json');assert len(geometry)==56 and all(g['valid'] for g in geometry)
 previous=read(OUT/'preflight_native/pool/workers.json');assert all(not Path('/proc',str(pid)).exists() for pid in [previous['parent'],*previous['workers']]);ban_learning();folder=OUT/'preflight_completion';folder.mkdir();counts=mp.get_context('spawn').RawArray('q',5);pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
 try:
  ids=[c.pid for c in pool.children];values=pool.request([dict(model=p['student_start_model'],job=p['main_jobs'][0],mode='actor_teacher',seed=None,cancel_probe=True,teacher_model=p['teacher_A_model'],teacher_sha256=p['teacher_A_sha256'])]*4);assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
 finally:pool.close()
 assert all(not Path('/proc',str(pid)).exists() for pid in ids)
 save(OUT/'preflight.json',dict(state='PASS',native=0,new_discarded_Adam=2,additional_recovery_Adam=0,zero_physics_commands_upper=72,independent_clone=read(OUT/'clone_comparison.json'),retained_original_geometry56=True,fresh_four_teacher_workers_cancelled_and_exited=True,original_interrupted_preflight_preserved=True,old_T422_audit_NOT_repaired=True,protocol_sha256=sha(OUT/'protocol.json')));E.N.manifest(OUT);print('T427_PREFLIGHT_COMPLETED_NO_EXTRA_ADAM_OR_NATIVE',flush=True)
if __name__=='__main__':main()

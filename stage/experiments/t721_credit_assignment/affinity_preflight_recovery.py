"""Only remaining metadata/affinity checks; previously consumed Adam is not repeated."""
from pathlib import Path
import sys,copy,time,os,io,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent));import t721_runtime as T
C,L,RN=T.C,T.L,T.RN
C.old_stopped();C.unit_stopped('stage-t721-preflight-recovery.service');p=C.read(C.OUT/'prepared.json')
T.fit_affinity(C.OUT/'worker_preflight','preflight')
contract=C.read(C.DOC/'NATIVE_PREFLIGHT.json');contract.update(state='PASS',fit8threads_vs_deploy1thread_owncarry_exact=True,actual_fit_thread_masks=C.read(C.OUT/'worker_preflight/preflight_THREAD_AFFINITY.json'),legacy_affinity_wrapper_bypassed_only_current_learner=True,no_repeated_optimizer_or_native=True)
# Actual highest-entropy NPZ fixture at maximum declared decisions. New arrays
# use original schema and dtypes, retaining all keys; zero native, zero optimizer.
a=RN.Base.C.E.N.S.arrays(p['training_jobs'][0]['source_path']);packed=RN.Base.C.E.N.S.packed(a);n=len(a['states']);rng=L.np.random.default_rng(721);upper=C.read(C.DOC/'STORAGE_BOUND.json');proof=[]
for length,bound in [(upper['train_decisions'],upper['train_file_bound']),(upper['eval_decisions'],upper['eval_file_bound'])]:
    arrays={}
    for key,v in packed.items():
        shape=(length,*v.shape[1:]) if v.ndim and len(v)==n else ((length+1,*v.shape[1:]) if v.ndim and len(v)==n+1 else v.shape)
        raw=rng.integers(0,256,size=int(L.np.prod(shape))*v.dtype.itemsize,dtype=L.np.uint8).tobytes();arrays[key]=L.np.frombuffer(raw,dtype=v.dtype).reshape(shape)
    arrays['source_prefix_actions']=L.np.frombuffer(rng.bytes(length*8),dtype=L.np.float32).reshape(length,2);arrays['recovery_first']=L.np.asarray(1,L.np.int64)
    output=io.BytesIO();L.np.savez_compressed(output,**arrays);size=len(output.getvalue());assert size<bound;proof.append(dict(decisions=length,actual_high_entropy_bytes=size,declared_bound=bound))
contract['full_schema_high_entropy_ZIP_worst_size']=proof
# Validate state writer tensor ceiling including both Adam moments and counters.
obj=L.make(p);state=L.state(obj);tensorbytes=sum(v.numel()*v.element_size() for v in state['actor'].values())+sum(v.numel()*v.element_size() for v in state['value'].values());plastic=sum(v.numel()*v.element_size() for v in obj[0].replay_head.parameters())+sum(v.numel()*v.element_size() for v in obj[1].parameters());assert tensorbytes+2*plastic+128*2**10<4*2**20
contract['state_tensor_plus_Adam_and_serialization_bound']=tensorbytes+2*plastic+128*2**10
# Real serialized state roundtrip verifies continuation seam without any steps.
row=L.write_state(C.OUT/'worker_preflight/continuation_fixture.pt.gz',state);again=L.make(p);L.load(again,L.read_state(row));assert L.digest(L.state(again))==L.digest(state);contract['serialized_Actor_Value_Adam_handoff_exact']=True
C.save(C.DOC/'NATIVE_PREFLIGHT.json',contract)
C.save(C.DOC/'AFFINITY_PREPARATION_RECOVERY.json',dict(state='PASS',original_red_preserved=True,cause='Historical imported affinity wrapper maps nonlegacy CPUs into14/15. Current process uses original Linux affinity syscall for approved8..15.',new_native=0,new_optimizer=0,proof=contract['actual_fit_thread_masks']))
print('T721_REMAINING_PREFLIGHT_PASS',proof,flush=True)

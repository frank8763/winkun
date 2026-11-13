"""Finish zero-native byte validation with ndarray-containing deque support.

Original source/protocol and failed equality attempt remain unchanged.
"""
from pathlib import Path
import collections,importlib.util,math,os,pickle,sys,time
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_t535_registered_prepare',HERE/'prepare.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
R,C,OUT=P.R,P.C,P.OUT
def equal(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple,collections.deque)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,R.torch.Tensor):return R.torch.equal(a,b)
    if isinstance(a,R.np.ndarray):return R.np.array_equal(a,b,equal_nan=True)
    if isinstance(a,(float,R.np.floating)) and math.isnan(a):return isinstance(b,(float,R.np.floating)) and math.isnan(b)
    if isinstance(a,R.np.random.Generator):return isinstance(b,R.np.random.Generator) and equal(a.bit_generator.state,b.bit_generator.state)
    if hasattr(a,'__dict__') and type(a)==type(b):return equal(vars(a),vars(b))
    return bool(a==b)
def main():
    p=R.read(OUT/'protocol.json');assert not (OUT/'summary.json').exists()
    for f,h in p['source_code'].items():assert R.sha(f)==h
    assert time.time()<p['absolute_all_stop']
    if not (OUT/'serialization_failed_001.json').exists():R.save(OUT/'serialization_failed_001.json',dict(new_native=0,new_optimizer=0,error='Original equality function omitted ndarray-containing deque; ambiguous array truth value during nonlogging-state comparison. No minimal artifact or successful summary had been written.',original_source_unchanged=True,recovery_source_sha256=R.sha(__file__)))
    rows=[]
    for i,source in enumerate(p['source_snapshots']):
        assert R.sha(source['path'])==source['sha256'];old=C.unpack(Path(source['path']).read_bytes());minimal=C.compact(old);C.verify_preserved(old,minimal,equal)
        blob=C.pack(minimal);assert len(blob)<=16*1024;path=OUT/('minimal_'+str(i)+'.pkl.gz');path.write_bytes(blob);C.verify_preserved(old,C.unpack(path.read_bytes()),equal)
        test=C.unpack(blob);test['native']['physics'][0]+=1.;assert not equal(test['native']['physics'],old['native']['physics'])
        rows.append(dict(source=source,minimal_path=str(path),minimal_sha256=R.sha(path),original_bytes=Path(source['path']).stat().st_size,minimal_bytes=len(blob),all_nonlogging_state_preserved=True,native_restore_equivalence_NOT_tested=True))
    for row in rows:C.verify_preserved(C.unpack(Path(row['source']['path']).read_bytes()),C.unpack(Path(row['minimal_path']).read_bytes()),equal)
    result=dict(state='PASS_SERIALIZATION_ONLY',pid=os.getpid(),rows=rows,new_native=0,new_optimizer=0,native_restore_gate=None,directions=None,navigation_result=None,recovery_source_sha256=R.sha(__file__),original_failed_equality_preserved=True,not_a_new_native_budget=True)
    R.save(OUT/'summary.json',result);R.save(OUT/'audit.json',dict(state='PASS',scope='ZERO_NATIVE_SERIALIZATION_SAME_PROCESS_DISK_REOPEN_CHECK_ONLY',new_native=0,new_optimizer=0,native_continuation_NOT_tested=True,independent_process_native_audit=False))
    assert R.used(OUT)<p['max_bytes'] and R.used(R.ROOT/'stage/runs')<=2**30
    R.save(R.DOC/'T535_PREPARATION.json',result);print('T535_COMPACT_STATE_SERIALIZATION_PASS_NATIVE_NOT_TESTED',flush=True)
if __name__=='__main__':main()

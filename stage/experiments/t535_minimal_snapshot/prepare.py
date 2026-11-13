"""Zero-native serialization contract only; no native continuation is launched."""
from pathlib import Path
import gzip,json,os,pickle,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t535_minimal_snapshot'
sys.path.insert(0,str(ROOT/'stage/experiments/t534_native_state_restore'));import t534_runtime as R
sys.path.insert(0,str(HERE));import compact_snapshot as C
def equal(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,R.torch.Tensor):return R.torch.equal(a,b)
    if isinstance(a,R.np.ndarray):return R.np.array_equal(a,b,equal_nan=True)
    return a==b
def main():
    prior=ROOT/'stage/runs/t534_native_state_restore';assert not OUT.exists()
    assert R.read(prior/'audit.json')['scope']=='EXECUTED_TWO_WINDOWS_AND_FAILURE_ACCOUNTING_ONLY'
    assert R.read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T534.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=dict(id='T535_ZERO_NATIVE_SNAPSHOT_SERIALIZATION_CONTRACT',source_original_protocol_sha256=R.sha(prior/'protocol.json'),source_failed_summary_sha256=R.sha(prior/'summary.json'),max_bytes=128*1024,new_native=0,new_Actor_Adam=0,new_Value_Adam=0,native_restore_gate=None,absolute_all_stop=R.read(prior/'protocol.json')['absolute_all_stop'],source_snapshots=[dict(path=str(prior/('snapshot_'+str(i)+'.pkl.gz')),sha256=R.sha(prior/('snapshot_'+str(i)+'.pkl.gz'))) for i in range(2)],source_code={str(f):R.sha(f) for f in HERE.glob('*.py')},scope='Serialization of already persisted old24decision snapshots ONLY. Removes historical logging buffers, leaves physical/causal state exactly unchanged. No new321decision snapshot, no actual restore continuation, no direction/Actor training, not a solution or extension of T534 physical budget.')
    assert time.time()+120<p['absolute_all_stop'] and R.used(ROOT/'stage/runs')+p['max_bytes']<2**30
    OUT.mkdir();R.save(OUT/'protocol.json',p);R.save(HERE/'protocol.json',p);rows=[]
    for i,source in enumerate(p['source_snapshots']):
        assert R.sha(source['path'])==source['sha256'];old=C.unpack(Path(source['path']).read_bytes());minimal=C.compact(old);C.verify_preserved(old,minimal,equal)
        blob=C.pack(minimal);assert len(blob)<=16*1024;path=OUT/('minimal_'+str(i)+'.pkl.gz');path.write_bytes(blob);loaded=C.unpack(path.read_bytes());C.verify_preserved(old,loaded,equal)
        # Mutating the compact clone cannot mutate the original state.
        test=C.unpack(blob);test['native']['physics'][0]+=1.;assert not equal(test['native']['physics'],old['native']['physics'])
        rows.append(dict(source=source,minimal_path=str(path),minimal_sha256=R.sha(path),original_bytes=Path(source['path']).stat().st_size,minimal_bytes=len(blob),all_nonlogging_state_bitwise_preserved=True,native_restore_equivalence_NOT_tested=True))
    result=dict(state='PASS_SERIALIZATION_ONLY',pid=os.getpid(),rows=rows,new_native=0,new_optimizer=0,native_restore_gate=None,directions=None,navigation_result=None,not_a_new_native_budget=True)
    R.save(OUT/'summary.json',result)
    # Re-open and recompare from original disk inputs for the independent
    # serialization check; this does not stand in for a physical replay audit.
    for row in rows:C.verify_preserved(C.unpack(Path(row['source']['path']).read_bytes()),C.unpack(Path(row['minimal_path']).read_bytes()),equal)
    R.save(OUT/'audit.json',dict(state='PASS',scope='ZERO_NATIVE_BYTE_STATE_SERIALIZATION_ONLY',new_native=0,new_optimizer=0,native_continuation_NOT_tested=True))
    assert R.used(OUT)<=p['max_bytes'] and R.used(ROOT/'stage/runs')<=2**30
    R.save(ROOT/'stage/docs/day_20261113/T535_PREPARATION.json',result)
    print('T535_SERIALIZATION_PASS_NATIVE_RESTORE_STILL_UNTESTED',flush=True)
if __name__=='__main__':main()

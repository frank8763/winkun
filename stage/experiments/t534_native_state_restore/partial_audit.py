"""Zero-native closeout of the snapshot-size stop; never rerun the native job."""
from pathlib import Path
import gzip,json,os,pickle,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t534_runtime as R
ROOT,OUT,DOC,read,save,sha,np,digest=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha,R.np,R.digest
def main():
    p=read(OUT/'protocol.json');R.verify(p,True);s=read(OUT/'summary.json');driver=read(OUT/'launch.json')
    assert s['state']=='FAILED' and 'snapshot_bytes' in s['error'] and 'pickle.dumps(snapshot' in s['error']
    assert not Path('/proc',str(s['pid'])).exists() and not Path('/proc',str(driver['pid'])).exists()
    assert not (OUT/'audit.json').exists() and not (OUT/'SEALED.json').exists()
    rr=s['result']['windows'];assert len(rr)==2 and all(r['exact'] for r in rr)
    for r,window in zip(rr,p['windows']):
        assert sha(r['snapshot_path'])==r['snapshot_sha256'] and sha(r['payload_path'])==r['payload_sha256']
        snap=pickle.loads(gzip.decompress(Path(r['snapshot_path']).read_bytes()));a=R.E.N.S.arrays(window['row']['path']);first=window['first']
        assert snap['steps']==first*100 and np.array_equal(snap['packet'],a['states'][first])
        assert np.array_equal(snap['hidden'].numpy().reshape(64),a['hidden_before'][first].reshape(64))
        assert len(snap['rows'])==len(snap['qstate'][2])==first
        for k,row in enumerate(snap['rows']):
            for name,v in zip(['states','actions','native_counts','next_states','poses','next_poses','end_flags','rewards'],row):assert np.array_equal(np.asarray(v),a[name][k]),(name,k)
        assert digest(dict(physics=snap['native']['physics'],packet=snap['packet'],hidden=snap['hidden'],meta=snap['meta'],steps=snap['steps'],previous=snap['previous'],pn=snap['pn']))==r['snapshot_digest']
        with np.load(r['payload_path']) as z:
            for k in z.files:
                assert digest(z[k])==r['array_hashes'][k]
                if k.startswith('continuous__'):assert np.array_equal(z[k],z[k.replace('continuous__','restored__')])
            for k in ['states','next_states','actions','native_counts','poses','next_poses','rewards','hidden_before','locations']:assert np.array_equal(z['continuous__'+k],a[k][first:first+p['holds']]),k
    partial=s['physics']-sum(r['physics'] for r in rr);assert s['physics']==38900 and partial==32100 and s['commands']==6
    assert not (OUT/'snapshot_2.pkl.gz').exists() and not (OUT/'snapshot_3.pkl.gz').exists()
    audit=dict(state='PASS',scope='EXECUTED_TWO_WINDOWS_AND_FAILURE_ACCOUNTING_ONLY',pid=os.getpid(),new_native=0,new_optimizer=0,completed_restore_windows=2,all_four_restore_gate=None,new_goal_restore_NOT_tested=True,unpersisted_third_prefix_physics=partial,unpersisted_third_prefix_NOT_independently_rebuilt_from_new_raw=True,physics=s['physics'],commands=s['commands'],reason='ENGINEERING_SINGLE_SNAPSHOT_SIZE_CAP_STOP_NOT_NAVIGATION_FAILURE',original_failed_summary_unchanged=True)
    save(OUT/'audit.json',audit)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=[s['pid'],driver['pid']],finalizer_writes_only_docs_after_this_point=True,engineering_partial_closeout=True,epoch=time.time()))
    save(DOC/'T534_PARTIAL_AUDIT_REGISTRATION.json',dict(new_native=0,new_optimizer=0,script_sha256=sha(__file__),original_protocol_sha256=sha(OUT/'protocol.json'),original_failed_summary_sha256=sha(OUT/'summary.json'),raw_preserved=True,no_restart=True,audit_scope=audit['scope']))
    print('T534_PARTIAL_EVIDENCE_AUDIT_PASS_FULL_GATE_UNTESTED',flush=True)
if __name__=='__main__':main()

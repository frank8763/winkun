"""One HTTP1 transport fallback, same original cache plan and deadline.

Go CLI returned EOF for real assets. Existing curl reader is used instead;
all original archive and member verifiers remain unchanged. No scientific run
is restarted. No partially verified archive authorizes any unlink.
"""
from pathlib import Path
import importlib.util,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=ROOT/'stage/experiments/t481_fresh16_rehearsal'
OUT=ROOT/'stage/runs/t481_fresh16_rehearsal';DOC=ROOT/'stage/docs/review_20261111'
sys.path.insert(0,str(HERE));import control as C
import t481_cache as K
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'));import transport as T

def main():
    p=C.cache_protocol();cache=p['remote_cache'];deadline=p['cache_hard_stop_epoch'];K._idle()
    assert not any((OUT/f).exists() for f in ['launch.json','preflight.json','cache_remote_verified.json','cache_summary.json','cache_ledger.jsonl','cache_http1_started.json'])
    assert (OUT/'T481_QUEUE_STOPPED.json').exists() and (DOC/'T481_TRANSPORT_RECOVERY_FAILED.json').exists()
    assert time.time()+90<deadline
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert C.read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    K._fresh_inputs(p,cache);before=K.used();start=time.time()
    K._exclusive(OUT/'cache_http1_started.json',dict(pid=os.getpid(),epoch=start,hard_stop_epoch=deadline,native=0,optimizer=0))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original T481 cache deadline')))
    signal.alarm(max(1,int(deadline-start-40)))
    original=T.subprocess.run
    def http1(args,*a,**kw):
        if isinstance(args,list) and args and args[0]=='curl':args=[args[0],'--http1.1',*args[1:]]
        return original(args,*a,**kw)
    T.subprocess.run=http1
    spec=importlib.util.spec_from_file_location('_original_t460_verifier',K.READER);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    recovered={};verification=[]
    for group in cache['groups']:
        assert time.time()<deadline-170
        b,parts,members=mod.archive(group,deadline=deadline-45,cli=False)
        assert not set(recovered).intersection(b);recovered.update(b)
        verification.append(dict(group=group['group'],segments=parts,members=members,archive_sha256=group['archive_sha256']))
    K._idle();K._fresh_inputs(p,cache)
    selected=[r for g in cache['groups'] for r in g['rows']]
    assert set(recovered)=={r['path'] for r in selected}
    protected=K._protected(set(p['inputs'])|set(p['constructor_reads'])|set(p['protected_files']))
    for r in selected:
        f=ROOT/r['path'];assert K._eligible(f,protected) and f.read_bytes()==recovered[r['path']]
    K._exclusive(OUT/'cache_remote_verified.json',dict(state='ALL_GROUPS_ALL_MEMBERS_VERIFIED_BEFORE_ANY_UNLINK',verified_epoch=time.time(),groups=verification,selected_files=len(selected),transport='HTTP1_curl_original_member_verifier'))
    ledger=OUT/'cache_ledger.jsonl'
    with ledger.open('xb') as log:
        for group in cache['groups']:
            for r in group['rows']:
                assert time.time()<deadline-40
                f=ROOT/r['path'];assert K._eligible(f,protected) and f.read_bytes()==recovered[r['path']]
                marker=Path(str(f)+'.stage-remote.json');ref=K._marker(group,r)
                assert len(K.encoded(ref))==r['marker_bytes']<=2048
                K._exclusive(marker,ref);marker.chmod(0o444);f.unlink()
                entry=K._entry(group,r);assert K.sha(marker.read_bytes())==entry['marker_sha256']
                assert len(K.encoded(entry))==r['ledger_bytes'];log.write(K.encoded(entry));log.flush();os.fsync(log.fileno())
    K._fresh_inputs(p,cache);after=K.used();own=C.M.used(OUT)
    assert after<before and after-own+p['max_bytes']<=2**30
    K._exclusive(OUT/'cache_summary.json',dict(state='COMPLETE_REMOTE_VERIFIED_CACHE_EVICTION',pid=os.getpid(),files=len(selected),groups=len(cache['groups']),before=before,after=after,actual_net_released_bytes=before-after,released_payload_bytes=sum(r['bytes'] for r in selected),elapsed_s=time.time()-start,all_groups_verified_before_any_unlink=True,ledger_sha256=K.sha(ledger.read_bytes()),original_bytes_remote_preserved=True,offline_local_bytes_NOT_claimed=True,unbacked_unique_evidence_deleted=0,retained_models_deleted=0,local_models_deleted=sum(str(ROOT/r['path']) in K.REJECTED_MODELS for r in selected),native=0,optimizer=0))
    signal.alarm(0)
    C.save(DOC/'T481_HTTP1_RECOVERY.json',dict(state='PASS_ORIGINAL_DEADLINE_ALL_MEMBERS',epoch=time.time(),native=0,optimizer=0,all_original_errors_preserved=True,cache_hard_stop_epoch=deadline,summary_sha256=C.sha(OUT/'cache_summary.json')))
if __name__=='__main__':
    try:main()
    except BaseException:
        C.save(DOC/'T481_HTTP1_RECOVERY_FAILED.json',dict(error=traceback.format_exc(),epoch=time.time(),no_automatic_retry=True,native=0,optimizer=0));raise

"""One transport-only recovery under the original cache deadline.

Original queue/error are immutable. Uses the already registered recovery=True
entry, unchanged member verifier and same original selected files. No science.
"""
from pathlib import Path
import json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'stage/experiments/t481_fresh16_rehearsal';OUT=ROOT/'stage/runs/t481_fresh16_rehearsal'
DOC=ROOT/'stage/docs/review_20261111'
sys.path.insert(0,str(HERE));import control as C
import t481_cache as K

def main():
    p=C.cache_protocol()
    assert not any((OUT/f).exists() for f in ['launch.json','preflight.json','cache_remote_verified.json','cache_summary.json','cache_readback_recovery_started.json'])
    assert (OUT/'T481_QUEUE_STOPPED.json').exists()
    assert time.time()+90<p['cache_hard_stop_epoch']
    K.evict(p,recovery=True)
    C.save(DOC/'T481_TRANSPORT_RECOVERY.json',dict(state='COMPLETE_TRANSPORT_ONLY_ORIGINAL_CACHE_DEADLINE',pid=os.getpid(),epoch=time.time(),
        native=0,optimizer=0,original_queue_error_preserved=True,source_protocol_unchanged=True,
        old_hard_stop=p['cache_hard_stop_epoch'],all_selected_remote_members_SHA_verified=True))
if __name__=='__main__':
    try:main()
    except BaseException:
        C.save(DOC/'T481_TRANSPORT_RECOVERY_FAILED.json',dict(error=traceback.format_exc(),epoch=time.time(),native=0,optimizer=0,no_automatic_retry=True))
        raise

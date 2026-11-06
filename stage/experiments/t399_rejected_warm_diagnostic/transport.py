"""Closed immutable T399 bytes; RAM archive and exact source synchronization."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
original=T.B.require_stopped
original_response=T.response
def response(method,url,body=None,headers=None):
    # Repeated ended Stage transfers saw transient TLS35. Only read-only GET
    # transport gets a bounded delay; POST/PATCH mutations are never retried.
    for attempt in range(3 if method=='GET' else 1):
        try:return original_response(method,url,body,headers)
        except T.urllib.error.URLError as error:
            if method!='GET' or 'exit 35' not in str(error) or attempt==2:raise
            T.time.sleep(attempt+1)
def stopped(d):
    if d.name!='t399_rejected_warm_diagnostic':return original(d)
    s=json.loads((d/'summary.json').read_text());a=json.loads((d/'audit.json').read_text())
    assert s['state']=='COMPLETE' and a['state']=='PASS' and s['all_native_sessions_closed'] and s['all_workers_stopped']
    assert s['Actor_updates']==s['optimizer_updates']==s['backward_calls']==0 and a['weights_unchanged']
    for f in ['summary.json','workers.json']:
        v=json.loads((d/f).read_text())
        for pid in [v.get('pid'),v.get('parent'),*v.get('workers',[])]:
            if pid:assert not Path('/proc',str(pid)).exists()
    for r in json.loads((d/'local_manifest.json').read_text())['files']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
if __name__=='__main__':
    T.response=response;T.install();T.B.require_stopped=stopped
    if sys.argv[1:]==['T399']:T.B.GROUPS['T399']=['t399_rejected_warm_diagnostic'];T.B.main('T399')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Closed T399 evidence or exact source only')

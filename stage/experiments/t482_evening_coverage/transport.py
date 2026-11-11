"""Transport only: HTTP1 TLS recovery, bounded retries, original member checks."""
from pathlib import Path
import sys,json,time,urllib.error
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'));import transport as T

def stopped(folder):
    if folder==ROOT/'stage/runs/t483_local_log_reserve':
        d=json.loads((folder/'summary.json').read_text());assert d['state']=='COMPLETE' and d['native']==d['optimizer']==0
        assert json.loads((folder/'compression_audit.json').read_text())['state']=='PASS_LOCAL_REVERSIBLE_ENCODING'
        assert not Path('/proc',str(d['pid'])).exists();return
    assert folder==ROOT/'stage/runs/t482_evening_coverage';d=json.loads((folder/'SEALED.json').read_text());assert d['all_scientific_processes_stopped'] and d['finalizer_writes_only_docs_after_this_point'];assert all(not Path('/proc',str(pid)).exists() for pid in d['scientific_pids'])
def install():
    T.install();run=T.subprocess.run
    def http1(args,*a,**kw):
        if isinstance(args,list) and args and args[0]=='curl':args=[args[0],'--http1.1',*args[1:]]
        return run(args,*a,**kw)
    T.subprocess.run=http1;original=T.response
    def bounded(*a,**kw):
        for i in range(4):
            try:return original(*a,**kw)
            except urllib.error.URLError:
                if i==3:raise
                time.sleep(1)
    T.response=bounded
if __name__=='__main__':
    install()
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        assert sys.argv[1]=='T482';T.B.GROUPS['T482']=['t482_evening_coverage','t483_local_log_reserve'];T.B.require_stopped=stopped
        import small_parts
        small_parts.main('T482')

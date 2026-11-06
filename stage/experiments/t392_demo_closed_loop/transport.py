"""Only closed T392 raw archives or exact source root synchronization."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
old=T.B.require_stopped
def stopped(directory):
    if directory.name!='t392_demo_closed_loop':return old(directory)
    s=json.loads((directory/'summary.json').read_text());audit=json.loads((directory/'audit.json').read_text())
    assert s['state']=='COMPLETE' and s['all_native_sessions_closed'] and s['all_workers_stopped'] and audit['state']=='PASS'
    assert not Path('/proc',str(s['pid'])).exists()
    for row in json.loads((directory/'local_manifest.json').read_text())['files']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1:]==['T392']:
        T.B.GROUPS['T392']=['t392_demo_closed_loop'];T.B.main('T392')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Only closed T392 or exact source sync')

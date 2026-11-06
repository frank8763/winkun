"""Closed immutable diagnostic bytes and exact Stage source synchronization."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
old=T.B.require_stopped
def stopped(directory):
    if directory.name!='t394_takeover_diagnostic':return old(directory)
    s=json.loads((directory/'summary.json').read_text());a=json.loads((directory/'audit.json').read_text())
    assert s['state']=='COMPLETE' and s['all_native_sessions_closed'] and a['state']=='PASS'
    assert s['Actor_updates']==s['optimizer_updates']==s['backward_calls']==0
    assert not Path('/proc',str(s['pid'])).exists()
    for row in json.loads((directory/'local_manifest.json').read_text())['files']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1:]==['T394']:
        T.B.GROUPS['T394']=['t394_takeover_diagnostic'];T.B.main('T394')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Only closed T394 or exact source sync')

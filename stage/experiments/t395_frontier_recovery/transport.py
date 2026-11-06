"""Completed storage metadata and navigation evidence; no active archive."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
old=T.B.require_stopped
def stopped(directory):
    if directory.name not in ['t395_storage_reserve','t395_frontier_recovery']:return old(directory)
    s=json.loads((directory/'summary.json').read_text());a=json.loads((directory/'audit.json').read_text())
    assert s['state']=='COMPLETE' and a['state']=='PASS' and s['all_workers_stopped']
    assert not Path('/proc',str(s['pid'])).exists()
    if directory.name=='t395_frontier_recovery':
        for row in json.loads((directory/'local_manifest.json').read_text())['files']:
            assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1:] in [['T395-STORAGE'],['T395']]:
        group=sys.argv[1];T.B.GROUPS[group]=['t395_storage_reserve' if group=='T395-STORAGE' else 't395_frontier_recovery'];T.B.main(group)
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Closed Stage evidence or exact source synchronization')

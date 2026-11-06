"""Byte-complete backup of cancelled T392 and separately completed T393."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
old=T.B.require_stopped
def stopped(directory):
    if directory.name=='t392_demo_closed_loop':
        v=json.loads((directory/'cancelled_ledger.json').read_text())
        assert v['new_Actor_Adam']==0 and v['all_producer_pids_absent']
        for pid in [323350,323352,323353,323356]:assert not Path('/proc',str(pid)).exists()
    elif directory.name=='t393_safe_demo_pg':
        s=json.loads((directory/'summary.json').read_text());a=json.loads((directory/'audit.json').read_text())
        assert s['state']=='COMPLETE' and s['all_native_sessions_closed'] and a['state']=='PASS'
        assert not Path('/proc',str(s['pid'])).exists()
    else:return old(directory)
    for row in json.loads((directory/'local_manifest.json').read_text())['files']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1:] in [['T392-CANCELLED'],['T393']]:
        group=sys.argv[1];T.B.GROUPS[group]=['t392_demo_closed_loop' if group=='T392-CANCELLED' else 't393_safe_demo_pg'];T.B.main(group)
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Closed Stage evidence or exact source sync only')

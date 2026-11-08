"""Optional post-seal RAM-only raw evidence archive and exact Git source sync."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
def stopped(folder):
    assert folder.name=='t426_joint_retention';read=lambda f:json.loads(Path(f).read_text());assert read(folder/'SEALED.json')['all_scientific_processes_stopped']
    for name in ['data','learn','gate','eight']:
        d=folder/name
        if not (d/'main_started.json').exists():continue
        ids=[read(d/'main_started.json')['pid']]
        if (d/'workers.json').exists():ids+=read(d/'workers.json')['workers']
        assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        if (d/'summary.json').exists() and read(d/'summary.json')['state']=='COMPLETE':assert read(d/'audit.json')['state']=='PASS'
    for path,h in read(folder/'protocol.json')['inputs'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
if __name__=='__main__':
    assert sys.argv[1:] in [['T426'],['source_sync']]
    T.install()
    if sys.argv[1]=='T426':
        T.B.require_stopped=stopped;T.B.GROUPS['T426']=['t426_joint_retention'];import small_parts;small_parts.main('T426')
    else:
        import sync_source;sync_source.main()

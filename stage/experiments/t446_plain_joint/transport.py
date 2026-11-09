"""Completed T446 raw archive/source only; no scientific execution."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
def stopped(folder):
    assert folder.name=='t446_plain_joint';v=json.loads((folder/'SEALED.json').read_text());assert v['all_scientific_processes_stopped'];assert all(not Path('/proc',str(pid)).exists() for pid in v['scientific_pids'])
if __name__=='__main__':
    T.install()
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        assert sys.argv[1]=='T446';T.B.GROUPS['T446']=['t446_plain_joint'];T.B.require_stopped=stopped
        import small_parts
        small_parts.main('T446')

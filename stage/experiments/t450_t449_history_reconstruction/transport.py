"""Only sealed T450 historical-evidence transport or tracked source sync."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
def stopped(folder):
    assert folder.name=='t450_t449_history_reconstruction';s=json.loads((folder/'summary.json').read_text());assert not Path('/proc',str(s['pid'])).exists()
if __name__=='__main__':
    T.install()
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        assert sys.argv[1]=='T450';T.B.GROUPS['T450']=['t450_t449_history_reconstruction'];T.B.require_stopped=stopped
        import small_parts
        small_parts.main('T450')

"""Archive the immutable complete relay; driver only writes docs after sealing."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T


def stopped(folder):
    assert folder==ROOT/'stage/runs/t465_matched_safe_recovery'
    d=json.loads((folder/'SEALED.json').read_text())
    assert d['all_scientific_processes_stopped'] and d['finalizer_writes_only_docs_after_this_point']
    assert all(not Path('/proc',str(pid)).exists() for pid in d['scientific_pids'])


if __name__=='__main__':
    T.install()
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        assert sys.argv[1]=='T465'
        T.B.GROUPS['T465']=['t465_matched_safe_recovery'];T.B.require_stopped=stopped
        import small_parts
        small_parts.main('T465')

"""Archive ended T444 failure and T445 diagnostic separately, never active files."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
def stopped(folder):
    assert folder.name in ['t444_paired_retention','t445_minimal_retention']
    s=json.loads((folder/'SEALED.json').read_text());assert s['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(pid)).exists() for pid in s['scientific_pids'])
if __name__=='__main__':
    T.install()
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        group=sys.argv[1];assert group in ['T444','T445'];T.B.GROUPS[group]=['t444_paired_retention' if group=='T444' else 't445_minimal_retention'];T.B.require_stopped=stopped
        import small_parts
        small_parts.main(group)

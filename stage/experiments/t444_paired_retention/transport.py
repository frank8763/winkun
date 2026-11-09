"""Only completed T444 evidence and exact Stage source transport."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
def stopped(folder):
    assert folder.name=='t444_paired_retention'
    sealed=json.loads((folder/'SEALED.json').read_text());assert sealed['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(pid)).exists() for pid in sealed['scientific_pids'])
if __name__=='__main__':
    T.install()
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        assert sys.argv[1]=='T444';T.B.GROUPS['T444']=['t444_paired_retention'];T.B.require_stopped=stopped
        import small_parts
        small_parts.main('T444')

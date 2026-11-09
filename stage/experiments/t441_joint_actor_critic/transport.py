"""T441 completed, sealed evidence only; source or RAM archive transport."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T

def stopped(folder):
    assert folder.name=='t441_joint_actor_critic'
    seal=json.loads((folder/'SEALED.json').read_text())
    assert seal['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])

if __name__=='__main__':
    T.install()
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        assert sys.argv[1]=='T441'
        T.B.GROUPS['T441']=['t441_joint_actor_critic'];T.B.require_stopped=stopped
        import small_parts
        small_parts.main('T441')

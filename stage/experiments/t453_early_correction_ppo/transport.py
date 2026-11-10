"""Completed T453 raw/member backup and exact tracked source."""
from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T


def stopped(folder):
    assert folder.name == 't453_early_correction_ppo'
    s = json.loads((folder/'SEALED.json').read_text())
    assert s['all_scientific_processes_stopped']
    assert all(not Path('/proc', str(pid)).exists() for pid in s['scientific_pids'])


if __name__ == '__main__':
    T.install()
    if sys.argv[1] == 'source_sync':
        import sync_source
        sync_source.main()
    else:
        assert sys.argv[1] == 'T453'
        T.B.GROUPS['T453'] = ['t453_early_correction_ppo']
        T.B.require_stopped = stopped
        import small_parts
        small_parts.main('T453')

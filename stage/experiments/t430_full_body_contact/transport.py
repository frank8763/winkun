"""Post-completion RAM backup and exact source sync; no scientific execution."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage/experiments/t377_full_human_observation_transfer'))
import transport as T


def stopped(folder):
    assert folder.name == 't430_full_body_contact'
    read = lambda file: json.loads(Path(file).read_text())
    assert read(folder / 'SEALED.json')['all_scientific_processes_stopped']
    summary = read(folder / 'summary.json')
    audit = read(folder / 'audit.json')
    assert summary['state'] == 'COMPLETE' and audit['state'] == 'PASS'
    assert summary['physics'] == summary['Adam'] == 0
    assert not Path('/proc', str(summary['pid'])).exists()
    for path, value in read(folder / 'protocol.json')['inputs'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == value, path


if __name__ == '__main__':
    assert sys.argv[1:] in [['T430'], ['source_sync']]
    T.install()
    if sys.argv[1] == 'T430':
        T.B.require_stopped = stopped
        T.B.GROUPS['T430'] = ['t430_full_body_contact']
        import small_parts
        small_parts.main('T430')
    else:
        import sync_source
        sync_source.main()

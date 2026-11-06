"""Transport-only closeout of the stopped T387 offline chain.

Reuse the existing RAM archive and full remote member verification. No native
simulation, fitting, optimizer reconstruction, data compaction, or deletion.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage/experiments/t377_full_human_observation_transfer'))
import transport as T


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absent(pid):
    assert not Path('/proc', str(pid)).exists(), pid


original_stopped = T.B.require_stopped


def stopped(directory):
    if directory.name != 't387_full_horizon_night':
        return original_stopped(directory)
    docs = ROOT / 'stage/docs/overnight_20261105_revision'
    final = read(docs / 'FINAL_REPORT.json')
    assert final['state'] == 'FINITE_FULL_TASK_CHAIN_COMPLETE'
    assert final['complete'] and final['all_child_groups_stopped']
    assert len(final['completed']) == 3 and not final['seven_eight']
    assert not (docs / 'NIGHT_ERROR.json').exists()
    absent(read(directory / 'launch.json')['pid'])
    absent(read(directory / 'driver_started.json')['pid'])
    absent(read(docs / 'CURRENT_PROCESS.json')['pid'])
    for path, expected in read(directory / 'protocol.json')['inputs'].items():
        assert sha(path) == expected, path
    physics = commands = 0
    for folder in ['action_residual', 'phase_warp', 'student']:
        out = directory / folder
        summary = read(out / 'summary.json')
        audit = read(out / 'audit.json')
        assert summary['state'] == 'COMPLETE'
        assert summary['all_workers_stopped'] and audit['state'] == 'PASS'
        assert audit['native'] == 0
        absent(summary['pid'])
        workers = read(out / 'workers.json')
        for pid in [workers.get('parent'), *workers.get('workers', [])]:
            if pid:
                absent(pid)
        assert audit['physics'] == summary['physics']
        assert audit['commands'] == summary['commands']
        physics += summary['physics']
        commands += summary['commands']
        for row in read(out / 'local_manifest.json')['files']:
            assert sha(ROOT / row['path']) == row['sha256'], row['path']
    prior = read(directory / 'protocol.json')['prior_actual']
    assert final['actual'] == dict(physics=physics + prior['physics'],
                                 commands=commands + prior['commands'])
    for row in read(directory / 'local_manifest.json')['files']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']


if __name__ == '__main__':
    T.install()
    T.B.require_stopped = stopped
    if sys.argv[1:] == ['T387-NIGHT']:
        T.B.GROUPS['T387-NIGHT'] = ['t387_full_horizon_night']
        T.B.main('T387-NIGHT')
    elif sys.argv[1:] == ['source_sync']:
        import sync_source
        sync_source.main()
    else:
        raise ValueError('Stopped T387 evidence or exact source sync only')

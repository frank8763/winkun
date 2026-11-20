"""Post-science full raw RAM archives, no deletion and no science networking."""
from pathlib import Path
import json
import importlib.util
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'stage/runs/t661_value_state_retention'
sys.path.insert(0, str(ROOT/'stage/experiments/t654_verified_deviation_recovery'))
spec = importlib.util.spec_from_file_location('_t659r_archive_prior', ROOT/'stage/experiments/t654_verified_deviation_recovery/transport.py')
Prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Prior)


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    deadline = min(time.time()+p['backup_seconds']-5, p['absolute_all_stop']-p['source_seconds']-20)
    def stopped():
        sealed = json.loads((OUT/'SEALED.json').read_text())
        assert sealed['all_scientific_processes_stopped']
        assert all(not Path('/proc', str(pid)).exists() for pid in sealed['scientific_pids'])
    stopped()
    Prior.A.stopped = stopped
    def timeout(*args):
        raise TimeoutError('T661 bounded post-science transport')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1, int(deadline-time.time())))
    client = Prior.client()
    original_write = client.B.write
    def write(path, value):
        if Path(path).name.startswith('source_manifest_T661_'):
            value = dict(value, scope='Ended bounded offline T661 joint BC and fresh original-start RL relay; no universal nonforgetting claim')
        return original_write(path, value)
    client.B.write = write
    release = json.loads(client.req('GET', 'https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'))
    files = sorted(f for folder in [OUT] for f in folder.rglob('*') if f.is_file() and f.name not in ['driver.log','affinity.log']
                   and f.suffix not in ['.pyc','.lock','.partial'])
    batches, batch, n = [], [], 0
    for file in files:
        if batch and n+file.stat().st_size > 32*2**20:
            batches.append(batch)
            batch, n = [], 0
        batch.append(file)
        n += file.stat().st_size
    if batch:
        batches.append(batch)
    receipts = [Prior.A.archive(f'T661_raw_{i}', group, release, deadline, client) for i, group in enumerate(batches)]
    client.B.write(ROOT/'stage/docs/long_distance_20261101/release_receipt_T661.json',
        dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED', group='T661',
             child_receipts=[r['group'] for r in receipts], bytes=sum(r['bytes'] for r in receipts), files_deleted=0))
    signal.alarm(0)
    print('T661_FULL_RAW_REMOTE_VERIFIED')


if __name__ == '__main__':
    main()

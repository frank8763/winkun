"""Transport-only GitHub CLI fallback for completed T453 evidence.

Uses the existing credential store, streams binary assets only through RAM,
and retains the original archive/member SHA checks and original time budget.
"""
from pathlib import Path
import json, subprocess, sys, time, urllib.error

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'stage/runs/t453_early_correction_ppo'
DOC = ROOT/'stage/docs/review_20261110'
sys.path.insert(0, str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T


def read(f): return json.loads(f.read_text())


def stopped(folder):
    assert folder == OUT
    s = read(OUT/'SEALED.json')
    assert s['all_scientific_processes_stopped']
    assert all(not Path('/proc', str(pid)).exists() for pid in s['scientific_pids'])
    assert not Path('/proc', str(read(OUT/'driver_started.json')['pid'])).exists()


END = read(OUT/'SEALED.json')['epoch']+read(OUT/'protocol.json')['backup_seconds']


def call(method, url, body=None, headers=None):
    assert time.time() < END-3
    args = [str(T.A.GH), 'api', '--method', method, url]
    for key, value in (headers or {}).items():
        assert key.lower() != 'authorization'
        args.extend(['--header', key+': '+value])
    if body is not None:
        args.extend(['--input', '-'])
    result = subprocess.run(args, input=body, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=max(1, min(35, END-time.time()-2)))
    if result.returncode:
        raise urllib.error.URLError('GitHub CLI transport exit '+str(result.returncode))
    return result.stdout


def req(method, url, payload=None, headers=None):
    # Binary upload worked with the original curl transport; only metadata and
    # asset readback use the independent CLI HTTP client after TLS failures.
    if method != 'GET':
        return T.req(method, url, payload, headers)
    fields = {'Accept': 'application/vnd.github+json', **(headers or {})}
    if isinstance(payload, dict):
        payload = json.dumps(payload).encode()
        fields['Content-Type'] = 'application/json'
    return call(method, url, payload, fields)


def readback(asset, size):
    data = call('GET', f'https://api.github.com/repos/{T.A.SLUG}/releases/assets/{asset["id"]}',
                headers={'Accept': 'application/octet-stream'})
    assert len(data) == size, (len(data), size)
    return data


if __name__ == '__main__':
    stopped(OUT)
    assert not (ROOT/'stage/docs/long_distance_20261101/release_receipt_T453.json').exists()
    T.A.req = req
    T.B.req = req
    T.B.readback = readback
    T.B.GROUPS['T453'] = [OUT.name]
    T.B.require_stopped = stopped
    import small_parts
    small_parts.main('T453')

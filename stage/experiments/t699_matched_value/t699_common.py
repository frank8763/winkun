"""New, bounded research comparison. Historical runs are read-only inputs."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / 'stage/docs/overnight_20261122_23'
OUT = ROOT / 'stage/runs/t699_matched_value'
WEEKEND = ROOT / 'stage/docs/weekend_20261120_23'
sys.path.insert(0, str(ROOT / 'stage/experiments/t698_cross_representation_success'))
import t698_resources as Resources
from status_io import save, read_status

def read(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict) and '_input_manifest' in value:
        manifest = value['_input_manifest']
        assert sha(manifest['path']) == manifest['sha256'], 'Input manifest changed'
        entries = json.loads(Path(manifest['path']).read_text())
        value['inputs'] = {r[0]: r[1] for r in entries}
        value['input_stats'] = {r[0]: r[2:] for r in entries}
    return value

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def used(folder):
    return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())

def resources():
    return Resources.resources()

def immutable(p, full=False):
    assert time.time() < p['absolute_all_stop'], 'Absolute final deadline'
    assert sha(HERE/'protocol.json') == sha(OUT/'protocol.json')
    assert used(OUT) < p['max_bytes']
    assert used(ROOT/'stage/runs') - used(OUT) + p['max_bytes'] < p['global_bytes']
    assert shutil.disk_usage(ROOT).free > 40*2**30
    resources()
    for f, h in p['inputs'].items():
        q = Path(f)
        assert [q.stat().st_size, q.stat().st_mtime_ns] == p['input_stats'][f], f
        if full:
            assert sha(q) == h, f

def signal_child(pid, sig):
    import signal
    try:
        assert pid > 1 and os.getpgid(pid) == pid
        assert '/stage-t699.service' in Path('/proc', str(pid), 'cgroup').read_text()
        os.killpg(pid, sig)
    except (FileNotFoundError, ProcessLookupError):
        pass

def helper():
    import signal
    p = read(OUT/'protocol.json')
    os.sched_setaffinity(0, {15})
    cg = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0:'))
    assert cg.endswith('/stage-t699.service')
    cg = Path('/sys/fs/cgroup' + cg)
    save(DOC/'AFFINITY_GUARD.json', dict(state='ACTIVE_RESOURCE_ONLY', pid=os.getpid(), cgroup=str(cg)))
    while time.time() < p['absolute_all_stop'] and not (OUT/'HELPER_STOP').exists():
        status = read_status(OUT/'status.json')
        try:
            resources(); clear = True
        except AssertionError as e:
            if not isinstance(e.args[0], tuple) or not str(e.args[0][0]).startswith('Phase physical CPU overlap'):
                raise
            clear = False
        if status.get('child_pid'):
            signal_child(status['child_pid'], signal.SIGCONT if clear else signal.SIGSTOP)
        for pid in (cg/'cgroup.procs').read_text().split():
            try:
                for thread in Path('/proc', pid, 'task').iterdir():
                    tid = int(thread.name); mask = os.sched_getaffinity(tid)
                    if not mask <= set(range(8,16)):
                        os.sched_setaffinity(tid, mask & set(range(8,16)) or {15})
            except (FileNotFoundError, ProcessLookupError):
                pass
        time.sleep(.5)
    save(DOC/'AFFINITY_GUARD.json', dict(state='ENDED', pid=os.getpid(), epoch=time.time()))

if __name__ == '__main__':
    helper()

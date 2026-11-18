"""Frozen T501 compatibility reproduction, using its original scientific path.

No optimiser, teacher, action transform, task router, or new policy selection.
Each native occurrence has a unique full-array path, including exact replays.
"""
from pathlib import Path
import ast
import copy
import fcntl
import gzip
import hashlib
import io
import json
import lzma
import os
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t636_joint_best_reproduction'
DOC = ROOT / 'stage/docs/day_20261118/t636'
sys.path.insert(0, str(ROOT / 'stage/experiments/t501_joint_command_repair'))
import t501_learning as L
G, P, E, torch, np = L.G, L.P, L.G.E, L.torch, L.np
read, save, sha, digest = G.read, G.save, G.sha, G.digest
IDS = ['dev_task6_24', 'dev_task7_13', 'wide117_80_dev6_LEFT_18',
       'wide117_80_dev6_RIGHT_00', 'wide117_8m_dev6_LEFT_13',
       'wide117_8m_dev6_RIGHT_18']
ORIGINAL = ROOT / 'stage/runs/t501_joint_command_repair/protocol.json.t605.xz'
CASES = ROOT / 'stage/runs/t501_joint_command_repair/joint/cases.jsonl.t595.xz'


def used(folder):
    return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())


def plan():
    return read(HERE / 'protocol.json')


def model_check(p):
    m = p['initial']
    assert sha(m['path']) == m['sha256']
    saved = P.checkpoint(m['path'])
    assert digest(saved['joint_state']) == m['joint_hash']
    q = L.frozen(p, m['path'])
    assert q.phash() == m['policy_hash'] and not any(v.requires_grad for v in q.parameters())
    assert not torch.cuda.is_initialized()
    return q


def verify(p):
    assert p['authorization_approved'] and time.time() < p['absolute_all_stop']
    assert (OUT / 'protocol.json').read_bytes() == (HERE / 'protocol.json').read_bytes()
    assert used(OUT) <= p['max_bytes']
    assert used(ROOT / 'stage/runs') - used(OUT) + p['max_bytes'] + 131072 < 2**30
    assert shutil.disk_usage(ROOT).free >= 40 * 2**30
    for f, h in p['inputs'].items():
        assert sha(f) == h, f
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()


def setup(p):
    os.sched_setaffinity = G._real_affinity
    os.sched_setaffinity(0, set(p['stage_cpus']))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    P.frozen = L.frozen


def worker(pipe, index, p, counts):
    # Original T501 -> T478 -> T477 -> T449 native path, with CPU placement only
    # changed. In particular do not import later recovery/action-readout layers.
    real = G._real_affinity
    def pin(pid, mask):
        return real(pid, {p['stage_cpus'][index]}) if pid == 0 and set(mask) == {index} else real(pid, mask)
    os.sched_setaffinity = pin
    P.frozen = L.frozen
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    G.T.worker(pipe, index, p, counts)


class Pool:
    def __init__(self, p, counts, guard):
        self.p, self.guard = p, guard
        self.children, self.pipes = [], []
        ctx = G.R.M.mp.get_context('spawn')
        try:
            for i in range(3):
                a, b = ctx.Pipe()
                c = ctx.Process(target=worker, args=(b, i, p, counts))
                c.start(); b.close()
                self.children.append(c); self.pipes.append(a)
            # Persist process IDs even if initialization later fails.
            save(OUT / 'worker_generations.json', [dict(parent=os.getpid(), workers=[c.pid for c in self.children])])
            self.ready = [self.receive(i) for i in range(3)]
            actual = [sorted(os.sched_getaffinity(c.pid)) for c in self.children]
            assert actual == [[8], [10], [12]], actual
            assert all(v['state'] == 'READY' for v in self.ready)
            save(OUT / 'worker_generations.json', [dict(parent=os.getpid(), workers=[c.pid for c in self.children], actual_affinity=actual)])
        except BaseException:
            self.close()
            raise

    def receive(self, i):
        start = time.time()
        while not self.pipes[i].poll(.05):
            self.guard()
            assert self.children[i].is_alive() and time.time() - start < 300, 'Native worker died/stalled'
        value = self.pipes[i].recv()
        if isinstance(value, dict) and value.get('state') == 'FAILED':
            raise RuntimeError(value['error'])
        return value

    def request(self, items):
        assert 1 <= len(items) <= 3
        for i, item in enumerate(items):
            self.pipes[i].send(item)
        result = []
        for i in range(len(items)):
            a, r = self.receive(i)
            # Preserve original worker metadata before derived assertions.
            name = Path(items[i]['full_path']).stem
            with gzip.open(OUT / ('native_' + name + '.json.gz'), 'xb') as f:
                f.write(json.dumps(dict(request=items[i],native_result=r)).encode())
            result.append((a, r))
        return result

    def close(self):
        for c, pipe in zip(self.children, self.pipes):
            if c.is_alive():
                try: pipe.send(None)
                except (EOFError, BrokenPipeError): pass
        for c in self.children:
            c.join(5)
            if c.is_alive(): c.terminate(); c.join(1)
            if c.is_alive(): c.kill(); c.join()
        for pipe in self.pipes: pipe.close()
        assert all(not c.is_alive() for c in self.children)


def compare(r, original):
    assert r['policy_hash'] == original['policy_hash']
    assert r['array_hashes'] == original['array_hashes'], 'Historical per-array mismatch'
    assert r['array_digest'] == original['array_digest']
    assert r['integration_sha256'] == original['integration_sha256']
    for k in ['complete', 'success', 'collision', 'timeout', 'physics', 'remaining_m']:
        assert r[k] == original[k], k


def full_audit(p, r, a):
    assert set(a) == set(r['array_hashes'])
    assert all(digest(v) == r['array_hashes'][k] for k, v in a.items())
    assert digest(a) == r['array_digest']
    assert int(a['native_counts'].sum()) == r['physics']
    assert not a['ownership'].any() and not a['stochastic'].any()
    assert not a['end_flags'][:-1].any() and a['end_flags'][-1].tolist() == [0, 1, 0]
    assert np.array_equal(a['states'][1:], a['next_states'][:-1])
    assert np.array_equal(a['states'][1:, 117:119], a['actions'][:-1])
    P.frozen = L.frozen
    G.M.audit_full_history(p, r, a)


def main():
    p = plan(); verify(p); setup(p); E.N.offline(); E.L.install_stop_handlers()
    lock = (ROOT / 'stage/runs/.stage_writer.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'launch.json').exists()
    save(OUT / 'launch.json', dict(pid=os.getpid(), epoch=time.time()))
    model_check(p)
    counts = G.R.M.mp.get_context('spawn').RawArray('q', 4)
    start = time.time(); end = min(start + p['main_seconds'], p['absolute_native_stop'])
    rows = []; commands = 3; pool = None; error = None; last = [0.]
    def guard():
        assert time.time() < end - 5 and sum(counts) <= p['max_physics']
        if time.time() - last[0] > 5:
            last[0] = time.time(); verify(p)
            save(OUT / 'status.json', dict(state='FROZEN_NATIVE', pid=os.getpid(), epoch=time.time(), physics=sum(counts), commands=commands, completed=len(rows)))
    try:
        pool = Pool(dict(p, out=str(OUT), native_stop_epoch=end), counts, guard)
        for phase in ['fresh', 'exact']:
            for off in range(0, 6, 3):
                originals = p['reference_rows'][off:off+3]
                assert sum(counts) + sum(r['request']['job']['case']['step_cap'] for r in originals) <= p['max_physics']
                assert commands + len(originals) + 3 <= p['max_commands']
                requests = [dict(model=p['initial']['path'], job=copy.deepcopy(r['request']['job']), seed=None,
                                 query=False, full_path=str(OUT / 'episodes' / f'{phase}_{off+i:02d}.npz')) for i,r in enumerate(originals)]
                assert all(not Path(r['full_path']).exists() for r in requests)
                commands += len(requests)
                for (a,r),req,old in zip(pool.request(requests),requests,originals):
                    path = Path(req['full_path'])
                    r.update(job_id=req['job']['id'], request=req, seed=None, role=phase, path=str(path),
                             sha256=sha(path), loaded_model_sha256=p['initial']['sha256'], full=True)
                    # Save the failure-bearing raw before any old-history check.
                    with (OUT / 'cases.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
                    rows.append(r)
                    assert path.stat().st_size <= p['full_bytes']
                    compare(r, old); full_audit(p,r,a)
        assert len(rows) == 12
    except BaseException:
        error = traceback.format_exc()
        save(OUT / 'failure.json', dict(error=error, no_restart=True, raw_preserved=True))
    finally:
        if pool: pool.close()
        commands += 3
        save(OUT / 'summary.json', dict(state='FAILED' if error else 'COMPLETE', error=error, pid=os.getpid(),
             physics=sum(counts), commands=commands, elapsed_s=time.time()-start,
             completed=len(rows), fresh=sum(r['role']=='fresh' for r in rows), exact=sum(r['role']=='exact' for r in rows),
             new_Actor=0,new_Critic=0,new_optimizer=0,full407=None,no_global_promotion=True))
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    assert error is None, error


def audit():
    p=plan();verify(p);setup(p);model_check(p)
    s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    ids=[s['pid']]+[v for g in read(OUT/'worker_generations.json') for v in [g['parent'],*g['workers']]]
    assert all(not Path('/proc',str(v)).exists() for v in ids)
    rows=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()]
    assert len(rows)==12 and len({r['path'] for r in rows})==12
    for i,r in enumerate(rows):
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        compare(r,p['reference_rows'][i%6]);full_audit(p,r,a)
    assert s['physics']==sum(r['physics'] for r in rows)<=p['max_physics']
    assert s['commands']==18==p['max_commands']
    model_check(p)
    save(OUT/'audit.json',dict(state='PASS',new_native=0,new_optimizer=0,all12_full_history_density_reward_rebuilt=True,
         all6_original_arrays_and_integrations_exact=True,unique_paths=True,model_bytes_unchanged=True,
         scope='Six known cases only. Historical54/54 not freshly retested; no generalization or candidate promotion.'))


def prepare():
    assert not OUT.exists(), 'Do not duplicate/restart experiment'
    DOC.mkdir(parents=True,exist_ok=True)
    old=json.loads(lzma.decompress(ORIGINAL.read_bytes()))
    rr=[json.loads(z) for z in lzma.decompress(CASES.read_bytes()).decode().splitlines()]
    refs=[next(r for r in rr if r['job_id']==key) for key in IDS]
    cfg={k:copy.deepcopy(old[k]) for k in ['source_protocol','input_mean','input_scale','actor_seed','adapter_seed','steer_seed','raw_sigma','gamma_native','lambda_native']}
    cfg['initial']=read(ROOT/'stage/runs/t502_fullfield_probe/LOCKED_BEST.json')['current_joint']
    cfg.update(id='T636_FROZEN_JOINT_BEST_REPRODUCTION',out=str(OUT),reference_rows=refs,
       stage_cpus=[8,10,12,14],full_bytes=2**21,max_bytes=32*2**20,max_physics=466672,max_commands=18,
       main_seconds=900,audit_seconds=450,transport_seconds=300,source_seconds=180,all_seconds=3600,
       new_Actor=0,new_Critic=0,new_optimizer=0,authorization_approved=False,inputs={})
    assert 2*sum(r['request']['job']['case']['step_cap'] for r in refs)==cfg['max_physics']
    assert all(r['success'] and not r['collision'] for r in refs)
    setup(cfg);q=model_check(cfg);before=digest(q.state_dict())
    # Genuine retained native observations, never a synthesized navigation run.
    fixture=Path(old['old_states']);a=E.N.S.arrays(fixture)
    x=torch.tensor(a['states'][:32]);q.sample=False;q.reset_noise();h=None;commands=[]
    with torch.no_grad():
        expected=q.sequence(x).clamp(-1,1).float()
        for z in x:
            u,_,h,_=q.step(z[None],h);commands.append(u[0])
    difference=float((torch.stack(commands)-expected).abs().max())
    assert difference<=2e-6 and digest(q.state_dict())==before
    assert h is not None and bool(h.abs().max()>0)
    # NPZ serialization must preserve every array and use unique full paths.
    full_source=read(ROOT/'stage/runs/t634_execution_mean_probe/summary.json')['result']['modes']['mode'][0]['path']
    full_fixture=E.N.S.arrays(full_source)
    schema=set(refs[0]['array_hashes'])
    a={k:full_fixture[k] for k in schema}
    b=io.BytesIO();np.savez_compressed(b,**E.N.S.packed(a))
    restored=E.N.S.arrays(io.BytesIO(b.getvalue()))
    assert digest(restored)==digest(a)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    # Bound full output without assuming compression. 534 decisions at maximum
    # cap; empirical fixture shapes are native schema, plus128KiB ZIP/header.
    max_n=534
    bound=sum(v.dtype.itemsize * (max_n*int(np.prod(v.shape[1:])) if v.ndim and v.shape[0]==len(a['states']) else v.size) for v in a.values())+131072
    assert bound<=cfg['full_bytes'],bound
    own_upper=15*cfg['full_bytes']+2*2**20
    global_worst=used(ROOT/'stage/runs')+cfg['max_bytes']+131072
    assert own_upper<=cfg['max_bytes'] and global_worst<2**30
    # Source config's physics dependencies and all imported repo code are locked.
    science=E.N.science(cfg)
    files={ORIGINAL,CASES,fixture,Path(full_source),Path(cfg['source_protocol']),Path(cfg['initial']['path'])}
    files|={Path(r['request']['job']['exact_snapshot']) for r in refs}
    files|={Path(f) for f in science['input_hashes']}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    files|=set(HERE.glob('*.py'))|{DOC/'PLAN.md'}
    cfg['inputs']={str(f):sha(f) for f in sorted(files)}
    save(HERE/'protocol.json',cfg)
    save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',original_model=cfg['initial'],
         genuine_fixture=str(fixture),fixture_sha256=sha(fixture),fixture_states=32,step_sequence_max_error=difference,
         serialization_fixture=full_source,serialization_fixture_sha256=sha(full_source),
         prior_preflight_error='Zero-native fixture lacked next_states; serialization now uses a genuine full native trajectory, projected to original T501 schema. No simulation or optimizer was run.',
         actual_nonzero_carry=True,serialization_roundtrip_exact=True,per_full_uncompressed_upper=bound,
         own_storage_upper=own_upper,global_worst=global_worst,free_bytes=shutil.disk_usage(ROOT).free,
         original_native_path=True,current_recovery_layers_not_loaded=True,new_native=0,new_optimizer=0))
    save(DOC/'BUDGET.json',{k:cfg[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','transport_seconds','source_seconds','all_seconds','new_optimizer']})
    print('T636_PREPARED_NOT_REGISTERED_NOT_LAUNCHED',global_worst,difference,bound,flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()

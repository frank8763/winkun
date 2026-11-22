"""Canonical original-start execution and exact protected/source physics."""
import os
import t696_actor as A
import t694_native as Parent

C, G, F, P, np = Parent.C, Parent.G, Parent.F, Parent.P, Parent.np
ORIGINAL_INSPECT = Parent.ORIGINAL_INSPECT
PHYSICAL_KEYS = Parent.PHYSICAL_KEYS


def worker(pipe, index, p, counts):
    C.ACTIVE = p
    real = G._real_affinity
    def pin(pid, mask):
        return real(pid, {p['stage_cpus'][index]}) if pid == 0 and set(mask) == {index} else real(pid, mask)
    os.sched_setaffinity = pin
    G.T.Session = C.Session
    G.B.native_setup = lambda folder, pc: C.C.consistent_reset.setup(F, folder, pc, C.C.CANONICAL_SETUP)
    P.frozen = A.frozen
    G.T.worker(pipe, index, p, counts)


def reference(p, key):
    found = [r for r in p['protected403_rows'] + p['novel_rows'] if r['id'] == key]
    assert len(found) <= 1
    return found[0] if found else None


def inspect(p, arrays, row):
    ORIGINAL_INSPECT(p, arrays, row)
    ref = reference(p, row['job_id'])
    if ref:
        assert A.sha(ref['path']) == ref['sha256']
        source = A.L.T.R.E.N.S.arrays(ref['path'])
        assert all(np.array_equal(arrays[k], source[k]) for k in PHYSICAL_KEYS)
        assert A.same_float32(arrays['actions'], source['actions'])


def audit_source_hashes(p, row):
    ref = reference(p, row['job_id'])
    if ref:
        assert A.sha(ref['path']) == ref['sha256']
        source = A.L.T.R.E.N.S.arrays(ref['path'])
        assert all(row['array_hashes'][k] == A.digest(source[k]) for k in PHYSICAL_KEYS)

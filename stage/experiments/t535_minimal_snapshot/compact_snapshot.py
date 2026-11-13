"""Drop logging buffers, retain all dynamical state; native equivalence UNTESTED."""
import copy,gzip,pickle
BUFFER_KEYS=('rows','hist','hb','ha','proposal','owner','digests')
def compact(snapshot):
    s=copy.deepcopy(snapshot)
    for k in BUFFER_KEYS:s[k]=[]
    # Previous raw/location affect possible stateful sampling; keep both.
    # This fixed policy is deterministic iid, not AR exploration.
    assert len(s['qstate'])==3
    s['qstate']=(s['qstate'][0],s['qstate'][1],[])
    return s
def pack(s):return gzip.compress(pickle.dumps(s,protocol=5),mtime=0)
def unpack(b):return pickle.loads(gzip.decompress(b))
def verify_preserved(original,reduced,equal):
    assert set(original)==set(reduced)
    for k,v in original.items():
        if k in BUFFER_KEYS:assert reduced[k]==[]
        elif k=='qstate':
            assert equal(v[0],reduced[k][0]) and equal(v[1],reduced[k][1]) and reduced[k][2]==[]
        else:assert equal(v,reduced[k]),k

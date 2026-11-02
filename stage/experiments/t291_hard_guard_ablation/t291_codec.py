"""Losslessly store the exact redundant decision-state chain, not approximate states."""
import numpy as np

def pack_full(a):
    assert len(a)==12 and a['states'].dtype==a['next_states'].dtype==np.float32
    assert a['states'].shape==a['next_states'].shape and a['states'].shape[1]==120 and len(a['states'])<=280
    assert np.array_equal(a['states'][1:],a['next_states'][:-1])
    out={k:v for k,v in a.items() if k!='next_states'}
    out['last_next_state']=a['next_states'][-1:].copy()
    assert sum(v.nbytes for v in out.values())+8192<=192*1024
    return out

def unpack_full(a):
    if 'last_next_state' not in a:return a
    out={k:v for k,v in a.items() if k!='last_next_state'}
    out['next_states']=np.concatenate([out['states'][1:],a['last_next_state']],axis=0)
    assert len(out)==12
    return out

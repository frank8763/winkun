"""Recursive state comparison, including history deques and Generator state."""
import collections,math
import numpy as np
import torch

def equal(a,b):
    if isinstance(a,dict):
        return isinstance(b,dict) and set(a)==set(b) and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple,collections.deque)):
        return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b,equal_nan=True)
    if isinstance(a,np.random.Generator):return isinstance(b,np.random.Generator) and equal(a.bit_generator.state,b.bit_generator.state)
    if isinstance(a,(float,np.floating)) and math.isnan(a):return isinstance(b,(float,np.floating)) and math.isnan(b)
    if hasattr(a,'__dict__') and type(a)==type(b):return equal(vars(a),vars(b))
    return bool(a==b)

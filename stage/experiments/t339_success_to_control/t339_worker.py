from t339_core import OUT,Actor
import t338_worker as original

def worker(pipe,index,p,counts,preflight=False):
    original.OUT=OUT;original.Actor=Actor
    return original.worker(pipe,index,p,counts,preflight)

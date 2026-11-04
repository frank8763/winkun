from t338r_core import *
import t338_worker as W
def worker(pipe,index,p,counts,preflight=False):
    W.OUT=OUT;W.Actor=Actor
    return W.worker(pipe,index,p,counts,preflight)

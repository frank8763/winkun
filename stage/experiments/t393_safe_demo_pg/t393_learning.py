"""Command-preserving T255 initialization and physically cancellable sessions."""
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t392_demo_closed_loop'))
from t392_learning import *
import t392_learning as PREVIOUS
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t393_safe_demo_pg'

class Actor(PREVIOUS.Actor):
    def __init__(self,base,p):
        # Preserve frozen T255 hidden-reset path. Nonzero encoder/core features
        # remain available, but both trainable residual readouts start at zero.
        N.S.NightActor.__init__(self,base,0.)
        self.contract=p;self.sample=False;self.rng=None
        with torch.no_grad():
            self.head.weight.zero_();self.head.bias.zero_()
            self.adapter[-1].weight.zero_();self.adapter[-1].bias.zero_()
        self.reset_noise()

def learner(base,p):
    q=Actor(base,p).eval();params=N.trainable(q);assert sum(v.numel() for v in params)==6864
    return q,torch.optim.Adam(params,lr=p['lr'])

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes'])
    assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=1024**3
    return N.science(p)

class Session(N.R.Session):
    def __init__(self,job,native,create,witness,guard,physics,q=None):
        # Same original Session contract, generalized only to original old-task
        # caps. task_env receives the original cap; no changed native deadline.
        self.env,self.obs,_=N.K.R.task_env(create,job);self.obs=N.K.C.perturb_anchor(self.env,self.obs,job)
        self.w=witness;self.native=native;self.q=q;self.guard=guard;self.physics=physics
        self.steps=0;self.previous=np.zeros(2,np.float32);self.pn=0;self.hidden=None
        self.rows=[];self.hist=[];self.hb=[];self.ha=[];self.proposal=[];self.owner=[];self.digests=[]
        self.term=self.trunc=False;self.info={};self.cap=job['case']['step_cap']
        self.goal=self.env.goal_xy.copy();self.scene=N.K.C.model_hash(self.env);self.initial=N.K.R.integration(self.env).copy()
        assert not self.env._collision() and 0<self.cap<=40000 and len(self.initial)==203
        if q is not None:q.reset_noise()

class StageStop(BaseException):pass
STOP_REQUESTED=None

def check_stop():
    if STOP_REQUESTED is not None:raise StageStop('T393 stop requested '+str(STOP_REQUESTED))

def install_stop_handlers():
    def stop(signum,frame):
        global STOP_REQUESTED
        STOP_REQUESTED=signum
    # Native factory imports may replace handlers. Reinstall after setup and
    # after every environment construction, before any physical advance. Mark
    # only in the signal handler: finish/count the in-flight native call, then
    # check_stop raises outside native code before its next physical step.
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    return stop

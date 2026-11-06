"""Single-Actor recovery data contract; teacher is restricted to training."""
from pathlib import Path
import copy,gzip,hashlib,importlib.util,io,json,os,shutil,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t391_actual_student_recovery'
DOC=ROOT/'stage/docs/review_20261106'
spec=importlib.util.spec_from_file_location('stage391_prior_contract',ROOT/'stage/experiments/t387_full_horizon_night/night_common.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
np,torch,read,save,sha,digest,equal,hashes=N.np,N.torch,N.read,N.save,N.sha,N.digest,N.equal,N.hashes
PRIOR=ROOT/'stage/runs/t390r_worker_entry_recovery'
PARENT=PRIOR/'coherent_teacher_only/candidate.pt.gz'
SOURCE=PRIOR/'coherent_teacher_only/centre.npz'
REFERENCE=ROOT/'stage/runs/t387_full_horizon_night/phase_warp/mixed.npz'


def native_worker(pipe,index,p,counts):
    return N.native_worker(pipe,index,p,counts)


def actor(base):
    q=N.S.NightActor(base,0.).eval().requires_grad_(False)
    q.load_learned(N.S.read_model(PARENT));q.sample=False;q.reset_noise()
    assert q.phash()==read(PRIOR/'summary.json')['results']['coherent_teacher_only']['model']['policy_hash']
    return q


def learner(base):
    q=actor(base);params=N.trainable(q)
    assert sum(v.numel() for v in params)==6864
    return q,torch.optim.Adam(params,lr=1e-4)


def verify(p):
    for f,h in p['inputs'].items():assert sha(f)==h,f
    root=read(OUT/'protocol.json')
    assert N.K.S.used(OUT)<=root['final_bytes'] and N.K.S.used(ROOT/'stage/runs')<=1024**3
    assert shutil.disk_usage(ROOT).free>=40*1024**3
    return N.science(p)


def dataset(base,q,recovery_path,first):
    groups=[]
    for name,path,start in [('original_success',REFERENCE,0),('actual_student_recovery',recovery_path,first)]:
        a=N.S.arrays(path)
        assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        x=torch.tensor(a['states'])
        groups.append(dict(id=name,x=x,b=q.base_actions(x),target=torch.tensor(a['actions'],dtype=torch.float64),first=start,weight=.5))
    return groups,None


def fit(q,opt,groups,old,steps,guard,event):
    # Same full-history optimization as T390, with only two coherent sources.
    assert old is None
    return N.fit(q,opt,groups,steps,guard,event,None)


def parameter_hashes(q,opt):
    return {k:digest(v) for k,v in N.learned_state(q,opt).items()}

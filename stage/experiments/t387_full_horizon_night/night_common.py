"""Sealed local Stage chain. Runtime never requires network or model service."""
from pathlib import Path
import copy,gzip,hashlib,importlib.util,io,json,os,shutil,signal,socket,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/overnight_20261105_revision'
OUT=ROOT/'stage/runs/t387_full_horizon_night'
spec=importlib.util.spec_from_file_location('stage384_locked_teacher_contract',ROOT/'stage/experiments/t382_receding_human_teacher/run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
R,S,K,np,torch=C.R,C.S,C.K,C.np,C.torch
read,save,digest,sha,equal,hashes=C.read,C.save,C.digest,C.sha,C.equal,C.hashes
OLD=ROOT/'stage/runs/t382_receding_human_teacher'
NAMES=C.NAMES
def offline():
    connect=socket.socket.connect;connect_ex=socket.socket.connect_ex;create=socket.create_connection
    def local(address):return isinstance(address,(str,bytes)) or address[0] in ['127.0.0.1','::1','localhost']
    def safe_connect(s,address):
        assert local(address),'Offline Stage chain: external network disabled';return connect(s,address)
    def safe_ex(s,address):
        assert local(address),'Offline Stage chain: external network disabled';return connect_ex(s,address)
    def safe_create(address,*a,**k):
        assert local(address),'Offline Stage chain: external network disabled';return create(address,*a,**k)
    socket.create_connection=safe_create;socket.socket.connect=safe_connect;socket.socket.connect_ex=safe_ex
def write_npz(f,a,limit):R.write_npz(f,a,limit)
def science(p):
    base=read(p['source_protocol'])
    for f,h in base['input_hashes'].items():assert sha(f)==h,f
    return base
def configure(out):
    out=Path(out);C.OUT=out;R.OUT=out;S.P.C.C.OUT=out
def capacity(out,limit):
    assert K.S.used(out)<=limit and K.S.used(ROOT/'stage/runs')<=1024**3
    assert shutil.disk_usage(ROOT).free>=40*1024**3
def compact(p):
    n,c,h=p['max_replans'],6,20
    return dict(controls=np.zeros((n,c,h,2),np.float32),next_poses=np.full((n,c,h,3),np.nan),
        counts=np.zeros((n,c,h),np.uint8),flags=np.zeros((n,c,h,3),np.uint8),lengths=np.zeros((n,c),np.uint8),
        costs=np.full((n,c),np.nan),remaining_m=np.full((n,c),np.nan),trace_sha256=np.zeros((n,c,32),np.uint8),
        integration_sha256=np.zeros((n,c,32),np.uint8),selected=np.full(n,-1,np.int8),snapshot_sha256=np.zeros((n,32),np.uint8))
def refseq(human,index):
    ids=np.minimum(np.arange(index,index+20),len(human)-1);return human[ids].copy()
def choose(costs,flags,lengths):
    ids=[i for i,n in enumerate(lengths[:5]) if n and not flags[i,int(n)-1,0] and (n==20 or flags[i,int(n)-1,1])]
    return min(ids,key=lambda i:(float(costs[i]),i)) if ids else None
def manifest(out):
    out=Path(out);rows=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='local_manifest.json']
    save(out/'local_manifest.json',dict(state='LOCAL_ALL_BYTES_SHA_VERIFIED_REMOTE_PENDING',files=rows,network_required=False,remote_verified=False,epoch=time.time()))
    for r in rows:assert sha(ROOT/r['path'])==r['sha256']
def stopped(out):
    for f in ['summary.json','launch.json']:
        v=read(Path(out)/f)
        for key in ['pid','main_pid']:
            if v.get(key):assert not Path('/proc',str(v[key])).exists(),v[key]
def load_npz(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}
def native_worker(pipe,index,p,counts):
    """Restore standard streams after the historical worker closes its log.

    The original Stage worker leaves stdout/stderr referencing its closed log;
    multiprocessing's exit flush would then fail even after a normal shutdown.
    Preserve its computation and restore process streams at this outer boundary.
    """
    offline();out,err=sys.stdout,sys.stderr
    try:S.night_worker(pipe,index,p,counts)
    finally:sys.stdout,sys.stderr=out,err
def data(q,base,teacher,guard):
    if teacher is None:
        result=[]
        for name,path,first in [('human361',base['demonstration'],0),('human364',base['correction'],20)]:
            a=S.arrays(path);assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any();x=torch.tensor(a['states']);guard()
            result.append(dict(id=name,x=x,b=q.base_actions(x),target=torch.tensor(a['actions'],dtype=torch.float64),first=first,weight=.5))
        return result
    result=[]
    for name,path,first,weight in [('human361',base['demonstration'],0,.25),('human364',base['correction'],20,.25),('native_teacher',teacher,20,.5)]:
        a=S.arrays(path);assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any();x=torch.tensor(a['states']);guard()
        result.append(dict(id=name,x=x,b=q.base_actions(x),target=torch.tensor(a['actions'],dtype=torch.float64),first=first,weight=weight))
    return result
def trainable(q):
    q.requires_grad_(False)
    for module in [q.encoder,q.core,q.head,q.adapter]:module.requires_grad_(True)
    return [v for v in q.parameters() if v.requires_grad]
def learner(base):
    q=R.actor(base);params=trainable(q);assert sum(v.numel() for v in params)==6864
    return q,torch.optim.Adam(params,lr=1e-4)
def learned_state(q,opt):return dict(learned=q.learned(),optimizer=copy.deepcopy(opt.state_dict()))
def old_data(q,base,guard):
    old,_=S.P.C.dataset(dict(base,sequence_sources=[]),q,guard)
    for g in old:g.update(target=g['b'].clone(),first=0,weight=.1)
    return old
def fit(q,opt,groups,steps,guard,event,old=None):
    params=[v for v in q.parameters() if v.requires_grad];trace=[]
    immutable=digest({n:v for n,v in q.state_dict().items() if n.startswith('base.') or n.startswith('future_head.')})
    for k in range(steps):
        guard();opt.zero_grad(set_to_none=True);vals=[]
        for g in [*groups,*([old[k%len(old)]] if old is not None else [])]:
            loc=q.sequence(g['x'],g['b']);loss=g['weight']*(loc[g['first']:]-g['target'][g['first']:]).square().mean()
            loss.backward();vals.append(float(loss.detach()))
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k)
        assert immutable==digest({n:v for n,v in q.state_dict().items() if n.startswith('base.') or n.startswith('future_head.')})
        trace.append([*vals,norm])
    return np.asarray(trace,np.float64)

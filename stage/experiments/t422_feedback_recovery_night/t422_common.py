"""Shared bounded Stage-only offline contracts; no work at import time."""
from pathlib import Path
import sys,os,time,json,copy,hashlib,shutil,contextlib,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t422_feedback_recovery_night';DOC=ROOT/'stage/docs/overnight_20261107'
sys.path.insert(0,str(ROOT/'stage/experiments/t418_evening_repair_chain'))
import repair_core as E
import reference as R
torch,np=E.torch,E.np
read,save,sha,digest=E.read,E.save,E.sha,E.digest
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert E.N.K.S.used(OUT)<=p['final_bytes'] and E.N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
def setup(p,folder):
    E.set_out(Path(folder).parent)
    module,native,create,w,_=E.N.S.P.C.B.setup(folder,E.N.science(p))
    module.ACTIVE=copy.deepcopy(module.ACTIVE);module.ACTIVE['circle_shift_x_m']=p['job']['circle_shift_x_m']
    return native,create,w
def frozen(p,model=None):
    q=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False)
    if model is not None:q.load_learned(E.N.S.read_model(model))
    q.sample=False;q.reset_noise();return q
def snap_hash(s):return digest(dict(native=E.N.K.R.integration(s.env),packet=s.packet(),history=s.w.get(s.env),hidden=s.hidden,steps=s.steps,previous=s.previous,pn=s.pn,goal=s.goal,scene=s.scene))
def ban_learning():
    def no(*a,**k):raise AssertionError('Search/native process cannot perform learning')
    torch.optim.Adam.step=no;torch.optim.AdamW.step=no;torch.optim.SGD.step=no;torch.Tensor.backward=no
def control_score(row,a):
    smooth=float(np.square(np.diff(a['actions'].astype(np.float64),axis=0)).sum())
    return [0,int(a['native_counts'].sum()),smooth] if row['success'] and not row['collision'] else [2 if row['collision'] else 1,float(row['remaining_m']),smooth]
def finish_feedback(s,ref,first,gains,mode):
    progress=first;diags=[None]*first
    while not s.done():
        x=s.packet();action,progress,diag=R.feedback(ref,np.asarray(s.env._pose()[:3]),float(.5*x[113]),s.steps*.002,progress,gains,mode)
        diags.append(diag);s.step(action,owner=1,actor_active=True)
    a=E.payload(s);a['feedback_progress']=np.asarray([-1 if d is None else d['reference_index'] for d in diags],np.int16)
    a['feedback_speed']=np.asarray([0. if d is None else d['speed_mps'] for d in diags],np.float32)
    return a
def recovery(p,native,create,w,guard,charge,model,first,gains,mode,partial_path=None):
    q=frozen(p,model);s=E.L.Session(p['job'],native,create,w,guard,charge,q)
    try:
        ref=R.load_reference(p['reference_path'],p['reference_sha256'])
        assert E.equal(s.initial,ref['arrays']['initial_native_integration']) and E.equal(s.packet(),ref['arrays']['states'][0])
        for _ in range(first):s.step(owner=0)
        a=finish_feedback(s,ref,first,gains,mode);row=s.result();row.update(policy_hash=q.phash(),array_digest=digest(a),integration_sha256=digest(a['final_native_integration']),first=first,mode=mode,gains=list(gains),physics=int(a['native_counts'].sum()),student_model_sha256=sha(model))
        return a,row
    except BaseException:
        if partial_path is not None and s.rows:
            E.N.write_npz(partial_path,E.N.S.packed(E.payload(s)),p['query_full_bytes'])
        raise
    finally:s.close()
def compact_episode(a):return {k:a[k] for k in ['actions','poses','next_poses','native_counts','end_flags','rewards','hold_sha256','feedback_progress','feedback_speed','final_native_integration']}
def verify_feedback(a,p,first,gains,mode):
    ref=R.load_reference(p['reference_path'],p['reference_sha256']);progress=first;elapsed=np.r_[0,np.cumsum(a['native_counts'])]*.002
    for i in range(first,len(a['actions'])):
        action,progress,_=R.feedback(ref,a['poses'][i],float(a['feedback_speed'][i]),float(elapsed[i]),progress,gains,mode)
        assert E.equal(action,a['actions'][i]) and progress==int(a['feedback_progress'][i]),i
    assert np.all(a['ownership'][:first]==0) and np.all(a['ownership'][first:]==1)
    return True

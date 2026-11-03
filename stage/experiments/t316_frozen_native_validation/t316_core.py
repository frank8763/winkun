"""Four fixed offline candidates: frozen native validation, no optimization."""
from pathlib import Path
import sys,os,json,io,gzip,copy,time,shutil,traceback,signal,contextlib,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t315_full_trajectory_gradient'))
import t315_core as U
np,torch,K=U.np,U.torch,U.K
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t316_frozen_native_validation';DOC=ROOT/'stage/docs/distance_20261103'
read=U.read

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def policies(p):
    out={}
    for name,r in p['candidates'].items():
        assert K.C.sha(r['path'])==r['sha256'];payload=torch.load(io.BytesIO(gzip.decompress(Path(r['path']).read_bytes())),map_location='cpu',weights_only=False)
        q=K.policy(p).eval().requires_grad_(False);q.load_state_dict(payload['policy'])
        assert payload['accepted'] and K.S.digest(q.state_dict())==r['policy_hash'];out[name]=q
    assert len(out)==4;return out

def forbid_step(*_args,**_kw):raise AssertionError('T316 frozen validation forbids Adam.step')

def compact(a):return {k:a[k] for k in ['actions','native_counts','end_flags','environment_rewards']}

def outcome(r):
    if r.get('error'):return 'INVALID'
    if not r['complete']:return 'CENSORED'
    if r['collision']:return 'CONTACT'
    if r['success']:return 'GOAL'
    if r['timeout']:return 'TIMEOUT'
    return 'INVALID'

def counts(rows,p):
    result=[]
    for name in p['candidates']:
        for role,group,want in [('old_guard',None,20),('six_development',None,54),('six_development','6m_LEFT',27),('six_development','6m_RIGHT',27)]:
            rr=[r for r in rows if r['arm']==name and r['role']==role and (group is None or r['group']==group)]
            result.append(dict(candidate=name,role=role,group=group,planned=want,executed=len(rr),not_executed=want-len(rr),
                outcomes={k:sum(outcome(r)==k for r in rr) for k in ['GOAL','CONTACT','TIMEOUT','INVALID','CENSORED']}))
    return result

def worker(pipe,index,p,physical):
    sys.path.insert(0,str(ROOT/'stage/experiments/t312_online_goal_ppo'))
    import ppo_worker as original
    original.OUT=OUT;original.B.OUT=OUT
    torch.optim.Adam.step=forbid_step
    return original.worker(pipe,index,p,physical)

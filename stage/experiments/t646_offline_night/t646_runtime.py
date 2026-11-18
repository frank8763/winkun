"""Offline finite mechanisms reuse verified iterative recovery and acceptance."""
from pathlib import Path
import io,json,os,shutil,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t646_offline_night';DOC=ROOT/'stage/docs/overnight_20261118'
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'))
import t645_runtime as O
import t646_learning as M
R=O.R
O.M=M;R.M=M
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np


def plan():
    root=read(HERE/'protocol.json');name=sys.argv[2];assert name in root['arms']+['profile','coverage']
    child=read(OUT/name/'protocol.json');assert child['_root_sha256']==sha(OUT/'protocol.json')
    p=dict(root);p.update(child);p['out']=str(OUT/name);return p


def verify(p,*args):
    root=read(OUT/'protocol.json')
    assert root['authorization_approved'] and time.time()<root['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert p['_root_sha256']==sha(OUT/'protocol.json')
    assert R.used(OUT)<=root['max_bytes'] and R.used(Path(p['out']))<=p['max_bytes']
    assert R.used(ROOT/'stage/runs')-R.used(OUT)+root['max_bytes']+131072<10*2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in root['inputs'].items():assert sha(f)==h,f


def load(p,guard):
    manifest=read(OUT/'OFFLINE_INPUTS.json');assert manifest['state']=='ALL384_ORIGINAL_BYTES_READY_AND_SHA_VERIFIED'
    by={row['id']:row for row in manifest['members']};q=R.L.frozen(p,p['initial']['path']);data=[]
    for row in R.D.selected():
        guard();r=by[row['job_id']];assert sha(r['offline_path'])==row['sha256']==r['sha256']
        a=R.E.N.S.arrays(r['offline_path']);g=R.D.features(q,a,row)
        g['x']=torch.tensor(a['states']);data.append(g)
    assert len(data)==384 and {g['id'] for g in data}==set(p['before_success_ids'])
    return data


def configure():
    p=plan();name=p['arm'];folder=OUT/name
    O.HERE=HERE;O.OUT=folder;O.DOC=DOC/name;O.plan=plan
    R.HERE=HERE;R.OUT=folder;R.DOC=DOC/name;R.verify=verify;R.D.load=load
    # Original learning/evaluation/aggregation/rollback/audit functions remain
    # shared. Representation branches re-evaluate entire causal sequences.
    R.E.N.offline()
    def no_remote(*args,**kwargs):raise RuntimeError('Offline scientific input may not fetch network')
    R.D.remote_bytes=no_remote
    prior_setup=O.setup
    def setup(pc):
        prior_setup(pc);R.D.load=load
    O.setup=setup
    original_gpu_guard=R.gpu_guard;last=[0.]
    def gpu_guard():
        if time.time()-last[0]>=5:
            original_gpu_guard();last[0]=time.time()
    R.gpu_guard=gpu_guard
    return p


def run():
    action=sys.argv[1]
    if action.startswith('coverage_'):
        import t646_coverage as Coverage
        getattr(Coverage,action[9:])();return
    if action.startswith('profile_'):
        import t646_profile as Profile
        getattr(Profile,action[8:])();return
    p=configure();folder=Path(p['out'])
    if action=='verify_inputs':save(folder/'INPUT_AUDIT.json',O.input_check(p))
    else:getattr(O,action)()


if __name__=='__main__':run()

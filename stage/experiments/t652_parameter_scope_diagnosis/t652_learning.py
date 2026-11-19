"""Single-variable scope ablation of valid T651; existing two adapters only.

No runtime task routing, new parameters, teacher, or algorithm guarantee.
"""
from pathlib import Path
import copy,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t651_canonical_outcome_pg'))
import t651_learning as Prior
L,P,E,torch,np=Prior.L,Prior.P,Prior.E,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
state,load,export=Prior.state,Prior.load,Prior.export


def make(p,device='cuda'):
    q=L.frozen(p,p['initial']['path']).to(device).train().requires_grad_(False)
    q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    variables=[v for v in q.parameters() if v.requires_grad]
    assert sum(v.numel() for v in variables)==11906
    opt=torch.optim.Adam(variables,lr=p['fit_lr'],foreach=False,fused=False)
    assert not opt.state
    return q,opt


def fit(obj,old,groups,p,guard,event):
    body=L.fixed_hash(obj[0]);before=state(obj)
    result=Prior.fit(obj,old,groups,p,guard,event)
    assert L.fixed_hash(obj[0])==body
    for key,value in before['actor'].items():
        if not key.startswith(('drive_adapter.','steer_adapter.')):assert torch.equal(value,obj[0].state_dict()[key])
    return dict(**result,trainable_parameters=11906,encoder_GRU_head_frozen_exact=True)

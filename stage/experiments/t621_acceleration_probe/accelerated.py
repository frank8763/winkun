"""Device-aware T620 losses. No optimizer application in the hardware probe."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t620_recurrent_prior_td3'))
REQUESTED_CUDA=os.environ.get('CUDA_VISIBLE_DEVICES','')
import t620_learning as L
# The immutable legacy runtime import deliberately hides GPUs for T493.
# Restore only this new, user-authorized learner process's requested device.
os.environ['CUDA_VISIBLE_DEVICES']=REQUESTED_CUDA
torch=L.torch

def move(obj,device):
    for name in ['actor','critics','target_actor','target_critics']:obj[name].to(device)
    # cuDNN needs a training reserve for recurrent backward. This GRU has no
    # dropout, so training mode introduces no stochastic forward operation.
    assert obj['actor'].branch.core.dropout==0
    obj['actor'].branch.train()
    return obj

def pack_to(pack,device):
    return (pack[0],*[v.to(device) for v in pack[1:]])

def loss(obj,reference,pack,p,generator,actor=False):
    selected,x,t,b,u,reward,discount,mask=pack
    if not actor:
        with torch.no_grad():
            action=obj['target_actor'].sequence(x).clamp(-1,1)[t+1,b]
            # Preserve the original CPU RNG stream across devices.
            noise=torch.randn(action.shape,dtype=action.dtype,generator=generator).to(action.device)*p['target_noise']
            action=(action+noise.clamp(-p['target_noise_clip'],p['target_noise_clip'])).clamp(-1,1)
            qs=[q.values(q.features(x)[t+1,b],action) for q in obj['target_critics']]
            target=reward+discount*mask*torch.minimum(*qs)
        values=[q.values(q.features(x)[t,b],u) for q in obj['critics']]
        return sum(torch.nn.functional.mse_loss(v,target) for v in values)
    q=obj['critics'][0]
    with torch.no_grad():h=q.features(x)[t,b];anchor=reference.sequence(x).clamp(-1,1)[t,b]
    actions=obj['actor'].sequence(x).clamp(-1,1)[t,b]
    protected=torch.tensor([selected[int(i)]['id'] in p['before_success_ids'] for i in b.cpu()],dtype=torch.bool,device=x.device)
    gain=-q.values(h,actions).mean()
    retain=(actions[protected]-anchor[protected]).square().mean() if bool(protected.any()) else gain.new_zeros(())
    return gain+p['retention_weight']*retain

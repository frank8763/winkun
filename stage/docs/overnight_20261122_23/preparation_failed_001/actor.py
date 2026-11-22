"""Same frozen T694 parent; plain RBF or replay MLP residual. No task router."""
import copy, gzip, io, sys
from common import ROOT, read, sha
sys.path.insert(0, str(ROOT/'stage/experiments/t696_existing_success_compatibility'))
import t696_actor as Existing

torch, np, digest = Existing.torch, Existing.np, Existing.digest
Parent = Existing.Parent
ARCH = 'T699_frozen_T694_replay_MLP_residual'

class Actor(Parent.Actor):
    def __init__(self, p, old_count, width):
        super().__init__(p, old_count)
        self.replay_head = torch.nn.Sequential(torch.nn.Linear(184, width), torch.nn.Tanh(), torch.nn.Linear(width, 2)).double()
        self.requires_grad_(False)

    def location(self, h, n):
        base = super().location(h, n)
        return base + self.replay_head(torch.cat([h, n], -1))

    def phash(self):
        return digest(dict(architecture=ARCH, state=self.state_dict(), sigma=self.p['raw_sigma']))

def frozen(p, path):
    saved = Existing.Base.P.checkpoint(path)
    if saved.get('architecture') != ARCH:
        return Existing.frozen(p, path)
    state = saved['joint_state']['actor']
    q = Actor(p, len(state['compat_radii']), state['replay_head.0.weight'].shape[0])
    q.load_state_dict(state)
    assert q.phash() == saved['policy_hash']
    return q.eval()

def export(q, p, path, architecture=ARCH):
    joint = dict(actor=copy.deepcopy(q.state_dict()))
    buffer = io.BytesIO()
    torch.save(dict(joint_state=joint, architecture=architecture, policy_hash=q.phash(), joint_hash=digest(joint),
                    original_optimizer_not_reused=True, new_RL=0, runtime_teacher=False), buffer)
    blob = gzip.compress(buffer.getvalue(), mtime=0)
    assert len(blob) < 8*2**20
    with path.open('xb') as out:
        out.write(blob)
    return dict(path=str(path), sha256=sha(path), bytes=len(blob), policy_hash=q.phash(), joint_hash=digest(joint))

def make_head(width, seed):
    torch.manual_seed(seed)
    q = torch.nn.Sequential(torch.nn.Linear(184, width), torch.nn.Tanh(), torch.nn.Linear(width, 2)).double()
    torch.nn.init.zeros_(q[2].weight); torch.nn.init.zeros_(q[2].bias)
    return q

def attach(p, head):
    parent = Parent.frozen(p, p['initial']['path'])
    q = Actor(p, len(parent.compat_radii), head[0].out_features)
    missing, extra = q.load_state_dict(parent.state_dict(), strict=False)
    assert set(missing) == {'replay_head.'+k for k in head.state_dict()} and not extra
    q.replay_head.load_state_dict(head.state_dict())
    assert all(torch.equal(q.state_dict()[k], v) for k,v in parent.state_dict().items())
    return q.eval()

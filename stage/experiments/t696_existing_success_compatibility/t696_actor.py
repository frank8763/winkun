"""Add one cardinal readout on top of the completely frozen T694 function."""
from pathlib import Path
import copy, gzip, io, sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage/experiments/t694_full_history_compatibility'))
import t694_actor as Parent

torch, np = Parent.torch, Parent.np
L, Base = Parent.L, Parent.Base
read, sha, digest = Parent.read, Parent.sha, Parent.digest
ARCH = 'T696_frozen_T694_plus_fixed_existing_success_cardinal_readout'
same_float32, trajectory = Parent.same_float32, Parent.trajectory


class Actor(Parent.Actor):
    def __init__(self, p, old_count, count):
        super().__init__(p, old_count)
        self.register_buffer('new_centers', torch.zeros(count, 184, dtype=torch.float64))
        self.register_buffer('new_radii', torch.ones(count, dtype=torch.float64))
        self.new_weights = torch.nn.Parameter(torch.zeros(count, 2, dtype=torch.float64))
        self.requires_grad_(False)

    def location(self, h, n):
        original = super().location(h, n)
        features = torch.cat([h, n], -1).reshape(-1, 184)
        values, active = [], []
        for offset in range(0, len(features), 32):
            phi = Parent.basis(features[offset:offset + 32], self.new_centers, self.new_radii)
            values.append(phi @ self.new_weights)
            active.append((phi != 0).any(-1))
        correction = torch.cat(values).reshape(original.shape)
        mask = torch.cat(active).reshape(original.shape[:-1])[..., None]
        return torch.where(mask, original + correction, original)

    def phash(self):
        return digest(dict(architecture=ARCH, state=self.state_dict(), sigma=self.p['raw_sigma']))


def frozen(p, path):
    saved = Base.P.checkpoint(path)
    if saved.get('architecture') != ARCH:
        return Parent.frozen(p, path)
    state = saved['joint_state']['actor']
    q = Actor(p, len(state['compat_radii']), len(state['new_radii']))
    q.load_state_dict(state)
    assert q.phash() == saved['policy_hash']
    return q.eval()


def data(p, guard):
    old = Parent.ORIGINAL(p, p['original_T501']['path'])
    current = Parent.frozen(p, p['initial']['path'])
    before = digest(current.state_dict())
    protected, sources, old_manifest, new_manifest = [], [], [], []
    for row in p['protected403_rows']:
        guard()
        assert sha(row['path']) == row['sha256']
        a = L.T.R.E.N.S.arrays(row['path'])
        assert not a['ownership'].any() and not a['stochastic'].any()
        assert a['end_flags'][-1, 1] and not a['end_flags'][:-1].any()
        features, _ = trajectory(old, a, row['id'], a['actions'] if row['original384'] else None, guard)
        protected.append(features)
        old_manifest.append(dict(id=row['id'], source_sha256=row['sha256'], decisions=len(features), feature_hash=digest(features)))
    assert len(protected) == 403
    body_keys = [k for k in old.state_dict() if not k.startswith(('drive_adapter.', 'steer_adapter.'))]
    for row in p['novel_rows']:
        guard()
        assert sha(row['path']) == row['sha256']
        a = L.T.R.E.N.S.arrays(row['path'])
        assert not a['ownership'].any() and a['end_flags'][-1, 1] and not a['end_flags'][:-1].any()
        source_p = dict(p, raw_sigma=row['source_raw_sigma'])
        source = Parent.frozen(source_p, row['source_model'])
        assert all(torch.equal(source.state_dict()[k], old.state_dict()[k]) for k in body_keys)
        features, mu = trajectory(current, a, row['id'], None, guard)
        target = torch.tensor(a['actions'], dtype=torch.float64)
        sources.append(dict(id=row['id'], features=features, base_location=mu, target=target))
        new_manifest.append(dict(id=row['id'], source_sha256=row['sha256'], decisions=len(features),
                                 feature_hash=digest(features), target_hash=digest(target), stochastic=bool(a['stochastic'].any())))
    assert len(sources) == 7 and digest(current.state_dict()) == before
    return torch.cat(protected), sources, dict(old=old_manifest, new=new_manifest, all_owncarry_batch1=True)


def construction(p, protected, sources, guard):
    geometry, info = Parent.construction(p, protected, sources, guard)
    count = sum(len(s['features']) for s in sources)
    info.update(source_decisions=count, exact_duplicate_rows_merged=count - len(geometry['radii']))
    return geometry, info


def fit(p, protected, sources, guard, event):
    geometry, info = construction(p, protected, sources, guard)
    parent = Parent.frozen(p, p['initial']['path'])
    q = Actor(p, len(parent.compat_radii), len(geometry['radii']))
    missing, extra = q.load_state_dict(parent.state_dict(), strict=False)
    assert set(missing) == {'new_centers', 'new_radii', 'new_weights'} and not extra
    q.new_centers.copy_(geometry['centers'])
    q.new_radii.copy_(geometry['radii'])
    event('BEFORE', 0)
    q.new_weights.copy_(geometry['weights'])
    event('AFTER', 0)
    assert all(torch.equal(q.state_dict()[k], v) for k, v in parent.state_dict().items())
    return q, info


def verify_commands(p, q, guard):
    for row in p['protected403_rows'] + p['novel_rows']:
        guard()
        a = L.T.R.E.N.S.arrays(row['path'])
        trajectory(q, a, row['id'], a['actions'], guard)
    return dict(all403_retained_and7_source_batch1_commands_owncarry_exact=True,
                finite_supervision_only_no_stochastic_or_unseen_history_claim=True)


def serialized(q, p):
    joint = copy.deepcopy(Base.P.checkpoint(p['initial']['path'])['joint_state'])
    joint['actor'] = copy.deepcopy(q.state_dict())
    buffer = io.BytesIO()
    torch.save(dict(joint_state=joint, architecture=ARCH, policy_hash=q.phash(), joint_hash=digest(joint),
                    unchanged_original_optimizer_is_provenance_NOT_compatible_new_head_optimizer=True,
                    fit='ONE_ADDED_CARDINAL_READOUT_FROZEN_T694', single_actor=True, teacher_at_deployment=False), buffer)
    blob = gzip.compress(buffer.getvalue(), mtime=0)
    assert len(blob) <= p['model_bytes']
    return blob, joint


def export(q, p, path):
    blob, joint = serialized(q, p)
    with Path(path).open('xb') as out:
        out.write(blob)
    return dict(path=str(path), sha256=sha(path), bytes=len(blob), policy_hash=q.phash(), joint_hash=digest(joint))

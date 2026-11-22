"""Add one cardinal readout on top of the completely frozen T694 function."""
from pathlib import Path
import copy, gzip, io, sys, fcntl, time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage/experiments/t696_existing_success_compatibility'))
import t696_actor as Parent
import t698_resources as Resources

torch, np = Parent.torch, Parent.np
L, Base = Parent.L, Parent.Base
read, sha, digest = Parent.read, Parent.sha, Parent.digest
ARCH = 'T698_frozen_T696_plus_two_legacy_success_cardinal_readout'
same_float32, trajectory = Parent.same_float32, Parent.trajectory

def source_trajectory(q,a,key,expected_commands=None,guard=lambda:None):
    """Own candidate carry; source policy internals are never imported."""
    q.sample=False;q.rng=None;q.reset_noise();h=None;features=[];locations=[]
    with torch.no_grad():
        for index,x in enumerate(a['states']):
            if index%64==0:guard()
            xx=torch.tensor(x[None]);command,mu,h,_=q.step(xx,h)
            if expected_commands is not None:
                assert same_float32(command[0].numpy(),expected_commands[index]),(key,index,'executed_float32')
            features.append(torch.cat([h[0,0],q.normalized(xx)[0]],-1).clone());locations.append(mu[0].clone())
    q.reset_noise()
    return torch.stack(features),torch.stack(locations)


class Actor(Parent.Actor):
    def __init__(self, p, old_count, prior_count, count):
        super().__init__(p, old_count, prior_count)
        self.register_buffer('last_centers', torch.zeros(count, 184, dtype=torch.float64))
        self.register_buffer('last_radii', torch.ones(count, dtype=torch.float64))
        self.last_weights = torch.nn.Parameter(torch.zeros(count, 2, dtype=torch.float64))
        self.requires_grad_(False)

    def location(self, h, n):
        original = super().location(h, n)
        features = torch.cat([h, n], -1).reshape(-1, 184)
        values, active = [], []
        for offset in range(0, len(features), 32):
            phi = Parent.Parent.basis(features[offset:offset + 32], self.last_centers, self.last_radii)
            values.append(phi @ self.last_weights)
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
    q = Actor(p, len(state['compat_radii']), len(state['new_radii']), len(state['last_radii']))
    q.load_state_dict(state)
    assert q.phash() == saved['policy_hash']
    return q.eval()


def data(p, guard):
    old = Parent.Parent.ORIGINAL(p, p['original_T501']['path'])
    current = Parent.frozen(p, p['initial']['path'])
    before = digest(current.state_dict())
    protected, sources, old_manifest, new_manifest = [], [], [], []
    for row in p['protected410_rows']:
        guard()
        assert sha(row['path']) == row['sha256']
        a = L.T.R.E.N.S.arrays(row['path'])
        assert not a['ownership'].any() and not a['stochastic'].any()
        assert a['end_flags'][-1, 1] and not a['end_flags'][:-1].any()
        features, _ = trajectory(old, a, row['id'], a['actions'] if row['original384'] else None, guard)
        protected.append(features)
        old_manifest.append(dict(id=row['id'], source_sha256=row['sha256'], decisions=len(features), feature_hash=digest(features)))
    assert len(protected) == 410
    for row in p['novel_rows']:
        guard()
        assert sha(row['path']) == row['sha256']
        a = L.T.R.E.N.S.arrays(row['path'])
        assert not a['ownership'].any() and a['end_flags'][-1, 1] and not a['end_flags'][:-1].any()
        # Source three-carry model was verified in the immutable preparation.
        assert read(row['source_audit'])['state'] == 'PASS'
        features, mu = source_trajectory(current, a, row['id'], None, guard)
        target = torch.tensor(a['actions'], dtype=torch.float64)
        sources.append(dict(id=row['id'], features=features, base_location=mu, target=target))
        new_manifest.append(dict(id=row['id'], source_sha256=row['sha256'], decisions=len(features),
                                 feature_hash=digest(features), target_hash=digest(target), stochastic=bool(a['stochastic'].any())))
    assert len(sources) == 2 and digest(current.state_dict()) == before
    return torch.cat(protected), sources, dict(old=old_manifest, new=new_manifest, all_owncarry_batch1=True)


def construction(p, protected, sources, guard):
    geometry, info = gpu_construction(p, protected, sources, guard)
    count = sum(len(s['features']) for s in sources)
    info.update(source_decisions=count, exact_duplicate_rows_merged=count - len(geometry['radii']))
    return geometry, info


def gpu_construction(p, protected, sources, guard):
    """GPU computes finite distances; the deployed CPU path remains unchanged."""
    Resources.gpu()
    __import__('os').environ['CUDA_VISIBLE_DEVICES']='0'
    with (ROOT/'stage/runs/.stage_gpu.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert torch.cuda.is_available()
        caller_guard=guard;last_gpu_probe=[0.]
        def geometry_guard():
            caller_guard()
            if time.time()-last_gpu_probe[0]>5:Resources.gpu();last_gpu_probe[0]=time.time()
        guard=geometry_guard
        unique={};centers=[];base=[];targets=[]
        for source in sources:
            for f,mu,target in zip(source['features'],source['base_location'],source['target']):
                key=tuple(f.tolist())
                if key in unique:
                    i=unique[key];assert torch.equal(base[i],mu) and same_float32(target.numpy(),targets[i].numpy()),'Conflicting identical labels'
                else:
                    unique[key]=len(centers);centers.append(f);base.append(mu);targets.append(target)
        cpu=torch.stack(centers);baseline=torch.stack(base);target=torch.stack(targets)
        assert 1<len(cpu)<=p['maximum_centers']
        c=cpu.to('cuda:0');old=protected.to('cuda:0');near=c.new_full((len(c),),float('inf'))
        for off in range(0,len(old),64):
            guard()
            distances=((c[:,None]-old[off:off+64])**2).sum(-1)
            near=torch.minimum(near,distances.min(-1).values)
        other=c.new_full((len(c),),float('inf'))
        for off in range(0,len(c),32):
            guard();distances=((c[off:off+32,None]-c)**2).sum(-1)
            distances[torch.arange(len(distances),device='cuda'),torch.arange(off,off+len(distances),device='cuda')]=float('inf')
            other[off:off+len(distances)]=distances.min(-1).values
        distance,neighbor=near.sqrt(),other.sqrt()
        assert torch.isfinite(distance).all() and float(distance.min())>p['minimum_feature_separation']
        assert torch.isfinite(neighbor).all() and float(neighbor.min())>p['minimum_feature_separation']
        radii=p['support_fraction']*torch.minimum(distance,neighbor)
        for off in range(0,len(old),64):
            guard();assert not Parent.Parent.basis(old[off:off+64],c,radii).any()
        for off in range(0,len(c),32):
            guard();actual=Parent.Parent.basis(c[off:off+32],c,radii);expected=torch.zeros_like(actual)
            expected[torch.arange(len(actual),device='cuda'),torch.arange(off,off+len(actual),device='cuda')]=1
            assert torch.equal(actual,expected)
        radius_cpu=radii.cpu();weights=target-baseline
        assert same_float32((baseline+weights).clamp(-1,1).numpy(),target.numpy())
        # Explicit independent CPU kernel check of all old/new features. The
        # radius is fixed above; no radius/precision/source rescue is tried.
        for off in range(0,len(protected),64):
            guard();assert not Parent.Parent.basis(protected[off:off+64],cpu,radius_cpu).any()
        for off in range(0,len(cpu),32):
            guard();actual=Parent.Parent.basis(cpu[off:off+32],cpu,radius_cpu);expected=torch.zeros_like(actual)
            expected[torch.arange(len(actual)),torch.arange(off,off+len(actual))]=1;assert torch.equal(actual,expected)
        info=dict(centers=len(cpu),source_decisions=sum(len(s['features']) for s in sources),
            protected_decisions=len(protected),minimum_protected_distance=float(distance.min()),
            minimum_other_center_distance=float(neighbor.min()),minimum_radius=float(radii.min()),
            maximum_radius=float(radii.max()),support_fraction=p['support_fraction'],
            protected_max_basis=0.,center_matrix_exact_identity=True,condition=1.,
            center_hash=digest(cpu),radius_hash=digest(radius_cpu),coefficient_formula_hash=digest(weights),
            geometry_device='cuda:0',actual_cpu_basis_all_old_zero_and_K_identity=True,
            no_new_Actor_applied_in_construction=True,supervised_interpolation_NOT_RL_or_generalization=True)
        del c,old,near,other,distances,distance,neighbor,radii
        torch.cuda.empty_cache()
        return dict(centers=cpu,radii=radius_cpu,weights=weights),info


def fit(p, protected, sources, guard, event):
    geometry, info = construction(p, protected, sources, guard)
    parent = Parent.frozen(p, p['initial']['path'])
    q = Actor(p, len(parent.compat_radii), len(parent.new_radii), len(geometry['radii']))
    missing, extra = q.load_state_dict(parent.state_dict(), strict=False)
    assert set(missing) == {'last_centers', 'last_radii', 'last_weights'} and not extra
    q.last_centers.copy_(geometry['centers'])
    q.last_radii.copy_(geometry['radii'])
    event('BEFORE', 0)
    q.last_weights.copy_(geometry['weights'])
    event('AFTER', 0)
    assert all(torch.equal(q.state_dict()[k], v) for k, v in parent.state_dict().items())
    return q, info


def verify_commands(p, q, guard):
    for row in p['protected410_rows']:
        guard(); a = L.T.R.E.N.S.arrays(row['path']); trajectory(q, a, row['id'], a['actions'], guard)
    for row in p['novel_rows']:
        guard(); a = L.T.R.E.N.S.arrays(row['path']); source_trajectory(q, a, row['id'], a['actions'], guard)
    return dict(all410_retained_and2_source_batch1_commands_owncarry_exact=True,
                finite_supervision_only_no_stochastic_or_unseen_history_claim=True)


def serialized(q, p):
    joint = copy.deepcopy(Base.P.checkpoint(p['initial']['path'])['joint_state'])
    joint['actor'] = copy.deepcopy(q.state_dict())
    buffer = io.BytesIO()
    torch.save(dict(joint_state=joint, architecture=ARCH, policy_hash=q.phash(), joint_hash=digest(joint),
                    unchanged_original_optimizer_is_provenance_NOT_compatible_new_head_optimizer=True,
                    fit='ONE_ADDED_CARDINAL_READOUT_FROZEN_T696', single_actor=True, teacher_at_deployment=False), buffer)
    blob = gzip.compress(buffer.getvalue(), mtime=0)
    assert len(blob) <= p['model_bytes']
    return blob, joint


def export(q, p, path):
    blob, joint = serialized(q, p)
    with Path(path).open('xb') as out:
        out.write(blob)
    return dict(path=str(path), sha256=sha(path), bytes=len(blob), policy_hash=q.phash(), joint_hash=digest(joint))

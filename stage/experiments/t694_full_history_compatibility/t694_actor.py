"""Finite T546-style compatibility replication, all384 protected histories.

One static120-input/GRU64 neural forward, expanded radial readout. This is
supervised interpolation, not fixed11906 learning or a new RL algorithm.
"""
from pathlib import Path
import copy,gzip,io,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t690_score_weighted_baseline'))
import t690_learning as L
Base=L.M.L
torch,np=L.torch,L.np
read,sha,digest=L.read,L.sha,L.digest
ARCH='T694_single120_frozen_T501_GRU64_compact_radial_readout'
ORIGINAL=Base.frozen


def basis(features,centers,radii):
    return torch.relu(1-((features[...,None,:]-centers)**2).sum(-1)/radii.square()).square()


class Actor(Base.Actor):
    def __init__(self,p,count):
        super().__init__(p)
        self.register_buffer('compat_centers',torch.zeros(count,184,dtype=torch.float64))
        self.register_buffer('compat_radii',torch.ones(count,dtype=torch.float64))
        self.compat_weights=torch.nn.Parameter(torch.zeros(count,2,dtype=torch.float64))
        self.requires_grad_(False)
    def location(self,h,n):
        original=super().location(h,n);shape=original.shape
        f=torch.cat([h,n],-1).reshape(-1,184);values=[];active=[]
        for offset in range(0,len(f),32):
            phi=basis(f[offset:offset+32],self.compat_centers,self.compat_radii)
            values.append(phi@self.compat_weights);active.append((phi!=0).any(-1))
        correction=torch.cat(values).reshape(shape)
        mask=torch.cat(active).reshape(shape[:-1])[...,None]
        # Exactly preserve old arithmetic, including signed zeros, whenever
        # every basis is zero. This has no task-ID or world-coordinate input.
        return torch.where(mask,original+correction,original)
    def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))


def frozen(p,path):
    saved=Base.P.checkpoint(path)
    if saved.get('architecture')!=ARCH:return ORIGINAL(p,path)
    state=saved['joint_state']['actor'];q=Actor(p,len(state['compat_radii']))
    q.load_state_dict(state);assert q.phash()==saved['policy_hash']
    return q.eval()


def same_float32(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    return a.shape==b.shape and np.array_equal(a.view(np.uint32),b.view(np.uint32))


def trajectory(q,a,key,expected_commands=None,guard=lambda:None):
    q.sample=False;q.rng=None;q.reset_noise();h=None;features=[];locations=[]
    with torch.no_grad():
        for i,x in enumerate(a['states']):
            if i%64==0:guard()
            before=np.zeros(64) if h is None else h[0,0].numpy().copy()
            assert np.array_equal(before,a['hidden_before'][i]),(key,i,'owncarry')
            xx=torch.tensor(x[None]);command,mu,h,_=q.step(xx,h)
            if expected_commands is not None:
                assert same_float32(command[0].numpy(),expected_commands[i]),(key,i,'executed_float32')
            features.append(torch.cat([h[0,0],q.normalized(xx)[0]],-1).clone())
            locations.append(mu[0].clone())
    assert np.array_equal(h[0,0].numpy(),a['last_hidden_after'])
    q.reset_noise()
    return torch.stack(features),torch.stack(locations)


def data(p,guard):
    q=ORIGINAL(p,p['initial']['path']);initial=digest(q.state_dict())
    protected=[];sources=[];old_manifest=[];new_manifest=[]
    for row in p['protected_rows']:
        guard();assert sha(row['source_path'])==row['sha256']
        a=L.T.R.E.N.S.arrays(row['source_path'])
        assert not a['ownership'].any() and not a['stochastic'].any()
        assert a['end_flags'][-1,1] and not a['end_flags'][:-1].any()
        f,mu=trajectory(q,a,row['task_id'],a['actions'],guard)
        protected.append(f);old_manifest.append(dict(id=row['task_id'],source_sha256=row['sha256'],feature_hash=digest(f),decisions=len(f)))
    assert len(protected)==384
    for row in p['novel_rows']:
        guard();assert sha(row['path'])==row['sha256'] and row['safe']
        a=L.T.R.E.N.S.arrays(row['path']);assert not a['stochastic'].any() and not a['ownership'].any()
        assert a['end_flags'][-1,1] and not a['end_flags'][:-1].any()
        assert bool((np.abs(a['poses'][:,:2])<=7).all())
        f,mu=trajectory(q,a,row['id'],None,guard)
        source=ORIGINAL(p,p['novel_source_model']['path'])
        assert L.M.L.fixed_hash(source)==L.M.L.fixed_hash(q)
        _,source_mu=trajectory(source,a,row['id'],a['actions'],guard)
        assert same_float32(source_mu.clamp(-1,1).float().numpy(),a['actions'])
        target=torch.tensor(a['actions'],dtype=torch.float64)
        sources.append(dict(id=row['id'],features=f,base_location=mu,target=target))
        new_manifest.append(dict(id=row['id'],source_sha256=row['sha256'],feature_hash=digest(f),target_hash=digest(target),decisions=len(f)))
    assert len(sources)==6 and sum(len(v['features']) for v in sources)==1405
    assert digest(q.state_dict())==initial
    return torch.cat(protected),sources,dict(old=old_manifest,new=new_manifest,all_owncarry_batch1=True)


def construction(p,protected,sources,guard):
    # Exact duplicates represent one function query only if both target command
    # AND original value agree. No approximate merge/center selection is used.
    unique={};center=[];base=[];targets=[]
    for source in sources:
        for f,mu,target in zip(source['features'],source['base_location'],source['target']):
            key=tuple(f.tolist())
            if key in unique:
                i=unique[key]
                assert torch.equal(base[i],mu) and same_float32(target.numpy(),targets[i].numpy()),'Conflicting identical input labels'
            else:
                unique[key]=len(center);center.append(f);base.append(mu);targets.append(target)
    centers=torch.stack(center);baseline=torch.stack(base);target=torch.stack(targets)
    assert 1<len(centers)<=p['maximum_centers']
    nearest=centers.new_full((len(centers),),float('inf'))
    for off in range(0,len(protected),32):
        guard();d=((centers[:,None]-protected[off:off+32])**2).sum(-1)
        nearest=torch.minimum(nearest,d.min(-1).values)
    other=centers.new_full((len(centers),),float('inf'))
    for off in range(0,len(centers),32):
        guard();d=((centers[off:off+32,None]-centers)**2).sum(-1)
        d[torch.arange(len(d)),torch.arange(off,off+len(d))]=float('inf')
        other[off:off+len(d)]=d.min(-1).values
    distance=nearest.sqrt();neighbor=other.sqrt()
    assert torch.isfinite(distance).all() and float(distance.min())>p['minimum_feature_separation']
    assert torch.isfinite(neighbor).all() and float(neighbor.min())>p['minimum_feature_separation']
    radii=p['support_fraction']*torch.minimum(distance,neighbor)
    for off in range(0,len(protected),32):
        guard();assert float(basis(protected[off:off+32],centers,radii).max())==0.
    for off in range(0,len(centers),32):
        guard();actual=basis(centers[off:off+32],centers,radii)
        expected=actual.new_zeros(actual.shape);expected[torch.arange(len(actual)),torch.arange(off,off+len(actual))]=1
        assert torch.equal(actual,expected),'Cardinal basis is not exact identity'
    # K=I exactly; there is no ridge, tunable linear solver or conditioning fix.
    weights=target-baseline
    assert same_float32((baseline+weights).clamp(-1,1).numpy(),target.numpy())
    info=dict(centers=len(centers),source_decisions=1405,exact_duplicate_rows_merged=1405-len(centers),
        protected_decisions=len(protected),minimum_protected_distance=float(distance.min()),minimum_other_center_distance=float(neighbor.min()),
        minimum_radius=float(radii.min()),maximum_radius=float(radii.max()),support_fraction=p['support_fraction'],
        protected_max_basis=0.,center_matrix_exact_identity=True,condition=1.,closed_form='weights=target-base_location because K=I',
        center_hash=digest(centers),radius_hash=digest(radii),coefficient_formula_hash=digest(weights),
        no_new_Actor_applied_in_construction=True,supervised_interpolation_NOT_RL_or_generalization=True)
    return dict(centers=centers,radii=radii,weights=weights),info


def fit(p,protected,sources,guard,event):
    geometry,info=construction(p,protected,sources,guard)
    base=ORIGINAL(p,p['initial']['path']);q=Actor(p,len(geometry['radii']))
    missing,extra=q.load_state_dict(base.state_dict(),strict=False)
    assert set(missing)=={'compat_centers','compat_radii','compat_weights'} and not extra
    q.compat_centers.copy_(geometry['centers']);q.compat_radii.copy_(geometry['radii'])
    event('BEFORE',0)
    q.compat_weights.copy_(geometry['weights'])
    event('AFTER',0)
    for key,value in base.state_dict().items():assert torch.equal(q.state_dict()[key],value)
    return q,info


def verify_commands(p,q,guard):
    for row in p['protected_rows']:
        guard();a=L.T.R.E.N.S.arrays(row['source_path']);trajectory(q,a,row['task_id'],a['actions'],guard)
    for row in p['novel_rows']:
        guard();a=L.T.R.E.N.S.arrays(row['path']);trajectory(q,a,row['id'],a['actions'],guard)
    return dict(all384_old_and6_new_batch1_commands_and_owncarry_exact=True,
                no_stochastic_robustness_or_unseen_history_claim=True)


def serialized(q,p):
    original=Base.P.checkpoint(p['initial']['path'])['joint_state']
    joint=copy.deepcopy(original);joint['actor']=copy.deepcopy(q.state_dict())
    buffer=io.BytesIO();torch.save(dict(joint_state=joint,architecture=ARCH,policy_hash=q.phash(),joint_hash=digest(joint),
        unchanged_original_optimizer_is_provenance_NOT_compatible_new_head_optimizer=True,
        fit='ONE_CARDINAL_COMPACT_SUPPORT_SUPERVISED_INTERPOLATION',single_actor=True,teacher_at_deployment=False),buffer)
    blob=gzip.compress(buffer.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
    return blob,joint


def export(q,p,path):
    blob,joint=serialized(q,p)
    with Path(path).open('xb') as f:f.write(blob)
    return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(joint))

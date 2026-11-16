"""Concrete finite proposal and zero-update contracts before new budget consent."""
from pathlib import Path
import sys,copy,time,datetime,ast,io,json
import t598_runtime as R
L=R.L;ROOT,HERE,OUT,DOC=R.ROOT,R.HERE,R.OUT,R.DOC
read,save,sha,digest,torch,np=L.read,L.save,L.sha,L.digest,L.torch,L.np


def prepare():
    assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists()
    DOC.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(read(ROOT/'stage/runs/t596_recurrent_joint_online/protocol.json'))
    end=datetime.datetime.fromisoformat('2026-11-16T20:00:00+08:00').timestamp()
    p.update(id='T598_INDEPENDENT_RECURRENT_RECOVERY_SUPERVISED_PILOT',out=str(OUT),
        initial=p['base_model'],fit_steps=128,fit_lr=3e-4,methods=['one_engineering_pilot'],
        inputs={},protected_files={},authorization_approved=False,
        main_seconds=1500,learning_seconds=900,audit_seconds=600,transport_seconds=180,source_seconds=120,
        absolute_learning_stop=end-1500,absolute_native_stop=end-900,absolute_all_stop=end,
        max_commands=63,max_bytes=12*2**20,child_bytes=12*2**20,full_bytes=2*2**20,model_bytes=2**20,
        key_replay_ids=['t502_field_sector0_offset2','t502_field_sector1_offset2','t502_field_sector4_offset2','wide117_8m_dev6_RIGHT_13'],
        new_Actor_Adam_cap=128,shadow_Adam_cap=128,audit_discarded_Adam_cap=128,preflight_discarded_Adam_cap=4,
        new_RL=0,new_Critic=0,
        scope='Complete T550 frozen; independent32GRU difference residual,128fullbatch supervised steps on39keep+3own-success recovery suffixes, equal family mass and halfmean/halftop10%.3mandatory full autonomous handoffs+compactexact; at least1safe then49compact+4exact. Research keep only all39initialIDs retained and >=1new safe takeover relative historical0/3. No globalpromotion/newRL/407/fresh16/reload78. Joint engineering pilot, no causal module novelty claim.')
    p['max_physics']=2*sum(j['case']['step_cap'] for j in p['handoff_jobs'])+sum(j['case']['step_cap'] for j in p['evaluation_jobs'])+sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids'])
    upper=3*p['full_bytes']+56*p['compact_bytes']+p['model_bytes']+2*2**20
    glob=R.R.used(ROOT/'stage/runs')+p['max_bytes']+131072
    assert upper<=p['max_bytes'] and glob<2**30
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=L.initial(p);base=L.Parent.frozen(p,p['base_model']['path']);data=L.dataset(p);h0=digest(q.state_dict())
    with torch.no_grad():
        for g in data:assert torch.equal(q.sequence(g['x']),base.sequence(g['x']))
        x=data[0]['x'][:48];h=None;bh=None;q.reset_noise();base.reset_noise();q.sample=base.sample=False
        for z in x:
            u,loc,h,_=q.step(z[None],h);bu,bl,bh,_=base.step(z[None],bh)
            assert torch.equal(u,bu) and torch.equal(loc,bl) and torch.equal(h,bh)
        assert torch.equal(q.sequence(x).clamp(-1,1).float(),torch.stack([torch.tensor(r['raw_actions']).clamp(-1,1).float() for r in q.tape]))
        assert all(r['extra_hidden_before'].shape==(32,) and r['anchor_hidden_before'].shape==(32,) for r in q.tape)
    q.branch.requires_grad_(True);packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);start=time.monotonic()
    loss,parts=L.objective(q,data,packed);loss.backward();duration=time.monotonic()-start
    grad={name:dict(parameters=sum(v.numel() for v in mod.parameters()),nonzero=sum(int(torch.count_nonzero(v.grad)) for v in mod.parameters())) for name,mod in [('encoder',q.branch.encoder),('core',q.branch.core),('head',q.branch.head)]}
    assert all(v['nonzero']>0 for v in grad.values()) and digest(q.state_dict())==h0
    assert all(v.grad is None for k,v in q.named_parameters() if not k.startswith('branch.'))
    q.zero_grad(set_to_none=True)
    # Worst-case raw NPZ size, including the two extra memories and final states.
    a=L.G.E.N.S.arrays(p['recovery_rows'][0]['path']);n=len(a['states']);N=max(j['case']['step_cap']//100+2 for j in p['handoff_jobs'])
    big={k:np.zeros((N+(len(v)-n),*v.shape[1:]),dtype=v.dtype) if v.ndim and len(v) in [n,n+1] else v for k,v in a.items()}
    big.update(extra_hidden_before=np.zeros((N,32)),anchor_hidden_before=np.zeros((N,32)),last_extra_hidden_after=np.zeros(32),last_anchor_hidden_after=np.zeros(32))
    b=io.BytesIO();np.savez(b,**big);assert len(b.getvalue())<=p['full_bytes']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    total_admission=1500+600+180+120+30
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',all42_full_history_initial_locations_exact=True,
        actual48step_initial_actions_and_legacy_hidden_exact=True,extra_memories_recorded=True,
        old_and_anchor_gradients_absent=True,all_branch_gradient_groups_nonzero=grad,
        gradient_only_seconds=duration,gradient_only_evaluations=1,new_optimizer=0,new_native=0,
        trainable_parameters=sum(v.numel() for v in L.variables(q)),extra_frozen_reference_parameters=sum(v.numel() for v in q.anchor.parameters()),
        uncompressed_worst_full_npz_bytes=len(b.getvalue()),own_upper=upper,global_worst=glob,
        total_admission_seconds=total_admission,required_remaining_seconds=total_admission)
    save(DOC/'PREFLIGHT.json',pre)
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
    for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json']+[Path(x['path']) for x in p['keep_rows']+p['recovery_rows']]:p['inputs'][str(f)]=sha(f)
    for j in p['evaluation_jobs']:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
    for j in p['handoff_jobs']:p['inputs'][j['source_path']]=sha(j['source_path'])
    m=p['base_model']
    while m:p['inputs'][m['path']]=m['sha256'];p['protected_files'][m['path']]=m['sha256'];m=m.get('parent')
    save(DOC/'PROPOSAL.json',p);save(DOC/'BUDGET.json',dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',max_physics=p['max_physics'],max_commands=63,own_bytes=p['max_bytes'],own_upper=upper,global_worst=glob,live_Adam=128,shadow_Adam=128,audit_historical_Adam=128,pref_discarded_Adam=4,main_seconds=1500,audit_seconds=600,transport_seconds=180,source_seconds=120,all_stop='20:00',total_admission_seconds=total_admission,budget_approval='PENDING'))
    print('T598_PREPARED_ZERO_UPDATE',json.dumps(pre),flush=True)


def optimizer_preflight():
    assert not OUT.exists() and not (DOC/'PREFLIGHT_OPTIMIZER.json').exists()
    p=read(DOC/'PROPOSAL.json');assert read(DOC/'AUTHORIZATION.json')['approved']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);data=L.dataset(p);a,b=L.make(p),L.make(p);before=L.state(a)
    aa=L.fit(a,data,p,lambda:None,lambda *_:None,steps=2);bb=L.fit(b,data,p,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(L.state(a))==digest(L.state(b));mature=L.state(a)
    L.load(a,before);assert digest(L.state(a))==digest(before);L.load(a,mature);assert digest(L.state(a))==digest(mature)
    file=DOC/'discarded_mature.pt.gz';m=L.export(a,p,file);q=L.frozen(p,file);assert q.phash()==a[0].phash()
    # Updated branch, not just zero initialization: current sequence versus real step.
    x=data[-1]['x'][:96];q.sample=False;q.reset_noise();h=None;loc=[]
    with torch.no_grad():
        z=q.sequence(x)
        for xx in x:_,ll,h,_=q.step(xx[None],h);loc.append(ll[0])
    err=float((z-torch.stack(loc)).abs().max());assert err<1e-12
    file.unlink();save(DOC/'PREFLIGHT_OPTIMIZER.json',dict(state='PASS_AUTHORIZED_ZERO_NATIVE',discarded_Adam=4,whole_ActorAdam_shadow_rollback_readback_exact=True,updated96step_sequence_location_max_error=err,checkpoint_bytes=m['bytes']))
    p['inputs'][str(DOC/'PREFLIGHT_OPTIMIZER.json')]=sha(DOC/'PREFLIGHT_OPTIMIZER.json');save(DOC/'PROPOSAL.json',p)


def register():
    assert not OUT.exists();p=read(DOC/'PROPOSAL.json');a=read(DOC/'AUTHORIZATION.json')
    assert a['approved'] and a['max_physics']==p['max_physics'] and a['max_commands']==p['max_commands'] and a['own_bytes']==p['max_bytes']
    assert time.time()+2430<p['absolute_all_stop'] and time.time()+900<p['absolute_learning_stop'] and time.time()+1500<p['absolute_native_stop']
    assert R.R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f
    p['authorization_approved']=True;p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');OUT.mkdir()
    blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    print('T598_REGISTERED_NOT_LAUNCHED',flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()

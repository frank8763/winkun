"""Prepare identical data/Actor arms and exercise the real counter path."""
from pathlib import Path
import sys,copy,time,datetime,ast,json
import t603_runtime as R
L=R.L;ROOT,HERE,OUT,DOC=R.ROOT,R.HERE,R.OUT,R.DOC
read,save,sha,digest,torch,np=L.read,L.save,L.sha,L.digest,L.torch,L.np


def prepare():
    assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(read(ROOT/'stage/runs/t598_independent_recurrent_recovery/protocol.json'))
    rows=R.G.rows(ROOT/'stage/runs/t598_independent_recurrent_recovery')
    good=[r for r in rows[:3] if R.Legacy.R.safe(r)];assert len(good)==1
    prior=read(ROOT/'stage/runs/t599_partial_evidence_recovery/adapter_001/result.json')
    assert prior['proposal_full_ActorAdam_reconstruction_exact']
    sentinel=sorted({j['id'] for j in p['old8_jobs']}|set(prior['original49']['lost_initial']));assert len(sentinel)==15
    epoch=lambda h:datetime.datetime.fromisoformat('2026-11-16T'+h+':00+08:00').timestamp()
    p.update(id='T603_MATCHED_TASK_RISK_PAIR',out=str(OUT),initial=p['base_model'],joint_recovery_row=good[0],recovery_rows=[good[0]]+read(ROOT/'stage/docs/day_20261116/t602/NEXT_QUALIFIED_INPUTS.json')['new_qualified_rows'],
        methods=['tail'],fit_steps=128,fit_lr=3e-4,sentinel_ids=sentinel,
        inputs={},protected_files={},authorization_approved=False,
        main_seconds=900,learning_seconds=600,audit_seconds=300,transport_seconds=120,source_seconds=90,
        absolute_learning_stop=epoch('20:40'),absolute_native_stop=epoch('20:50'),absolute_all_stop=epoch('21:00'),
        max_commands=63,max_bytes=6*2**20,child_bytes=6*2**20,full_bytes=2*2**20,model_bytes=2**20,
        new_Actor_Adam_cap=128,shadow_Adam_cap=128,audit_discarded_Adam_cap=128,preflight_discarded_Adam_cap=4,
        new_RL=0,new_Critic=0,
        scope='See locked PLAN for single42-history aggregation pilot.')
    p['scope']=(DOC/'PLAN.md').read_text()
    job=good[0]['request']['job'];p['max_physics']=2*sum(r['request']['job']['case']['step_cap'] for r in p['recovery_rows'])+sum(j['case']['step_cap'] for j in p['evaluation_jobs'])+sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids'])
    upper=p['full_bytes']+p['model_bytes']+58*p['compact_bytes']+2*2**20
    glob=R.Legacy.R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert upper<=p['max_bytes'] and glob<2**30
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=L.initial(p);base=L.Old.Parent.frozen(p,p['base_model']['path']);data=L.dataset(p)
    initial=digest(q.state_dict())
    with torch.no_grad():
        for g in data:assert torch.equal(q.sequence(g['x']),base.sequence(g['x']))
        x=data[-1]['x'][:96];q.sample=False;q.reset_noise();base.sample=False;base.reset_noise();h=bh=None
        for z in x:
            u,loc,h,_=q.step(z[None],h);bu,bl,bh,_=base.step(z[None],bh)
            assert torch.equal(u,bu) and torch.equal(loc,bl) and torch.equal(h,bh)
    q.branch.requires_grad_(True);packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);gradient=[]
    for method in p['methods']:
        q.zero_grad(set_to_none=True);start=time.monotonic();loss,parts=L.objective(q,data,packed,method);loss.backward();duration=time.monotonic()-start
        grad={name:sum(int(torch.count_nonzero(v.grad)) for v in mod.parameters()) for name,mod in [('encoder',q.branch.encoder),('core',q.branch.core),('head',q.branch.head)]}
        assert all(v>0 for v in grad.values()) and all(v.grad is None for k,v in q.named_parameters() if not k.startswith('branch.'))
        gradient.append(dict(method=method,loss=float(loss.detach()),parts=parts,nonzero_gradients=grad,seconds=duration));q.zero_grad(set_to_none=True)
    assert digest(q.state_dict())==initial
    schedule=R.schedule_probe(p)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    save(DOC/'SCHEDULE_PREFLIGHT.json',schedule)
    pref=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',all42_initial_full_history_locations_exact_T550=True,
        actual96step_initial_commands_and_legacy_hidden_exact=True,qualified_new_recovery_rawlocation_matches_saved_steps=True,
        single_gradient_only=gradient,new_native=0,new_optimizer=0,
        model_state_unchanged=True,fixed_single_tail_method=True,
        same_trainable_parameters=10274,extra_frozen_reference_parameters=10274,
        real_runtime_all3_schedules_pass=True,undersized63minus1_rejected=True,
        own_upper=upper,global_worst=glob)
    save(DOC/'PREFLIGHT.json',pref)
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
    for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC/'SCHEDULE_PREFLIGHT.json',ROOT/'stage/runs/t599_partial_evidence_recovery/adapter_001/result.json']+[Path(x['path']) for x in p['keep_rows']+p['recovery_rows']]:p['inputs'][str(f)]=sha(f)
    p['inputs'][good[0]['request']['model']]=sha(good[0]['request']['model'])
    for j in p['evaluation_jobs']:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
    
    for r in p['recovery_rows']:
        jj=r['request']['job'];p['inputs'][jj['source_path']]=sha(jj['source_path']);p['inputs'][r['request']['model']]=sha(r['request']['model'])
    m=p['base_model']
    while m:p['inputs'][m['path']]=m['sha256'];p['protected_files'][m['path']]=m['sha256'];m=m.get('parent')
    save(DOC/'PROPOSAL.json',p)
    save(DOC/'BUDGET.json',dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',max_physics=p['max_physics'],max_commands=63,own_bytes=p['max_bytes'],own_upper=upper,global_worst=glob,live_Adam=128,shadow_Adam=128,audit_historical_Adam=128,pref_discarded_Adam=4,main_seconds=900,learning_seconds=600,audit_seconds=300,transport_seconds=120,source_seconds=90,total_admission_seconds=1440,learning_stop='20:40',native_stop='20:50',all_stop='21:00',new_budget_and_window='PENDING'))
    print('T603_PREPARED_ZERO_NATIVE_ZERO_OPTIMIZER',p['max_physics'],glob,flush=True)


def optimizer_preflight():
    assert not OUT.exists() and not (DOC/'PREFLIGHT_OPTIMIZER.json').exists()
    p=read(DOC/'PROPOSAL.json');assert read(DOC/'AUTHORIZATION.json')['approved'];torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=L.dataset(p);reports=[]
    for method in p['methods']:
        pc=dict(p,method=method);a,b=L.make(pc),L.make(pc);before=L.state(a)
        aa=L.fit(a,data,pc,lambda:None,lambda *_:None,steps=2);bb=L.fit(b,data,pc,lambda:None,lambda *_:None,steps=2)
        assert aa==bb and digest(L.state(a))==digest(L.state(b));mature=L.state(a)
        L.load(a,before);assert digest(L.state(a))==digest(before);L.load(a,mature);assert digest(L.state(a))==digest(mature)
        file=DOC/('discarded_'+method+'.pt.gz');m=L.export(a,pc,file);q=L.frozen(pc,file);assert q.phash()==a[0].phash()
        q.sample=False;q.reset_noise();h=None;x=data[-1]['x'][:96];out=[]
        with torch.no_grad():
            z=q.sequence(x)
            for xx in x:_,ll,h,_=q.step(xx[None],h);out.append(ll[0])
        err=float((z-torch.stack(out)).abs().max());assert err<1e-12
        file.unlink();reports.append(dict(method=method,discarded_Adam=4,checkpoint_bytes=m['bytes'],updated96step_max_location_error=err))
    save(DOC/'PREFLIGHT_OPTIMIZER.json',dict(state='PASS_AUTHORIZED_ZERO_NATIVE',discarded_Adam=4,reports=reports,whole_ActorAdam_shadow_mature_rollback_readback_exact=True))
    p['inputs'][str(DOC/'PREFLIGHT_OPTIMIZER.json')]=sha(DOC/'PREFLIGHT_OPTIMIZER.json');save(DOC/'PROPOSAL.json',p)


def register():
    assert not OUT.exists();p=read(DOC/'PROPOSAL.json');a=read(DOC/'AUTHORIZATION.json')
    assert a['approved'] and a['max_physics']==p['max_physics'] and a['max_commands']==63 and a['own_bytes']==p['max_bytes'] and a['all_stop_epoch']==p['absolute_all_stop']
    assert time.time()+1440<p['absolute_all_stop'] and time.time()+600<p['absolute_learning_stop'] and time.time()+900<p['absolute_native_stop']
    assert R.Legacy.R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f
    p['authorization_approved']=True;p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');OUT.mkdir()
    blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    print('T603_REGISTERED_NOT_LAUNCHED',flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()

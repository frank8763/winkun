"""Prepare offline; no registration or native launch without a valid new window."""
from pathlib import Path
import ast,copy,datetime,importlib.util,json,sys,time
import t582_runtime as R
ROOT,HERE,OUT,DOC,L=R.ROOT,R.HERE,R.OUT,R.DOC,R.L
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np


def prepare():
    assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True)
    old=ROOT/'stage/runs/t581_preserved_t550_rl';assert read(old/'audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/day_20261116/t581/COMPLETION_VERIFICATION.json')['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    sealed=read(old/'SEALED.json');assert all(not Path('/proc',str(i)).exists() for i in sealed['scientific_pids'])
    p=copy.deepcopy(read(old/'protocol.json'));res=read(old/'summary.json')['result']
    ids=res['check']['lost_initial'];assert len(ids)==5
    training=[p['target_job']]+[next(j for j in p['evaluation_jobs'] if j['id']==i) for i in ids]
    keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']]
    cap=8*sum(j['case']['step_cap'] for j in training)+2*(sum(j['case']['step_cap'] for j in p['evaluation_jobs'])+sum(j['case']['step_cap'] for j in keys))
    epoch=lambda text:datetime.datetime.fromisoformat('2026-11-16T'+text+':00+08:00').timestamp()
    p.update(id='T582_FRESH_OLD_TASK_POLICY_GRADIENT_PAIR',out=str(OUT),training_jobs=training,
        training_seeds=[[[582,t,k] for k in range(4)] for t in range(6)],arms=['NEW_ONLY','NEW_PLUS_OLD'],
        old_family_weight=1.,rl_steps=8,learning_seconds=2100,main_seconds=3600,audit_seconds=600,
        transport_seconds=300,source_seconds=180,absolute_learning_stop=epoch('12:10'),absolute_native_stop=epoch('12:40'),absolute_all_stop=epoch('13:00'),
        max_physics=cap,max_commands=160,max_bytes=49*2**20,child_bytes=49*2**20,
        inputs={},protected_files={},new_Actor_Adam_cap=16,shadow_Adam_cap=16,audit_discarded_Adam_cap=16,preflight_discarded_Adam_cap=12,
        new_teacher_queries=0,new_BC=0,new_Critic=0,independent_new_mechanism_budget_not_T581_extension=True,
        user_authorization='2026-11-16 user: 下一轮开始; proposed extension beyond12:00 STILL PENDING',authorization_approved=False,
        count_contract='T582 explicit fields only; inherited fields are loading compatibility.',
        scope='Shared24 same initial policy fresh episodes+24exact; NEW_ONLY vs NEW_PLUS_OLD each<=8Adam with same newtask coefficient1,oldfamily0vs1;49autonomous+key4exact each. No grid or automatic next launch.')
    for k in ['historical_baseline','baseline_jobs']:p.pop(k,None)
    assert cap==7521044
    upper=32*p['full_bytes']+90*p['compact_bytes']+3*p['model_bytes']+4*2**20+2*p['full_bytes']
    assert upper==50651136<=p['max_bytes']
    global_worst=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert global_worst<2**30
    assert __import__('shutil').disk_usage(ROOT).free>=40*2**30
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=L.references(p);base=L.Prior.fresh(p)
    assert base.phash()==res['behavior']['policy_hash']
    for g in data:assert torch.equal(L.loc(base,g['f'],g['base']),g['target'])
    a=R.E.N.S.arrays(old/'episodes/full_0.npz');x=a['states'][:16];arrays=[]
    q=L.Prior.fresh(p)
    for task in range(6):
        for seed in range(4):
            q.sample=True;q.rng=np.random.default_rng(582+task*4+seed);q.reset_noise();h=None;locations=[];tapes=[]
            with torch.no_grad():
                for z in x:
                    _,b,h,_=q.step(torch.tensor(z[None]),h);locations.append(b[0].numpy().copy());tapes.append(q.tape[-1])
            flags=np.zeros((len(x),3),np.uint8);flags[-1,2]=1
            arrays.append(dict(states=x.copy(),next_states=np.r_[x[1:],x[-1:]],locations=np.asarray(locations),
                raw_actions=np.asarray([t['raw_actions'] for t in tapes]),atom=np.asarray([t['atom'] for t in tapes]),
                logp=np.asarray([t['logp'] for t in tapes]),ownership=np.zeros(len(x),np.uint8),stochastic=np.ones(len(x),np.uint8),
                end_flags=flags,native_counts=np.full(len(x),100,np.int64),rewards=np.linspace(-.001,.001,len(x))*(1+.2*seed)))
    results=[];total=0
    for arm in p['arms']:
        obj,shadow=L.make(p),L.make(p);before=L.state(obj)
        u=L.fit(obj,arrays,data,p,arm,lambda:None,lambda *_:None,steps=2)
        v=L.fit(shadow,arrays,data,p,arm,lambda:None,lambda *_:None,steps=2)
        assert u==v and digest(L.state(obj))==digest(L.state(shadow)) and u['Actor_Adam']==2
        mature=L.state(obj);L.load(obj,before);assert digest(L.state(obj))==digest(before)
        L.load(obj,mature);assert digest(L.state(obj))==digest(mature)
        temp=DOC/('discarded_'+arm+'.pt.gz');model=L.export(obj,p,temp)
        assert L.frozen(p,temp).phash()==obj[0].phash();temp.unlink();total+=4
        results.append(dict(arm=arm,discarded_Adam=4,live_shadow_mature_rollback_exact=True,
            model_bytes=model['bytes'],old_family_first_loss=u['trace'][0]['old_family_policy_loss']))
    obj,other=L.make(p),L.make(p)
    L.fit(obj,arrays,data,p,'NEW_ONLY',lambda:None,lambda *_:None,steps=2)
    L.fit(other,arrays,data,dict(p,old_family_weight=0.),'NEW_PLUS_OLD',lambda:None,lambda *_:None,steps=2)
    assert digest(L.state(obj))==digest(L.state(other));total+=4;assert total==12
    # This flag must block registration even if a caller attempts it now.
    spec=importlib.util.spec_from_file_location('_t582_transport_preflight',HERE/'transport.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);assert mod.OUT==OUT
    assert 't582_runtime.py' in (HERE/'driver.py').read_text()
    for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
    save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_NOT_REGISTERED',new_native=0,new_navigation_Adam=0,
        synthetic_discarded_Adam=12,arms=results,old_gradient_zero_identity_exact=True,
        task_baselines_computed_independently=True,new_task_gradient_coefficient_identical=True,
        same_initial_T581_zero_residual_policy_hash=base.phash(),initial_keep_loss_exact_zero=True,
        same_frozen_body_and_actual_history_cache=True,transport_OUT_matches_runtime=True,
        own_storage_upper=upper,own_bytes=p['max_bytes'],global_worst=global_worst,
        partial_two_worker_slots_included=True,synthetic_NOT_navigation=True))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
    for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'SOURCES.md',DOC/'PREFLIGHT.json',Path(p['actor_configuration']),old/'audit.json']+[Path(r['path']) for r in p['keep_rows']]:p['inputs'][str(f)]=sha(f)
    for model in [p['base_model'],p['base_model']['parent'],p['base_model']['parent']['parent']]:
        assert sha(model['path'])==model['sha256'];p['inputs'][model['path']]=model['sha256'];p['protected_files'][model['path']]=model['sha256']
    for j in p['evaluation_jobs']:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
    save(DOC/'PROPOSAL.json',p)
    save(DOC/'BUDGET.json',dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',window_authorization='PENDING',
        max_physics=cap,max_commands=160,actual_worst_commands=158,main_seconds=3600,audit_seconds=600,
        transport_seconds=300,source_seconds=180,all_work_reserve_seconds=4720,
        own_bytes=p['max_bytes'],own_storage_upper=upper,global_worst=global_worst,
        training_unique=24,training_exact=24,autonomous=98,key_exact=8,full32_compact90=True,
        live_Actor_Adam_cap=16,shadow_Adam_cap=16,audit_historical_Adam_cap=16,preflight_discarded_Adam=12,
        proposed_learning_stop='12:10',proposed_native_stop='12:40',proposed_all_stop='2026-11-16 13:00+08:00',
        existing_authorized_all_stop='2026-11-16 12:00+08:00',no_existing_budget_extension=True))
    print('T582_PREPARED_PREFLIGHT_PASS_NO_NATIVE_NO_REGISTRATION',cap,total,global_worst,flush=True)


def register():
    assert not OUT.exists();p=read(DOC/'PROPOSAL.json');auth=read(DOC/'AUTHORIZATION.json')
    assert auth['approved'] is True and auth['all_stop_epoch']==p['absolute_all_stop']
    assert auth['max_physics']==p['max_physics'] and auth['max_commands']==p['max_commands']
    assert auth['own_bytes']==p['max_bytes'] and auth['user_reply']
    assert time.time()+4720<p['absolute_all_stop'] and time.time()+3600<p['absolute_native_stop'] and time.time()+2100<p['absolute_learning_stop']
    assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
    p.update(user_authorization=auth['user_reply'],authorization_approved=True)
    p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n'
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    R.verify(p,True);print('T582_REGISTERED_NOT_LAUNCHED',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()

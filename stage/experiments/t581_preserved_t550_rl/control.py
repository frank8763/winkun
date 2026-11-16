"""Finite registration: preserve the strong original function before learning."""
from pathlib import Path
import copy,datetime,json,sys,time
import t581_runtime as R
ROOT,HERE,OUT,DOC,L=R.ROOT,R.HERE,R.OUT,R.DOC,R.L
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np

def prepare():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    old=ROOT/'stage/runs/t580_meta_replay_transfer'
    assert read(old/'audit.json')['state']=='PASS' and read(old/'SEALED.json')['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(i)).exists() for i in read(old/'SEALED.json')['scientific_pids'])
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T580.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(old/'protocol.json'))
    for key in ['arms','blocks','inner_steps','rows_per_chunk','inner_lr','meta_gamma','batch_seed','initial_success_ids','inputs','protected_files']:
        p.pop(key,None)
    final=ROOT/'stage/runs/t550_offline_obstacle_relay/final407'
    allrows=R.G.rows(final);finalp=read(final.parent/'protocol.json');alljobs=finalp['final_jobs']
    assert len(allrows)==len(alljobs)==407
    by={r['job_id']:r for r in allrows};base=p['neural_teacher']
    assert all(r['policy_hash']==base['policy_hash'] for r in allrows)
    chosen=list(p['evaluation_jobs']);seen={j['id'] for j in chosen}
    # Fixed, evenly spaced consumed successful IDs, four per distance family.
    for low,high in [(0,275),(275,329),(329,383)]:
        choices=[j for j in alljobs[low:high] if j['id'] not in seen and by[j['id']]['success'] and not by[j['id']]['collision']]
        picks=[choices[int(i)] for i in np.linspace(0,len(choices)-1,4,dtype=int)]
        for j in picks:chosen.append(j);seen.add(j['id'])
    assert len(chosen)==len(seen)==49
    for j in chosen:
        assert next(z for z in alljobs if z['id']==j['id'])==j
    hist=R.G.rows(ROOT/'stage/runs/t579_frozen_progress_attribution')
    baseline=[r for r in hist if r['role']=='FROZEN_T550' and r['job_id'] in ['wide117_80_dev6_LEFT_00','t502_field_sector6_offset3']]
    assert len(baseline)==2 and all(not r['success'] for r in baseline)
    target=next(j for j in chosen if j['id']=='wide117_80_dev6_LEFT_00')
    keep=p['keep_rows']+p['repair_rows'];assert len(keep)==27
    keys=['wide117_80_dev6_LEFT_00','wide117_80_dev6_RIGHT_23','wide117_8m_dev6_LEFT_13','wide117_8m_dev6_RIGHT_13']
    epoch=lambda text:datetime.datetime.fromisoformat('2026-11-16T'+text+':00+08:00').timestamp()
    p.update(id='T581_PRESERVE_T550_FUNCTION_SINGLE_RESIDUAL_RL',out=str(OUT),base_model=base,
        initial=base,keep_rows=keep,repair_rows=[],evaluation_jobs=chosen,
        baseline_jobs=[next(j for j in chosen if j['id']==r['job_id']) for r in baseline],historical_baseline=baseline,
        initial_success_ids=[j['id'] for j in chosen if by[j['id']]['success'] and not by[j['id']]['collision'] and by[j['id']]['field_detail']['field_qualified_success']],
        target_job=target,key_replay_ids=keys,residual_seed=581,residual_scale=.1,
        rl_lr=3e-5,rl_steps=8,target_kl=.01,keep_weight=100.,training_seeds=[[581,0,k] for k in range(4)],
        main_seconds=1800,learning_seconds=900,audit_seconds=480,transport_seconds=300,source_seconds=180,
        absolute_learning_stop=epoch('11:15'),absolute_native_stop=epoch('11:30'),absolute_all_stop=epoch('12:00'),
        max_commands=70,max_bytes=22*2**20,child_bytes=22*2**20,full_bytes=1280*1024,compact_bytes=12*1024,model_bytes=256*1024,
        inputs={},protected_files={},new_Actor_Adam_cap=8,shadow_Adam_cap=8,audit_discarded_Adam_cap=8,
        preflight_discarded_Adam_cap=4,new_teacher_queries=0,new_BC=0,new_Critic=0,
        independent_new_mechanism_budget_not_T580_extension=True,
        count_contract='Only explicit T581 fields apply; inherited configuration is model loading compatibility.',
        user_authorization='2026-11-16 下一步开始 after T550 exact-function residual RL plan; within12:00 cutoff',
        scope='One fresh same-policy4episode batch,one8Adam-cap proposal; expanded49consumed autonomous+key4exact; retain ALL original success IDs and gain>=1. No grid or automatic next launch.')
    p['max_physics']=sum(j['case']['step_cap'] for j in p['baseline_jobs'])+8*target['case']['step_cap']+sum(j['case']['step_cap'] for j in chosen)+sum(j['case']['step_cap'] for j in chosen if j['id'] in keys)
    assert time.time()+1800<p['absolute_native_stop'] and time.time()+900<p['absolute_learning_stop']
    assert time.time()+1800+480+300+180+30<p['absolute_all_stop']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    obj=L.make(p);shadow=L.make(p);q=obj[0];parent=L.ORIGINAL_FROZEN(p,base['path']);body=L.fixed(q)
    data=L.references(p);assert all(torch.equal(L.loc(q,g['f'],g['base']),g['target']) for g in data)
    checks=[]
    with torch.no_grad():
        for row in keep+baseline:
            a=R.E.N.S.arrays(row['path']);x=torch.tensor(a['states'])
            assert torch.equal(q.sequence(x),parent.sequence(x))
            checks.append(dict(id=row['job_id'],decisions=len(x),initial_location_bitwise_equal=True))
        x=R.E.N.S.arrays(baseline[0]['path'])['states'][:16];h=hp=None
        for z in x:
            u,b,h,_=q.step(torch.tensor(z[None]),h);v,c,hp,_=parent.step(torch.tensor(z[None]),hp)
            assert torch.equal(u,v) and torch.equal(b,c) and torch.equal(h,hp)
        future=torch.tensor(x);changed=future.clone();changed[8:]+=1
        assert torch.equal(q.sequence(future)[:8],q.sequence(changed)[:8])
    aa=[]
    for seed in range(4):
        q.sample=True;q.rng=np.random.default_rng(581+seed);q.reset_noise();h=None;locations=[];tapes=[]
        with torch.no_grad():
            for z in x:
                u,b,h,_=q.step(torch.tensor(z[None]),h);locations.append(b[0].numpy().copy());tapes.append(q.tape[-1])
        flags=np.zeros((len(x),3),np.uint8);flags[-1,2]=1
        aa.append(dict(states=x.copy(),next_states=np.r_[x[1:],x[-1:]],locations=np.asarray(locations),
            raw_actions=np.asarray([t['raw_actions'] for t in tapes]),atom=np.asarray([t['atom'] for t in tapes]),
            logp=np.asarray([t['logp'] for t in tapes]),ownership=np.zeros(len(x),np.uint8),stochastic=np.ones(len(x),np.uint8),
            end_flags=flags,native_counts=np.full(len(x),100,np.int64),rewards=np.linspace(-.001,.001,len(x))*(1+.2*seed)))
    before=L.state(obj);u=L.fit(obj,aa,data,p,lambda:None,lambda *_:None,steps=2)
    v=L.fit(shadow,aa,data,p,lambda:None,lambda *_:None,steps=2)
    assert u==v and digest(L.state(obj))==digest(L.state(shadow)) and u['Actor_Adam']==2 and L.fixed(q)==body
    assert any(t['gradient_norm']>0 for t in u['trace'])
    mature=L.state(obj);L.load(obj,before);assert digest(L.state(obj))==digest(before)
    L.load(obj,mature);assert digest(L.state(obj))==digest(mature)
    temp=DOC/'discarded_checkpoint.pt.gz';m=L.export(obj,p,temp);assert L.frozen(p,temp).phash()==q.phash();temp.unlink()
    full=R.E.N.S.arrays(ROOT/'stage/runs/t577_offline_recovery_night/rl_04_round/episodes/full_0.npz')
    temp=DOC/'discarded_codec.npz';R.E.N.write_npz(temp,full,p['full_bytes']);codec=temp.stat().st_size;temp.unlink()
    upper=10*p['full_bytes']+45*p['compact_bytes']+2*p['model_bytes']+8*2**20
    assert upper<=p['max_bytes'] and R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
    save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE',new_native=0,new_navigation_Adam=0,
        synthetic_discarded_Adam=4,initial29_history_function_bitwise_equal=checks,
        same_step_action_hidden_density_verified=True,future_causality=True,initial_keep_loss_exact_zero=True,
        live_shadow_and_mature_ActorAdam_rollback_exact=True,synthetic_current_policy_credit=u['peer_batch'],
        synthetic_control_NOT_navigation_success=True,checkpoint_bytes=m['bytes'],codec_max_seen_bytes=codec,
        own_storage_upper=upper,global_worst=R.used(ROOT/'stage/runs')+p['max_bytes']+131072))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
    for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'SOURCES.md',DOC/'PREFLIGHT.json',Path(p['actor_configuration']),old/'audit.json']+[Path(r['path']) for r in keep+baseline]:p['inputs'][str(f)]=sha(f)
    for model in [base,base['parent'],base['parent']['parent']]:
        assert sha(model['path'])==model['sha256'];p['inputs'][model['path']]=model['sha256'];p['protected_files'][model['path']]=model['sha256']
    for j in chosen:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
    OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n'
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',max_physics=p['max_physics'],max_commands=70,actual_worst_commands=67,
        live_Actor_Adam_cap=8,shadow_Adam_cap=8,audit_historical_Adam_cap=8,preflight_discarded_Adam=4,
        same_initial_success_count=len(p['initial_success_ids']),evaluations=49,training_unique=4,exact_replays=8,
        historical_physical_repeats=2,own_bytes=p['max_bytes'],full10_compact45=True,main_seconds=1800,audit_seconds=480,
        learning_stop='11:15',native_stop='11:30',all_stop='2026-11-16 12:00+08:00'))
    print('T581_PREFLIGHT_REGISTERED_NOT_LAUNCHED',p['max_physics'],len(p['initial_success_ids']),flush=True)

if __name__=='__main__':prepare()

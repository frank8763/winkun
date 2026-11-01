from t280_core import *
import ast,builtins,copy,dis,importlib.util,io,json,types
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t280_local_cost'


def main():
    assert not (OUT/'preflight.json').exists();p=json.loads((HERE/'protocol.json').read_text())
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    for filename,entry in [('run.py','main'),('t280_worker.py','generate')]:
        spec=importlib.util.spec_from_file_location('entry_'+filename.replace('.','_'),HERE/filename);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);missing=set()
        def check(code):
            for instruction in dis.get_instructions(code):
                if instruction.opname=='LOAD_GLOBAL' and instruction.argval not in vars(mod) and instruction.argval not in vars(builtins):missing.add(instruction.argval)
            for co in code.co_consts:
                if isinstance(co,types.CodeType):check(co)
        check(getattr(mod,entry).__code__);assert not missing,(filename,missing)
    cap=28000;n=np.full(280,100,dtype=np.uint8);weights=R.GAMMA**np.r_[0,np.cumsum(n.astype(np.int64))[:-1]]
    dummy=dict(states=np.zeros((280,120),np.float32),next_states=np.zeros((280,120),np.float32),native_counts=n);dummy['states'][0,48]=.1;dummy['next_states'][:,48]=.1
    c=endpoint_cost(dummy,cap);assert abs(float(weights@c)-1)<1e-12
    dummy['next_states'][:,48]=.02;c2=endpoint_cost(dummy,cap);assert 0<float(weights@c2)<float(weights@c)
    length=np.arange(1,cap+1);timecost=-.000002*(1-R.GAMMA**length)/(1-R.GAMMA)
    goal=timecost+R.GAMMA**(length-1);contact=timecost-R.GAMMA**(length-1);timeout=float(timecost[-1]-.5*R.GAMMA**(cap-1))
    assert float(goal.min())-.1>timeout and timeout-.1>float(contact.max())
    q=policy(p);opt=L.optimizer(q,p);raw=io.BytesIO();torch.save(q.state_dict(),raw)
    model_raw_bound=3*raw.tell()+256*1024;assert model_raw_bound<p['model_bytes']
    rows=list(map(json.loads,(ROOT/'stage/runs/t263_adaptive_goal_course/training.jsonl').read_text().splitlines()))[:16];items=[]
    for row in rows:
        assert C.sha(row['path'])==row['sha256']
        with np.load(row['path']) as z:a={k:z[k].copy() for k in z.files}
        items.append(dict(row=row,arrays=a))
    format_sizes=[]
    for item in items:
        a=dict(item['arrays']);a['environment_rewards']=a['rewards'].copy();a['aux_costs']=endpoint_cost(a,28000);a['rewards']=a['environment_rewards']-.1*a['aux_costs']
        full=io.BytesIO();np.savez_compressed(full,**a)
        compact=io.BytesIO();np.savez_compressed(compact,**{k:a[k] for k in ['actions','native_counts','end_flags','rewards','environment_rewards','aux_costs']})
        format_sizes.append(dict(full=full.tell(),compact=compact.tell()))
    rng=np.random.default_rng(2026111800);worst=io.BytesIO()
    np.savez_compressed(worst,actions=rng.normal(size=(280,2)).astype(np.float32),native_counts=np.full(280,100,np.uint8),end_flags=rng.integers(0,2,size=(280,3),dtype=np.uint8),rewards=rng.normal(size=280),environment_rewards=rng.normal(size=280),aux_costs=rng.random(280))
    assert max(x['full'] for x in format_sizes)<=p['training_episode_bytes'] and max(x['compact'] for x in format_sizes)<=p['compact_episode_bytes'] and worst.tell()<=p['compact_episode_bytes']
    preserved=sum(f.stat().st_size for f in OUT.glob('preflight_failed_*/*') if f.is_file())+sum(f.stat().st_size for f in OUT.glob('preflight_failed_*/source/*') if f.is_file())
    metadata_upper=(p['max_training_payloads']+p['max_compact_payloads'])*2560+64*2560+17*32768+4*32768+4*2560+16*2560+4*65536+8*16384+65536+2*(HERE/'protocol.json').stat().st_size+preserved
    assert metadata_upper<=p['metadata_bytes'],metadata_upper
    d=L.training_batch(items,q);ax,ay,offset=anchors(p,copy.deepcopy(q).eval().requires_grad_(False))
    local=dict(p,actor_attempts_per_round=1);clone_calls=0
    def counted(optimizer):
        original=optimizer.step
        def run(*args,**kw):
            nonlocal clone_calls
            clone_calls+=1;return original(*args,**kw)
        optimizer.step=run
    counted(opt);shadow=copy.deepcopy(q);so=L.optimizer(shadow,p);counted(so)
    events=[]
    live,audit=paired_update(q,opt,shadow,so,d,ax,ay,offset,local,0,lambda:None,
        lambda actor:dict(policy_hash=S.digest(actor.state_dict()),cases=20,safe=20,all_safe=True,failed_ids=[]),lambda phase,event:events.append(dict(phase=phase,**event)))
    assert live==audit and S.digest(q.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(so.state_dict())
    before=S.digest(q.state_dict());before_opt=S.digest(opt.state_dict())
    for tensor in q.parameters():
        if tensor.requires_grad:tensor.grad=torch.zeros_like(tensor)
    L.guarded_adam_step(q,opt,lambda actor:dict(accept=False),[1.,.25]);assert S.digest(q.state_dict())==before and S.digest(opt.state_dict())==before_opt
    def interrupt(actor):raise RuntimeError('registered disposable-clone interrupt')
    try:L.guarded_adam_step(q,opt,interrupt,[1.])
    except RuntimeError:pass
    else:raise AssertionError('Expected interrupt')
    assert S.digest(q.state_dict())==before and S.digest(opt.state_dict())==before_opt and clone_calls<=p['preflight_clone_optimizer_max_calls']
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes'] and S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes']
    previous=sum(json.loads(f.read_text()).get('disposable_clone_optimizer_calls',0) for f in OUT.glob('preflight_failed_*/preflight.json'))
    assert previous+clone_calls<=p['preflight_total_clone_optimizer_max_calls']
    C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,native_envs_created=0,navigation_Actor_updates=0,
        disposable_clone_optimizer_calls=clone_calls,canonical_real_history_density=d['density'],
        total_preflight_clone_optimizer_calls_including_preserved_attempts=previous+clone_calls,
        actual_old_16case_max_full12_bytes=max(x['full'] for x in format_sizes),actual_old_16case_max_compact6_bytes=max(x['compact'] for x in format_sizes),synthetic_high_entropy_280hold_compact_bytes=worst.tell(),
        metadata_worst_case_upper_bytes=metadata_upper,
        synthetic_endpoint_cost_bound_exact=True,trajectory_class_ordering_margin_after_max_aux=float(timeout-.1-contact.max()),
        objective_is_not_policy_invariant=True,worker_and_nested_main_global_names_resolve=True,paired_clone_Actor_Adam_exact=True,
        rejection_and_interruption_restore_entire_Actor_Adam=True,full_model_raw_bound=model_raw_bound,
        full_payload_and_metadata_reserved=p['payload_bytes']+p['metadata_bytes'],protocol_sha256=C.sha(HERE/'protocol.json'),
        synthetic_old20_not_real_old20_safety=True,real_new_reset_and_failure_sorting_not_yet_executed=True))
    print(json.dumps(dict(state='PASS',clone_optimizer_calls=clone_calls,physics_steps=0)),flush=True)


if __name__=='__main__':main()

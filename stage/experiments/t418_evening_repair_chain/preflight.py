"""One root preflight: eight discarded synthetic Adam steps, zero physics."""
from repair_core import *
import repair_core as C
import importlib.util,dis,builtins,types

ROOT_OUT=ROOT/'stage/runs/t418_evening_repair_chain'

def globals_check():
    for name in ['run.py','audit.py','driver.py']:
        spec=importlib.util.spec_from_file_location('stage418_pref_'+Path(name).stem,HERE/name)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={v.argval for v in dis.get_instructions(code) if v.opname=='LOAD_GLOBAL' and v.argval not in vars(module) and not hasattr(builtins,v.argval)}
            for item in code.co_consts:
                if isinstance(item,types.CodeType):bad|=check(item)
            return bad
        for item in vars(module).values():
            if isinstance(item,types.FunctionType) and item.__module__==module.__name__:
                assert not check(item.__code__),(name,check(item.__code__))

def fixture(q,p):
    """Synthetic own-location anchors on existing causal observations.

    This tests the anchoring operator, never asserts a new rollout/success.
    Real phase anchors must be rebuilt from the initial14 native histories.
    """
    origin=ROOT/'stage/runs/t417_representation_adaptation'
    rows=[json.loads(line) for line in (origin/'cases.jsonl').read_text().splitlines()]
    olds=[N.S.arrays(r['path']) for r in rows[:8]]
    arrays=olds+[N.S.arrays(path) for path in p['left_history_files']]+[N.S.arrays(p['right_history_file'])]
    jobs=[*p['old8_jobs'],*p['left5_jobs'],p['right_job']];success=set(p['initial_expected_success_ids']);items=[]
    for a,job in zip(arrays,jobs):
        a=dict(a)
        with torch.no_grad():loc=q.sequence(torch.tensor(a['states']),q.base_actions(torch.tensor(a['states']))).numpy()
        a['locations']=loc.copy();a['actions']=loc.clip(-1,1).astype(np.float32)
        items.append((a,dict(job_id=job['id'],success=job['id'] in success)))
    g=build_groups(q,items,p);assert len(g['anchors'])==12
    # Historical successful expert labels are legal synthetic fitting fixtures;
    # they are not new actual-state corrections or PPO transitions.
    queries=read(origin/'blocks/0/result.json')['queries']
    g['corrections']=[query_group(q,N.S.arrays(row['row']['path'])) for row in queries]
    assert max(float(old_kl(q,item,p)) for item in g['anchors'])<1e-26
    return g,olds

def carry(q,x):
    assert q.core.batch_first is False
    q.reset_noise();h=None;locs=[];hs=[]
    with torch.no_grad():
        for row in x:
            _,loc,h,_=q.step(row[None],h);locs.append(loc[0]);hs.append(h[0,0])
        whole,last=q.core(q.encoder(q.normalized(x))[:,None])
        le=float((q.sequence(x,q.base_actions(x))-torch.stack(locs)).abs().max())
        he=float((whole[:,0]-torch.stack(hs)).abs().max())
        assert le<1e-10 and he<1e-10 and float((last-h).abs().max())<1e-10
    return dict(location_error=le,hidden_error=he,batch_first=False)

def main():
    p=read(ROOT_OUT/'protocol.json');assert (HERE/'protocol.json').read_bytes()==(ROOT_OUT/'protocol.json').read_bytes()
    assert not (ROOT_OUT/'preflight.json').exists()
    for path,h in p['inputs'].items():assert sha(path)==h,path
    assert sha(p['student_start_model'])==p['student_start_sha256']
    assert N.K.S.used(ROOT/'stage/runs')+3*p['final_bytes']<2**30
    globals_check();base=N.science(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    total=0;checks=[]
    for phase,count in [('repair',4450),('protect',10722)]:
        config=dict(p,phase=phase,out=str(ROOT_OUT));records=[]
        for clone in range(2):
            q,opt=learner(base,config);assert q.phash()==p['initial_expected_policy_hash']
            assert sum(v.numel() for v in params(q))==count
            fixed=original_hash(q);g,old=fixture(q,config);snapshots=[]
            def event(where,k):
                nonlocal total
                with (ROOT_OUT/'preflight_optimizer_calls.jsonl').open('a') as stream:stream.write(json.dumps([phase,clone,where,k])+'\n')
                if where=='AFTER':total+=1
                snapshots.append(L.state(q,opt)) if where=='AFTER' else None
            fit=fit_bc(q,opt,g,config,lambda:None,event,steps=2)
            assert len(snapshots)==2 and snapshots[0]['optimizer']['state']
            assert original_hash(q)==fixed
            assert all(v.grad is None for name,v in q.named_parameters() if not name.startswith(trained_prefixes(q)))
            assert torch.count_nonzero(q.extra[-1].weight)>0
            cc=carry(q,torch.tensor(old[0]['states']))
            # Fresh iid replay density uses actual float32 boundary atoms.
            q.reset_noise();q.rng=np.random.default_rng(418);h=None
            with torch.no_grad():
                for x in old[0]['states'][:12]:
                    q.sample=True;_,loc,h,_=q.step(torch.tensor(x[None]),h);row=q.tape[-1]
                    lp=L.logp(loc,torch.tensor(row['raw_actions'][None]),torch.tensor(row['atom'][None]),config)
                    assert abs(float(lp)-float(row['logp']))<1e-12
            complete=L.statehash(q,opt);L.load(q,opt,snapshots[0]);assert digest(L.state(q,opt))==digest(snapshots[0])
            try:raise L.StageStop('preflight synthetic interruption')
            except L.StageStop:L.load(q,opt,snapshots[0])
            assert digest(L.state(q,opt))==digest(snapshots[0])
            records.append(dict(fit=fit,complete=complete,rollback=L.statehash(q,opt),carry=cc))
        assert records[0]==records[1];checks.append(dict(phase=phase,parameters=count,exact_clones=True,carry=records[0]['carry']))
    assert total==8
    # One set of workers tests the shared native path. No native step is taken.
    config=dict(p,phase='repair',out=str(ROOT_OUT),native_stop_epoch=time.time()+180)
    set_out(ROOT_OUT);q,opt=learner(base,config);model=export(q,opt,ROOT_OUT/'preflight_clone.pt.gz')
    assert Path(model['path']).stat().st_size<=p['model_bytes']
    counts=mp.get_context('spawn').RawArray('q',5);pool=Pool(config,counts,lambda:None)
    try:
        ids=[child.pid for child in pool.children]
        result=pool.request([dict(model=model['path'],job=p['right_job'],first=0,seed=None,cancel_probe=True)]*4)
        assert all(row['state']=='CANCEL_PASS' for row in result) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(ROOT_OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=total,zero_physics_commands=8,
        phase_trainability=checks,all_consumed_input_SHA=True,recursive_actual_entry_globals=True,
        nonzero_readout_sequence_step_hidden_carry=True,synthetic_anchor_loss_initial_zero=True,
        synthetic_anchors_not_new_actual_rollout_evidence=True,clones_and_mature_Adam_rejection_interruption_rollback_exact=True,
        iid_float32_atom_density_exact=True,actual_four_native_worker_cancel_and_exit=True,workers=ids))
    N.manifest(ROOT_OUT);print('T418_PREFLIGHT_PASS_0NATIVE_8DISCARDED_ADAM',flush=True)

if __name__=='__main__':main()

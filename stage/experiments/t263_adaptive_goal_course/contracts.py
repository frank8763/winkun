"""Disposable synthetic proposal/shadow integration, no navigation interaction."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import *
from course import contracts as course_contracts
import json,math,io
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t263_adaptive_goal_course'
def main():
    curriculum_contract=course_contracts();plan=json.loads((HERE/'protocol.json').read_text());assert not (OUT/'contracts.json').exists()
    import importlib.util,dis,types,builtins
    spec=importlib.util.spec_from_file_location('stage263_entry_contract',HERE/'run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);missing=set()
    def check(code):
        for ins in dis.get_instructions(code):
            if ins.opname=='LOAD_GLOBAL' and ins.argval not in vars(module) and ins.argval not in vars(builtins):missing.add(ins.argval)
        for constant in code.co_consts:
            if isinstance(constant,types.CodeType):check(constant)
    check(module.main.__code__);assert not missing,sorted(missing)

    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p=policy(plan);original=S.digest(p.state_dict());parent=copy.deepcopy(p).eval().requires_grad_(False);x,y,offset=anchors(plan,parent);mse,mx=anchor_measure(p,x,y,offset);assert float(mse)==float(mx)==0
    rng=np.random.default_rng(9101);items=[]
    for task in range(4):
        for k in range(4):
            xx=x[task*24:(task+1)*24].numpy();means=[];acts=[];logps=[];latent=None
            for row in xx:
                first=latent is None;eps=rng.normal(size=2);latent=eps if first else L.AR.RHO*latent+math.sqrt(1-L.AR.RHO**2)*eps
                with torch.no_grad():
                    _,mean,_,_=p.step(torch.tensor(row[None]));mean=mean[0];a=L.AR.sample(mean,latent);lp=L.AR.logp(mean[None],a[None],(mean if first else means[-1])[None],(a if first else acts[-1])[None],torch.tensor([first]))
                means.append(mean);acts.append(a);logps.append(float(lp))
            reward=np.zeros(24);reward[-1]=1 if k%2 else -1;items.append(dict(row=dict(source_reset_id=str(task)),arrays=dict(states=xx,means=torch.stack(means).numpy(),actions=torch.stack(acts).numpy(),logp=np.asarray(logps),native_counts=np.full(24,100,dtype=np.uint8),rewards=reward)))
    buf=io.BytesIO();np.savez_compressed(buf,**items[0]['arrays']);assert len(buf.getvalue())<plan['training_episode_bytes']
    with np.load(io.BytesIO(buf.getvalue())) as stored:assert 'states' in stored.files and all(np.array_equal(stored[k],v) for k,v in items[0]['arrays'].items())
    data=L.training_batch(items,p);q=copy.deepcopy(p);opt=L.optimizer(p,plan);qo=L.optimizer(q,plan);testplan=dict(plan,actor_attempts_per_round=1);live,audit=paired_update(p,opt,q,qo,data,x,y,offset,testplan,0,lambda:None,lambda actor:dict(all_safe=True,cases=len(plan['acceptance_guard_jobs']),synthetic_only=True,policy_hash=S.digest(actor.state_dict())),lambda phase,event:None);assert live==audit and S.digest(p.state_dict())==S.digest(q.state_dict()) and S.digest(opt.state_dict())==S.digest(qo.state_dict())
    # A failed native guard must roll back Actor and Adam in both copies.
    rejected=policy(plan);ro=L.optimizer(rejected,plan);rq=copy.deepcopy(rejected);rqo=L.optimizer(rq,plan);rh=S.digest(rejected.state_dict());oh=S.digest(ro.state_dict());calls=[]
    def deny(actor):calls.append(True);return dict(all_safe=False,cases=len(plan['acceptance_guard_jobs']),synthetic_only=True,policy_hash=S.digest(actor.state_dict()))
    rr,aa=paired_update(rejected,ro,rq,rqo,data,x,y,offset,dict(testplan,step_factors=[.015625]),0,lambda:None,deny,lambda phase,event:None)
    assert calls and not any(r['accepted'] for r in rr) and S.digest(rejected.state_dict())==rh and S.digest(ro.state_dict())==oh
    # An interruption after a live step rolls back the entire unaudited round.
    interrupted=policy(plan);interrupted_opt=L.optimizer(interrupted,plan);iq=copy.deepcopy(interrupted);iqo=L.optimizer(iq,plan);ih=S.digest(interrupted.state_dict());ioh=S.digest(interrupted_opt.state_dict());ledger=[]
    def interrupt(phase,event):
        ledger.append(dict(phase=phase,**event))
        if phase=='live':raise RuntimeError('synthetic interruption after recorded live proposal')
    try:paired_update(interrupted,interrupted_opt,iq,iqo,data,x,y,offset,dict(testplan,step_factors=[.015625]),0,lambda:None,lambda actor:dict(all_safe=True,cases=len(plan['acceptance_guard_jobs']),synthetic_only=True,policy_hash=S.digest(actor.state_dict())),interrupt)
    except RuntimeError:pass
    else:raise AssertionError('Interruption was not propagated')
    assert S.digest(interrupted.state_dict())==ih and S.digest(interrupted_opt.state_dict())==ioh and ledger[-1]['phase']=='rollback'
    result=dict(state='PASS',curriculum_contract=curriculum_contract,full_training_states_storage_contract=True,entrypoint_all_nested_global_names_resolve=True,protocol_sha256=C.sha(HERE/'protocol.json'),physics_steps=0,navigation_RL_updates=0,disposable_synthetic_optimizer_proposals=len(live)+len(audit)+len(rr)+len(aa)+sum(e['phase']=='live' for e in ledger),failed_native_guard_rollback_exact=True,interrupted_round_rollback_exact=True,native_guard_is_synthetic_here=True,initial_policy_hash=original,retention_states=len(x),initial_full_retention_MSE=0,initial_full_retention_max=0,density=data['density'],synthetic_proposals_shadow_exact=True,synthetic_report=live,old_models_unchanged=True)
    C.save(OUT/'contracts.json',result);print(json.dumps(dict(state='PASS',retention_states=len(x),shadow_exact=True,density=data['density'])),flush=True)
if __name__=='__main__':main()

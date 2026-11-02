from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t291_hard_guard_ablation'
sys.path.insert(0,str(ROOT/'stage/experiments/t282_capped_balance'))
from balanced_learning import *
sys.path.insert(0,str(ROOT/'stage/experiments/t287_time_aligned_credit'))
from t287_credit import *
from t291_learning import paired_capability_update
from t291_codec import pack_full,unpack_full
import ast,builtins,copy,dis,importlib.util,io,json,types


def main():
    assert not (OUT/'preflight.json').exists()
    p=json.loads((HERE/'protocol.json').read_text());assert not any((OUT/f).exists() for f in ['launch.json','status.json','summary.json'])
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    for filename in ['run.py','audit.py']:
        spec=importlib.util.spec_from_file_location('t287_'+filename[:-3],HERE/filename);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);missing=set()
        def check(code):
            for ins in dis.get_instructions(code):
                if ins.opname=='LOAD_GLOBAL' and ins.argval not in vars(mod) and ins.argval not in vars(builtins):missing.add(ins.argval)
            for co in code.co_consts:
                if isinstance(co,types.CodeType):check(co)
        check(mod.main.__code__);assert not missing,(filename,missing)
    items=[]
    for r in p['historical_first16']:
        with np.load(r['path']) as z:a={k:z[k].copy() for k in z.files}
        assert all(S.digest(v)==r['array_hashes'][k] for k,v in a.items());items.append(dict(row=r,arrays=a))
    credit_contract=enumeration_contract();C.save(OUT/'credit_contract.json',credit_contract)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=policy(p);opt=L.optimizer(q,p);frozen=copy.deepcopy(q).eval().requires_grad_(False)
    ax,ay,offset=anchors(p,frozen);d=L.training_batch(items,q);calls=[]
    def count(optimizer,phase):
        original=optimizer.step
        def invoke(*args,**kw):
            assert len(calls)+1<p['preflight_clone_optimizer_max_calls'];calls.append(phase)
            C.save(OUT/'preflight_optimizer_calls.json',dict(calls=calls,navigation_updates=0,physics_steps=0));return original(*args,**kw)
        optimizer.step=invoke
    shadow=copy.deepcopy(q);so=L.optimizer(shadow,p);count(opt,'clone_live');count(so,'clone_shadow');local=dict(p,actor_attempts_per_round=1);events=[]
    live,audit=paired_capability_update(q,opt,shadow,so,d,ax,ay,offset,local,0,lambda:None,
        lambda actor:dict(policy_hash=S.digest(actor.state_dict()),cases=0,safe=0,all_safe=False,failed_ids=['SYNTHETIC_NO_NATIVE_TEST'],synthetic=True),lambda phase,event:events.append(dict(phase=phase,**event)))
    assert live[0]['accepted'] and all(not t['native_old_task_guard']['all_safe'] for t in live[0]['attempts'] if t.get('native_old_task_guard') is not None)
    assert live==audit and S.digest(q.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(so.state_dict())
    balance=live[0]['fixed_round_balance'];assert all(1<=f<=p['balance_cap'] for f in balance['factors'])
    assert abs(balance['rescaled_weighted_PCGrad_norm']/balance['original_PCGrad_norm']-1)<2e-6
    before=S.digest(q.state_dict());before_opt=S.digest(opt.state_dict())
    for tensor in q.parameters():
        if tensor.requires_grad:tensor.grad=torch.zeros_like(tensor)
    L.guarded_adam_step(q,opt,lambda actor:dict(accept=False),p['step_factors']);assert S.digest(q.state_dict())==before and S.digest(opt.state_dict())==before_opt
    def interrupt(actor):raise RuntimeError('T287 disposable clone interrupt')
    try:L.guarded_adam_step(q,opt,interrupt,p['step_factors'])
    except RuntimeError:pass
    else:raise AssertionError('Expected registered interrupt')
    assert S.digest(q.state_dict())==before and S.digest(opt.state_dict())==before_opt and len(calls)==4
    C.save(OUT/'preflight_clone_events.json',dict(events=events,synthetic_acceptance_not_native20=True))
    raw=io.BytesIO();torch.save(q.state_dict(),raw);assert raw.tell()*3+256*1024<p['model_bytes']
    full_sizes=[];compact_sizes=[]
    for i in items:
        a=i['arrays'];f=io.BytesIO();np.savez_compressed(f,**pack_full(a));assert all(np.array_equal(v,unpack_full(pack_full(a))[k]) for k,v in a.items());c=io.BytesIO();np.savez_compressed(c,**{k:a[k] for k in ['actions','native_counts','end_flags','rewards','environment_rewards','aux_costs']});full_sizes.append(f.tell());compact_sizes.append(c.tell())
    assert max(full_sizes)<=p['training_episode_bytes'] and max(compact_sizes)<=p['compact_episode_bytes']
    metadata_upper=(p['max_training_payloads']+p['max_compact_payloads'])*2560+64*2560+10*32768+4*32768+4*2560+16*2560+4*65536+8*16384+65536+2*(HERE/'protocol.json').stat().st_size+S.used(OUT)
    assert metadata_upper<p['metadata_bytes'] and p['payload_bytes']+p['metadata_bytes']<p['final_bytes']
    outside=S.used(ROOT/'stage/runs')-S.used(OUT);assert outside+p['final_bytes']<p['global_bytes']
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,native_environments_created=0,navigation_Actor_updates=0,
        disposable_clone_optimizer_calls=len(calls),earlier_interrupted_invocations_upper_bound=1,total_preflight_optimizer_invocations_upper_bound=len(calls)+1,hard_old20_disabled_only_for_research=True,lossless_full120_chain_verified=True,clone_Actor_Adam_shadow_exact=True,rejection_and_interruption_Actor_Adam_restore_exact=True,
        fixed_round_balance=balance,canonical_density=d['density'],synthetic_old20_not_real_native=True,
        all_nested_run_and_audit_global_names_resolve=True,full_payload_and_metadata_reserved=p['payload_bytes']+p['metadata_bytes'],metadata_upper=metadata_upper,
        global_if_own_full=outside+p['final_bytes'],protocol_sha256=C.sha(HERE/'protocol.json'),launch_blocked_pending_source_sync_and_GitHub_backup=True,
        historical_control_consumed_development_not_new_concurrent_replication=True))
    print(json.dumps(dict(state='PASS_LOCAL_ONLY',clone_optimizer_calls=len(calls),balance=balance)),flush=True)


if __name__=='__main__':main()

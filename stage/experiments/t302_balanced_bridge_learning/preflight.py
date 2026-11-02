from t302_learning import *
import json,io,builtins,dis,types,importlib.util
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t302_balanced_bridge_learning'

def main():
    assert not (OUT/'preflight.json').exists();p=json.loads((HERE/'protocol.json').read_text());torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    for filename in ['run.py','audit.py']:
        spec=importlib.util.spec_from_file_location('t302_'+filename[:-3],HERE/filename);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);missing=set()
        def check(code):
            for ins in dis.get_instructions(code):
                if ins.opname=='LOAD_GLOBAL' and ins.argval not in vars(m) and ins.argval not in vars(builtins):missing.add(ins.argval)
            for v in code.co_consts:
                if isinstance(v,types.CodeType):check(v)
        check(m.main.__code__);assert not missing,(filename,missing)
    q=policy(p);items=auxiliary(p);scale=gradient_scale(q,items);C.save(OUT/'gradient_scale.json',scale);p['auxiliary_NLL_scale']=scale['factor'];p['input_hashes'][str(OUT/'gradient_scale.json')]=C.sha(OUT/'gradient_scale.json')
    for f in [HERE/'protocol.json',OUT/'protocol.json']:f.write_text(json.dumps(p,separators=(',',':'))+'\n')
    frozen=copy.deepcopy(q).eval().requires_grad_(False);ax,ay,offset=anchors(p,frozen);calls=[]
    def count(o,name):
        original=o.step
        def step(*a,**kw):
            assert len(calls)<4;calls.append(name);C.save(OUT/'preflight_optimizer_calls.json',dict(calls=calls,native_steps=0,navigation_updates=0));return original(*a,**kw)
        o.step=step
    shadow=copy.deepcopy(q);opt=L.optimizer(q,p);so=L.optimizer(shadow,p);count(opt,'clone_live');count(so,'clone_shadow');before=S.digest(q.state_dict());bo=S.digest(opt.state_dict());events=[]
    small=dict(p,supervised_blocks=1,supervised_block_steps=2)
    report=supervised_blocks(q,shadow,opt,so,items,ax,ay,offset,small,lambda:None,lambda actor:dict(policy_hash=S.digest(actor.state_dict()),cases=0,safe=0,all_safe=False,synthetic=True),events.append)
    assert len(calls)==4 and report['attempted_steps']==2 and report['retained_steps']==0 and S.digest(q.state_dict())==S.digest(shadow.state_dict())==before and S.digest(opt.state_dict())==S.digest(so.state_dict())==bo
    # An interrupted block must restore both models/Adams before propagating.
    def fail_guard():raise RuntimeError('synthetic pre-optimizer block interruption')
    try:supervised_blocks(q,shadow,opt,so,items,ax,ay,offset,small,fail_guard,lambda actor:None,lambda row:None)
    except RuntimeError:pass
    else:raise AssertionError('expected interruption')
    assert S.digest(q.state_dict())==before and S.digest(opt.state_dict())==bo
    raw=io.BytesIO();torch.save(dict(policy=q.state_dict(),optimizer=L.optimizer(q,p).state_dict(),report=report),raw);assert raw.tell()+16384<p['model_bytes']
    assert 200+3+3+275+54+16+78+4<=p['max_compact_payloads'] and p['max_compact_payloads']<=p['max_commands'] and p['max_commands']*28000==p['max_physics']
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes']
    C.save(OUT/'preflight_clone_events.json',dict(report=report,events=events,synthetic_native=False))
    C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,native_environments_created=0,disposable_optimizer_calls=4,new_navigation_updates=0,live_shadow_exact=True,failed_block_restores_whole_ActorAdam_exact=True,interruption_restores_ActorAdam=True,side_balance_weights=task_weights(),frozen_gradient_scale=scale,model_raw_bytes=raw.tell(),all_global_names_resolve=True,global_if_full=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']));print('T302 preflight PASS4disposableAdam/0native, block rejection rollback exact')

if __name__=='__main__':main()

"""Zero-optimizer label, split, hazard, gradient and capacity contracts."""
from event_model import *
import builtins
import dis
import importlib.util
import shutil
import types
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t272_event_value_pilot'


def main():
    assert not (OUT/'preflight.json').exists()
    p=json.loads((HERE/'protocol.json').read_text())
    for path,h in p['input_hashes'].items():assert sha(path)==h,path
    spec=importlib.util.spec_from_file_location('t272_entry',HERE/'train.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    missing=set()
    def check(code):
        for i in dis.get_instructions(code):
            if i.opname=='LOAD_GLOBAL' and i.argval not in vars(mod) and i.argval not in vars(builtins):missing.add(i.argval)
        for c in code.co_consts:
            if isinstance(c,types.CodeType):check(c)
    check(mod.main.__code__);assert not missing,sorted(missing)
    d=dataset(p);assert len(d['train_ids'])==80
    # Split by native pose-offset family, not adjacent states or replay copies.
    a={r['family'] for r in p['records'] if r['split']=='train'}
    b={r['family'] for r in p['records'] if r['split']=='test'};assert not a&b
    tr={tuple(r['signature']) for r in p['records'] if r['split']=='train'}
    te={tuple(r['signature']) for r in p['records'] if r['split']=='test'};assert not tr&te
    assert len(b)==1 and len(te)==16
    # Native class order was historically reversed in T262; independently test.
    for flags,c in [([[0,0,0],[0,1,0]],0),([[0,0,0],[1,0,0]],1),([[0,0,0],[0,0,1]],2)]:
        y,m=native_targets([100,50],flags,150,True,[50,100,150])
        assert y[0,2,c]==1 and m[0,2]
    y,m=native_targets([100],[[0,0,0]],300,False,[100,200]);assert m[0,0] and not m[0,1]
    torch.manual_seed(p['seed']);q=EventModel();z=q(d['x'][:128])
    assert float((z.sum(-1)-1).abs().max())<3e-7
    assert bool((torch.diff(z[:,:,:2],dim=1)>=-1e-7).all())
    assert bool((torch.diff(z[:,:,2],dim=1)<=1e-7).all())
    loss=-(d['y'][:128]*z.clamp_min(1e-8).log()).sum();loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in q.parameters())
    n=sum(v.numel() for v in q.parameters());assert n<15000
    assert used(ROOT/'stage/runs')-used(OUT)+p['final_bytes']<p['global_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,
        records=96,decision_rows=len(d['x']),train_episodes=80,test_episodes=16,
        family_and_exact_trajectory_split_disjoint=True,labels_native_order_and_censoring_checked=True,
        original_goal_history_unchanged=True,train_only_normalization=True,probability_mass_and_monotonicity=True,
        finite_all_parameter_gradients=True,model_parameters=n,protocol_sha256=sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='PASS',parameters=n)),flush=True)


if __name__=='__main__':main()

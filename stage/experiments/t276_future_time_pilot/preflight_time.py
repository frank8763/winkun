from time_model import *
import builtins,dis,importlib.util,shutil,types
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t276_future_time_pilot'


def main():
    assert not (OUT/'preflight.json').exists();p=json.loads((HERE/'protocol.json').read_text())
    for f,h in p['input_hashes'].items():assert sha(f)==h,f
    spec=importlib.util.spec_from_file_location('t276_entry',HERE/'train_time.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    missing=set()
    def check(code):
        for i in dis.get_instructions(code):
            if i.opname=='LOAD_GLOBAL' and i.argval not in vars(mod) and i.argval not in vars(builtins):missing.add(i.argval)
        for c in code.co_consts:
            if isinstance(c,types.CodeType):check(c)
    check(mod.main.__code__);assert not missing,sorted(missing)
    d=load_data(p);assert len(d['train_ids'])==80 and int(d['train'].sum())==21836 and (~d['train']).sum()==4559
    coef=fit_ridge(d,p);assert np.isfinite(coef).all() and len(coef)==12
    torch.manual_seed(p['seed']);q=TimeModel();loss=(q(d['x'][:128])-d['y'][:128]).square().mean();loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in q.parameters())
    assert sum(v.numel() for v in q.parameters())==11393
    assert used(ROOT/'stage/runs')-used(OUT)+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,pairs=26395,parameters=11393,
        all_inputs_and_semantic_contract_locked=True,train_only_balanced_normalization_and_ridge=True,
        all_nested_entry_globals_resolve=True,finite_gradients=True,protocol_sha256=sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='PASS',pairs=26395,parameters=11393)),flush=True)


if __name__=='__main__':main()

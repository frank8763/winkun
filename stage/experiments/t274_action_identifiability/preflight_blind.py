"""Zero-optimizer matching and input intervention contracts."""
from blind_model import *
import importlib.util
import shutil
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t274_action_identifiability'


def main():
    assert not (OUT/'preflight.json').exists()
    p=json.loads((HERE/'protocol.json').read_text());parent=json.loads((ROOT/'stage/runs/t272_event_value_pilot/protocol.json').read_text())
    for path,h in p['input_hashes'].items():assert sha(path)==h,path
    for k in ['records','seed','optimizer_steps','batch_size','learning_rate','horizons_physics','horizons_seconds']:
        assert p[k]==parent[k],k
    d=dataset(p);torch.manual_seed(p['seed']);a=EventModel();torch.manual_seed(p['seed']);b=ActionBlind()
    for name,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[name])
    assert sum(v.numel() for v in a.parameters())==sum(v.numel() for v in b.parameters())==13202
    x=d['x'][:64].clone().requires_grad_(True);z=b(x)
    change=x.detach().clone();change[:,-2:]=torch.randn_like(change[:,-2:])*10
    assert torch.equal(z.detach(),b(change).detach())
    grad=torch.autograd.grad(z[:,:,0].sum(),x)[0];assert torch.equal(grad[:,-2:],torch.zeros_like(grad[:,-2:]))
    assert float(grad[:,:120].abs().max())>0
    spec=importlib.util.spec_from_file_location('t274_adapter',HERE/'run_blind.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    assert callable(mod.registered_training.main)
    assert used(ROOT/'stage/runs')-used(OUT)+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,parameters_matched=13202,
        initial_all_parameter_values_bit_exact_T272=True,all_data_split_optimizer_draw_schedule_hyperparameters_matched=True,
        current_action_counterfactual_invariance_bit_exact=True,current_action_input_derivative_exact_zero=True,
        original_120_context_still_used=True,original_trainer_source_locked=True,protocol_sha256=sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='PASS',only_current_action_masked=True)),flush=True)


if __name__=='__main__':main()

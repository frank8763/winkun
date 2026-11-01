"""Zero-native gradient, exact causal input, entry and whole-capacity checks."""
from t273_native import *
from proposals import propose
import builtins,dis,importlib.util,shutil,types
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t273_value_direction_probe'


def main():
    assert not (OUT/'preflight.json').exists()
    p=json.loads((HERE/'protocol.json').read_text())
    for path,h in p['input_hashes'].items():assert C.sha(path)==h,path
    spec=importlib.util.spec_from_file_location('t273_entry',HERE/'run.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    missing=set()
    def check(code):
        for i in dis.get_instructions(code):
            if i.opname=='LOAD_GLOBAL' and i.argval not in vars(mod) and i.argval not in vars(builtins):missing.add(i.argval)
        for c in code.co_consts:
            if isinstance(c,types.CodeType):check(c)
    check(mod.main.__code__);assert not missing,sorted(missing)
    actual=propose(p);assert actual==p['proposals']
    q=policy(p).eval().requires_grad_(False);assert S.digest(q.state_dict())==p['policy_hash']
    with torch.no_grad():a=q.step(torch.tensor([actual['context120']],dtype=torch.float32))[0][0].numpy()
    assert np.array_equal(a,np.array(actual['actions']['control'],dtype=np.float32))
    for key in ['up','down']:
        b=np.array(actual['actions'][key],dtype=np.float32);sigma=np.array([.035,.03])
        assert np.linalg.norm((b-a)/sigma)<=p['marginal_sigma_radius']+1e-5 and (np.abs(b)<1).all()
    assert 6*p['episode_bytes']+p['metadata_bytes']<=p['final_bytes']
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,
        original_Actor_action_and120_context_exact=True,finite_difference_relative_error=actual['finite_difference_relative_error'],
        one_precomputed_radius_no_search=True,strict_predicted_up_control_down=True,
        all_nested_entry_globals_resolve=True,full_output_bound=6*p['episode_bytes']+p['metadata_bytes'],
        native_restore_not_yet_executed=True,protocol_sha256=C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='PASS',predicted_score16=actual['score16'],actions=actual['actions'])),flush=True)


if __name__=='__main__':main()

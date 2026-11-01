"""Zero-native-step entry, actor, array and full-output budget checks."""
from pathlib import Path
import builtins
import dis
import importlib.util
import json
import sys
import types

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from t271_runtime import *
OUT=ROOT/'stage/runs/t271_frozen_feature_exploration'


def main():
    assert not (OUT/'preflight.json').exists()
    p=json.loads((HERE/'protocol.json').read_text())
    for path,h in p['input_hashes'].items():assert C.sha(path)==h,path
    spec=importlib.util.spec_from_file_location('stage271_entry',HERE/'run.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    missing=set()
    def check(code):
        for ins in dis.get_instructions(code):
            if ins.opname=='LOAD_GLOBAL' and ins.argval not in vars(mod) and ins.argval not in vars(builtins):missing.add(ins.argval)
        for co in code.co_consts:
            if isinstance(co,types.CodeType):check(co)
    check(mod.main.__code__);assert not missing,sorted(missing)
    assert json.loads(Path(p['density_contract']).read_text())['state']=='PASS'
    q=policy(p).eval().requires_grad_(False);assert S.digest(q.state_dict())==p['policy_hash']
    with np.load(p['real_source']) as z:
        for x,a in zip(z['states'][:64],z['actions'][:64]):
            with torch.no_grad():mode,mean,h,raw=q.step(torch.tensor(x[None]))
            assert np.array_equal(mode[0].numpy(),a)
            f=F.unit_feature(h);assert abs(float(f.square().sum())-1)<1e-12
    assert len(p['probe_jobs'])==32 and len(p['control_jobs'])==4 and len(p['geometry_jobs'])==4
    for k in range(0,32,2):
        a,b=p['probe_jobs'][k:k+2]
        assert a['source_reset_id']==b['source_reset_id'] and a['seed']==b['seed'] and a['case']==b['case']
        assert a['mode']=='conditional_AR' and b['mode']=='feature_copula'
    bound=16*p['full_episode_bytes']+20*p['compact_episode_bytes']+72*p['record_bytes']+p['metadata_bytes']
    assert bound<=p['final_bytes']
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,
        frozen_actor_first64_actual_actions_exact=True,unit_frozen_features=True,
        all_nested_entry_globals_resolve=True,matched16_pair_jobs=True,full_output_worst_case_bytes=bound,
        protocol_sha256=C.sha(HERE/'protocol.json'),native_geometry_and_deterministic_controls_not_yet_executed=True))
    print(json.dumps(dict(state='PASS',worst_case_bytes=bound)),flush=True)


if __name__=='__main__':main()

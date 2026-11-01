from t277_native import *
from t277_labels import first_hit_record,fixed_targets
import builtins,dis,importlib.util,json,shutil,types
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t277_physical_hindsight'


def main():
    assert not (OUT/'preflight.json').exists();p=json.loads((HERE/'protocol.json').read_text())
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    spec=importlib.util.spec_from_file_location('t277_entry',HERE/'run_recover.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    missing=set()
    def check(code):
        for i in dis.get_instructions(code):
            if i.opname=='LOAD_GLOBAL' and i.argval not in vars(mod) and i.argval not in vars(builtins):missing.add(i.argval)
        for co in code.co_consts:
            if isinstance(co,types.CodeType):check(co)
    check(mod.main.__code__);assert not missing,sorted(missing)
    total=0
    for r in p['references']:
        with np.load(r['path']) as z:
            assert all(S.digest(z[k])==h for k,h in r['array_hashes'].items())
            assert z['states'].shape[1]==120 and z['actions'].dtype==np.float32
            assert int(z['native_counts'].astype(np.int64).sum())==r['physics_steps'] and len(fixed_targets(z))==8
            assert np.array_equal(z['end_flags'][-1],np.array([r['collision'],r['success'],r['timeout']],dtype=np.uint8))
            total+=r['physics_steps']
    assert total==201980 and len(p['references'])==8
    # Native-step threshold and collision priority are checked independently on
    # synthetic geometry; no simulator constructed during preflight.
    xy=np.array([[.1,0],[.3,0],[.5,0],[.7,0],[.9,0]],dtype=np.float64);flag=np.zeros(5,dtype=np.uint8);ends=np.array([2,5])
    a=first_hit_record(xy,flag,np.zeros(2),xy[-1],ends,1)
    assert a['first_goal_step']==5 and a['terminal_hold_actual_steps']==3 and not a['inside_original_hold']
    xy2=np.array([[.1,0],[.3,0],[.5,0],[.71,0],[.9,0]],dtype=np.float64)
    b=first_hit_record(xy2,flag,np.zeros(2),xy2[-1],ends,1)
    assert b['first_goal_step']==4 and b['inside_original_hold'] and b['bootstrap_discount']==0
    bad=flag.copy();bad[-1]=1;assert not first_hit_record(xy2,bad,np.zeros(2),xy2[-1],ends,1)['eligible']
    assert not first_hit_record(xy2,flag,xy2[-1],xy2[-1],ends,1)['eligible']
    # A physics payload is written only AFTER exact original length/all8array
    # checks. Failed/inflight commands may spend the larger physical ceiling,
    # but cannot emit a longer accepted trajectory payload.
    assert p['expected_exact_physics_steps']==total
    worst=total*17+8*p['per_payload_overhead_bytes']+p['metadata_bytes'];assert worst<=p['final_bytes']
    # Both protocol copies, all per-case rows and64 bounded label records plus
    #64KiB remaining startup/status/preflight/log metadata are included.
    failed=sum(f.stat().st_size for d in OUT.glob('preflight_failed_*') for f in d.rglob('*') if f.is_file())
    meta=2*(HERE/'protocol.json').stat().st_size+8*4096+64*p['max_label_bytes']+65536+failed
    assert meta<=p['metadata_bytes'],meta
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,native_environments_created=0,
        all8_original_native_array_sha_verified=True,static_nested_entry_names_resolve=True,
        synthetic_collision_priority_initial_radius_and_insidehold_labels=True,
        worst_total_bytes=worst,worst_metadata_bytes=meta,protocol_sha256=C.sha(HERE/'protocol.json'),
        actual_native_reconstruction_not_yet_executed=True))
    print(json.dumps(dict(state='PASS',expected_physics=total,worst_bytes=worst,metadata_bound=meta)),flush=True)


if __name__=='__main__':main()

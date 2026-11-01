"""Zero-new-optimizer engineering resource repair; keep all scientific source."""
from pathlib import Path
import sys,copy,io,json,builtins,dis,importlib.util,types
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t287_time_aligned_credit'
sys.path.insert(0,str(HERE));from t287_credit import *
sys.path.insert(0,str(ROOT/'stage/experiments/t282_capped_balance'));from balanced_learning import *


def main():
    assert not any((OUT/n).exists() for n in ['preflight.json','launch.json','status.json','summary.json'])
    failed=HERE/'preflight_failed_001';old=json.loads((failed/'protocol.json').read_text())
    assert (HERE/'protocol.json').read_bytes()==(failed/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in old['input_hashes'].items():assert C.sha(f)==h,f
    for f in failed.glob('*.py'):assert f.read_bytes()==(HERE/f.name).read_bytes()
    calls=json.loads((OUT/'preflight_optimizer_calls.json').read_text())['calls']
    assert calls==['clone_live','clone_shadow','clone_live','clone_live']
    evidence=json.loads((OUT/'preflight_clone_events.json').read_text());events=evidence['events']
    live=[r for r in events if r['phase']=='live'];shadow=[r for r in events if r['phase']=='shadow']
    assert len(live)==len(shadow)==1
    assert {k:v for k,v in live[0].items() if k!='phase'}=={k:v for k,v in shadow[0].items() if k!='phase'}
    assert evidence['synthetic_acceptance_not_native20'] is True
    for n in ['preflight_optimizer_calls.json','preflight_clone_events.json','credit_contract.json']:assert (OUT/n).read_bytes()==(failed/n).read_bytes()
    p=copy.deepcopy(old);p.update(max_compact_payloads=1060,metadata_bytes=int(5.5*1024**2),payload_bytes=64*384*1024+1060*12288+8*1024**2)
    p['contract']=p['contract'].replace('1100compact12KiB','1060compact12KiB').replace('metadata5MiB','metadata5.5MiB')
    p['preflight_engineering_repair']=dict(old_failure=str(failed/'failure.json'),science_and_existing_py_byte_identical=True,
        compact_reserve_reduced_1100_to_1060=True,metadata_5_to_5point5_MiB=True,own_50MiB_and_all_physical_wall_scientific_limits_unchanged=True,
        previous_four_clone_checks_reused_exact_source=True,new_clone_optimizer_calls=0)
    p['input_hashes'].update({str(f):C.sha(f) for f in [Path(__file__),*failed.iterdir()] if f.is_file()})
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=policy(p);parent=copy.deepcopy(q).eval().requires_grad_(False)
    items=[]
    for r in p['historical_first16']:
        with np.load(r['path']) as z:a={k:z[k].copy() for k in z.files}
        assert C.sha(r['path'])==r['sha256'] and all(S.digest(v)==r['array_hashes'][k] for k,v in a.items());items.append(dict(row=r,arrays=a))
    data=L.training_batch(items,q);contract=enumeration_contract();assert contract==json.loads((OUT/'credit_contract.json').read_text())
    for filename in ['run.py','audit.py']:
        spec=importlib.util.spec_from_file_location('t287_final_'+filename[:-3],HERE/filename);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);missing=set()
        def walk(code):
            for ins in dis.get_instructions(code):
                if ins.opname=='LOAD_GLOBAL' and ins.argval not in vars(mod) and ins.argval not in vars(builtins):missing.add(ins.argval)
            for v in code.co_consts:
                if isinstance(v,types.CodeType):walk(v)
        walk(mod.main.__code__);assert not missing,(filename,missing)
    for item in items:
        a=item['arrays'];f=io.BytesIO();np.savez_compressed(f,**a);assert f.tell()<=p['training_episode_bytes']
        f=io.BytesIO();np.savez_compressed(f,**{k:a[k] for k in ['actions','native_counts','end_flags','rewards','environment_rewards','aux_costs']});assert f.tell()<=p['compact_episode_bytes']
    # At most 480 native guard + 48 centers + 329 development +94 gated final
    # +64 possible partial replays =1015 compact slots; 1060 includes margin.
    assert p['max_compact_payloads']>=480+48+329+94+64
    blob=json.dumps(p,indent=2)+'\n'
    upper=(64+1060)*2560+64*2560+17*32768+4*32768+4*2560+16*2560+4*65536+8*16384+65536+2*len(blob.encode())+S.used(OUT)
    assert upper<p['metadata_bytes'] and p['payload_bytes']+p['metadata_bytes']<p['final_bytes']
    outside=S.used(ROOT/'stage/runs')-S.used(OUT);assert outside+p['final_bytes']<p['global_bytes']
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,native_environments_created=0,navigation_Actor_updates=0,
        disposable_clone_optimizer_calls_total=4,new_finalize_optimizer_calls=0,prior_exact_clone_evidence_reused=True,
        clone_Actor_Adam_shadow_exact=True,rejection_and_interruption_Actor_Adam_restore_exact=True,
        fixed_round_balance=live[0]['event']['fixed_round_balance'] if 'event' in live[0] else live[0].get('fixed_round_balance'),
        canonical_density=data['density'],synthetic_old20_not_real_native=True,all_nested_run_and_audit_global_names_resolve=True,
        full_payload_and_metadata_reserved=p['payload_bytes']+p['metadata_bytes'],metadata_upper=upper,global_if_own_full=outside+p['final_bytes'],protocol_sha256=C.sha(HERE/'protocol.json'),
        independent_peer_baseline_enumeration=contract,engineering_repair=str(failed/'failure.json')))
    print(json.dumps(dict(state='FINAL_PREFLIGHT_PASS',new_optimizer_calls=0,total_prior_clones=4,metadata_upper=upper,global_if_own_full=outside+p['final_bytes'])),flush=True)


if __name__=='__main__':main()

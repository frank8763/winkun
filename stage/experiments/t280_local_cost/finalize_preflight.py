"""Finish storage validation without repeating the completed optimizer probes."""
from t280_core import *
import ast,builtins,dis,gzip,importlib.util,json,types
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t280_local_cost'


def main():
    assert not any((OUT/f).exists() for f in ['preflight.json','launch.json','status.json','summary.json'])
    plan=json.loads((HERE/'protocol.json').read_text());old=OUT/'preflight_failed_002';proof=json.loads((old/'preflight.json').read_text())
    assert proof['state']=='PASS' and proof['total_preflight_clone_optimizer_calls_including_preserved_attempts']==8
    # The learning, runtime, independent audit and deployment paths are identical
    # to those covered by the completed prior clone checks. Only storage setup changed.
    for name in ['run.py','t280_core.py','t280_worker.py','audit.py']:
        assert (HERE/name).read_bytes()==(old/'source'/name).read_bytes(),name
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    for file in HERE.glob('*.py'):ast.parse(file.read_text())
    for filename,entry in [('run.py','main'),('t280_worker.py','generate')]:
        spec=importlib.util.spec_from_file_location('final_'+filename.replace('.','_'),HERE/filename);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);missing=set()
        def check(code):
            for instruction in dis.get_instructions(code):
                if instruction.opname=='LOAD_GLOBAL' and instruction.argval not in vars(mod) and instruction.argval not in vars(builtins):missing.add(instruction.argval)
            for co in code.co_consts:
                if isinstance(co,types.CodeType):check(co)
        check(getattr(mod,entry).__code__);assert not missing,missing
    for record in plan['original_dependency_encodings']:
        assert C.sha(record['encoded_path'])==record['encoded_sha256']
        import hashlib
        assert hashlib.sha256(gzip.decompress(Path(record['encoded_path']).read_bytes())).hexdigest()==record['original_sha256']
    preserved=sum(f.stat().st_size for f in OUT.glob('preflight_failed_*/**/*') if f.is_file())
    upper=(plan['max_training_payloads']+plan['max_compact_payloads'])*2560+64*2560+17*32768+4*32768+4*2560+16*2560+4*65536+8*16384+65536+2*(HERE/'protocol.json').stat().st_size+preserved
    assert upper<plan['metadata_bytes'] and plan['payload_bytes']+plan['metadata_bytes']<plan['final_bytes']
    outside=S.used(ROOT/'stage/runs')-S.used(OUT);assert outside+plan['final_bytes']<plan['global_bytes']
    proof.update(protocol_sha256=C.sha(HERE/'protocol.json'),final_validation_physics_steps=0,final_validation_optimizer_calls=0,
        clone_checks_reused_from=str(old/'preflight.json'),clone_learning_runtime_audit_sources_byte_exact=True,
        confirmed_preflight_clone_calls=8,interrupted_final_preflight_unreceipted_additional_clone_calls_range=[0,4],
        total_preflight_clone_calls_upper_bound=12,additional_clones_not_repeated=True,
        metadata_worst_case_upper_bytes=upper,metadata_reserved_bytes=plan['metadata_bytes'],
        full_payload_and_metadata_reserved=plan['payload_bytes']+plan['metadata_bytes'],own_full_reserve_bytes=plan['final_bytes'],
        global_if_own_full_bytes=outside+plan['final_bytes'],
        final_validation='Completed prior Actor/Adam clone evidence reused after exact runtime/learning/audit-source comparison. Interrupted last tool produced no receipt; its optional0–4 extra clone calls not asserted exact. Storage/source/encoding checks completed with zero new optimizer calls.')
    C.save(OUT/'preflight.json',proof);print(json.dumps(dict(state='PASS',final_new_optimizer_calls=0,metadata_upper=upper,global_if_full=outside+plan['final_bytes'])),flush=True)


if __name__=='__main__':main()

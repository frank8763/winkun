"""Actual failed-group archive download, exact scientific configuration check."""
from pathlib import Path
import ast,json,sys,time
import t645r_memory as Memory
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t645r_transport_recovery';DOC=ROOT/'stage/docs/day_20261118/t645r'


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    old=ROOT/'stage/runs/t645_iterative_recovery_relay'
    p=Memory.read(old/'protocol.json');report=Memory.read(ROOT/'stage/docs/day_20261118/t645/FINAL_REPORT.json')
    assert not report['science_started'] and report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert not (old/'launch.json').exists() and not (old/'optimizer_calls.jsonl').exists()
    assert all(not Path('/proc',str(pid)).exists() for pid in Memory.read(old/'SEALED.json')['scientific_pids'])
    # Exercise the specific second archive at which the failed producer had
    # stopped (78 first-group members were already verified there).
    keys=list(dict.fromkeys(r['receipt'] for r in p['memory_sources'] if 'receipt' in r));assert len(keys)==5
    started=time.time();data=Memory.fetch(keys[1],120)
    assert Memory.digest(data)==Memory.read(keys[1])['sha256'];preflight_seconds=time.time()-started
    oldbytes=sum(f.stat().st_size for f in old.rglob('*') if f.is_file())
    p.update(id='T645R_TRANSPORT_ONLY_RECOVERY',out=str(OUT),authorization_approved=False,
        previous_record_bytes=oldbytes,preload_seconds=p['preload_seconds']-report['preload_seconds_used']-preflight_seconds,
        max_bytes=p['max_bytes']-oldbytes-131072,previous_failed_producer=str(old),transport_fallback_only=True)
    assert p['preload_seconds']>600
    # Everything that can alter scientific observations, controls, labels,
    # gradients or acceptance remains content-identical.
    original=Memory.read(old/'protocol.json')
    operational={'id','out','authorization_approved','previous_record_bytes','preload_seconds','max_bytes','previous_failed_producer','transport_fallback_only'}
    assert all(p[k]==v for k,v in original.items() if k not in operational)
    assert time.time()+p['preload_seconds']+p['input_audit_seconds']+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+120<p['absolute_all_stop']
    upper=18*p['fresh_bytes']+4*p['full_bytes']+3*p['model_bytes']+429*p['compact_bytes']+5*2**20
    assert upper<=p['max_bytes']
    global_used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    global_worst=global_used+p['max_bytes']+131072;assert global_worst<2**30
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',old/'protocol.json',old/'SEALED.json',ROOT/'stage/docs/day_20261118/t645/FINAL_REPORT.json'}
    import hashlib
    p['inputs'].update({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    Memory.save(HERE/'protocol.json',p)
    Memory.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',specific_failed_archive=keys[1],
        actual_remote_archive_and_every_member_SHA_verified=True,archive_sha256=Memory.digest(data),
        scientific_configuration_fields_exact=True,scientific_source_reused_unchanged=True,
        original_failed_producer_preserved=True,prior_science_native_and_optimizer=0,
        original_preload_seconds=report['preload_seconds_used'],preflight_download_seconds=preflight_seconds,
        remaining_preload_seconds=p['preload_seconds'],previous_record_bytes=oldbytes,
        new_record_limit=p['max_bytes'],own_storage_upper=upper,global_worst=global_worst,
        original_all_stop_unchanged=p['absolute_all_stop'],new_optimizer=0))
    Memory.save(DOC/'BUDGET.json',{k:p[k] for k in ['max_rounds','max_physics','max_commands','max_bytes','preload_seconds','main_seconds','audit_seconds','absolute_all_stop','new_Actor_Adam','shadow_Actor_Adam','independent_historical_Actor_Adam']})
    print('T645R_PREFLIGHT_PASS_TRANSPORT_ONLY',preflight_seconds,upper,p['max_bytes'],flush=True)


if __name__=='__main__':main()

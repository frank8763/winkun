"""Transport-independent reversible log reserve, zero native/optimizer."""
from pathlib import Path
import hashlib,json,os,sys,time
import t483_storage as S
ROOT=S.ROOT;OUT=S.OUT;DOC=ROOT/'stage/docs/review_20261111'
def read(f):return json.loads(Path(f).read_text())
def save(f,d):Path(f).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def prepare():
    assert not OUT.exists();S.idle()
    old=read(ROOT/'stage/runs/t481_fresh16_rehearsal/protocol.json')
    assert not any((ROOT/'stage/runs/t481_fresh16_rehearsal'/f).exists() for f in ['cache_remote_verified.json','cache_summary.json','launch.json','preflight.json'])
    files=set(old['inputs'])|set(old['constructor_reads'])|set(old['protected_files'])
    files|={str(f) for f in Path(__file__).resolve().parent.glob('*.py')}
    files.add(str(ROOT/'stage/experiments/t482_evening_coverage/t482_cache.py'))
    needed=max(1,S.used()+old['max_bytes']+3*2**20-2**30)
    enc=S.plan(files,needed);assert enc['admitted']
    for row in enc['rows']:
        for key in ['receipt','manifest']:files.add(str(ROOT/row[key]))
    p=dict(id='T483_LOCAL_REVERSIBLE_LOG_RESERVE',max_bytes=2**20,inputs={f:sha(f) for f in sorted(files)},constructor_reads=sorted(files),protected_files=old['protected_files'],storage_rows=enc['rows'],storage_expected_release=enc['selected_net_release_bytes'],deadline=time.time()+600,native=0,optimizer=0)
    OUT.mkdir();save(OUT/'protocol.json',p)
    save(DOC/'T483_STORAGE_PLAN.json',dict(state='REGISTERED_NOT_EXECUTED',required=needed,selected_release=enc['selected_net_release_bytes'],files=len(enc['rows']),native=0,optimizer=0,original_bytes_local_and_remote=True,all_models_and_unique_evidence_preserved=True,wall_seconds=600,max_bytes=2**20))
def execute():
    p=read(OUT/'protocol.json');assert time.time()+300<p['deadline'];S.execute(p)
def audit():
    p=read(OUT/'protocol.json');assert time.time()<p['deadline'];S.audit(p)
    old=read(ROOT/'stage/runs/t481_fresh16_rehearsal/protocol.json')
    assert S.used()-sum(f.stat().st_size for f in (ROOT/'stage/runs/t481_fresh16_rehearsal').rglob('*') if f.is_file())+old['max_bytes']<=2**30
    for f,h in old['inputs'].items():assert sha(f)==h
    # Correctly states zero remote evictions; the failed plans remain intact.
    target=ROOT/'stage/runs/t481_fresh16_rehearsal/cache_audit.json';assert not target.exists()
    save(target,dict(state='PASS_REFERENCE_RECEIPT_MARKERS_AND_PROTECTED_INPUTS',files=0,native=0,optimizer=0,
        alternate_local_storage_admission=True,original_remote_cache_plan_NOT_executed=True,no_remote_cache_unlink=True,
        local_original_logs_preserved_losslessly=True,storage_run='T483',storage_audit_sha256=sha(OUT/'compression_audit.json'),
        full_T481_root_reserved=True,all_registered_T481_input_SHA_unchanged=True))
    save(DOC/'T481_ALTERNATE_STORAGE_ADMISSION.json',dict(state='PASS_LOCAL_LOSSLESS_ALTERNATIVE_ZERO_REMOTE_CACHE_EVICTIONS',native=0,optimizer=0,
        registered_scientific_protocol_unchanged=True,original_failed_preparation_preserved=True,cache_plan_unexecuted=True,
        storage_audit=read(OUT/'compression_audit.json'),original_T481_protocol_sha256=sha(ROOT/'stage/runs/t481_fresh16_rehearsal/protocol.json')))
    save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),native=0,optimizer=0,files=len(p['storage_rows']),audit='PASS',all_original_bytes_local_and_remote=True))
if __name__=='__main__':globals()[sys.argv[1]]()

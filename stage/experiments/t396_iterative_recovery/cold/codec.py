"""Fixed six ended actors, audited lossless delta storage, never new training."""
from pathlib import Path
import importlib.util,sys,signal
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('stage396_fixed_byte_codec',ROOT/'stage/experiments/t321_cold_checkpoint_reserve/codec.py')
C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
OUT=ROOT/'stage/runs/t396_cold_reserve';DOC=ROOT/'stage/docs/review_20261106'
C.HERE,C.OUT,C.DOC,C.LEDGER=HERE,OUT,DOC,DOC/'T396_COLD_ENCODING_LEDGER.json'
C.REF=ROOT/'stage/runs/t211_guarded_progress/models/original_round0.pt.gz'
FIXED=['t228_balanced_stability','t231_retained_behavior_repair','t239_ar_policy_gradient','t240_retention_ablation','t243_induced_retention_repair','t245_multitask_pcgrad']
def prepare():
    assert not OUT.exists() and not C.LEDGER.exists();ref=C.REF.read_bytes();keep=C.protected()
    safe=ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz';keep[str(safe.relative_to(ROOT))]=C.sha(safe.read_bytes())
    for f,h in C.read(ROOT/'stage/runs/t395_frontier_recovery/protocol.json')['inputs'].items():keep[str(Path(f).relative_to(ROOT))]=h
    for n in ['t391_actual_student_recovery','t393_safe_demo_pg','t394_takeover_diagnostic','t395_frontier_recovery']:
        for f in (ROOT/'stage/runs'/n).rglob('*'):
            if f.is_file():keep[str(f.relative_to(ROOT))]=C.sha(f.read_bytes())
    records=[]
    for folder in FIXED:
        f=ROOT/'stage/runs'/folder/'final_Actor.pt.gz';b=f.read_bytes();rel=str(f.relative_to(ROOT));assert rel not in keep
        r=dict(original_path=rel,original_bytes=len(b),original_sha256=C.sha(b),encoded_path=rel+'.t396.dxz',prior_receipt='stage/docs/long_distance_20261101/release_receipt_T396-COLD-ORIGINALS.json',prior_manifest='stage/docs/long_distance_20261101/source_manifest_T396-COLD-ORIGINALS.json')
        C.verify_prior(r);z=C.encode(b,ref);assert C.decode(z,ref)==b
        r.update(encoded_bytes=len(z),encoded_sha256=C.sha(z));records.append(r)
    gain=sum(r['original_bytes']-r['encoded_bytes'] for r in records);assert gain>=4*1024**2
    assert C.used()+256*1024<1024**3
    p=dict(experiment='T396_COLD_FIXED6',records=records,protected_hashes=keep,reference=str(C.REF.relative_to(ROOT)),reference_sha256=C.sha(ref),source_hashes={str(f.relative_to(ROOT)):C.sha(f.read_bytes()) for f in [Path(__file__),Path(C.__file__),DOC/'T396_COLD_PLAN.md']},wall_seconds=600,audit_seconds=300,own_metadata_bytes=256*1024,global_bytes=1024**3,min_free_bytes=40*1024**3,expected_released_bytes=gain,physics_steps=0,optimizer_calls=0)
    OUT.mkdir();C.write(HERE/'protocol.json',p);C.write(OUT/'protocol.json',p);C.write(OUT/'preflight.json',dict(state='PASS',files=6,modified=0,original_gzip_raw_zip_exact=True,prior_full_remote_readback=True,released_payload_bytes=gain));print(gain)
def launch():
    assert not (OUT/'launch.json').exists()
    assert not C.subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    r=C.read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert r['root_tree']==C.subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (OUT/'run.log').open('xb') as log:q=C.subprocess.Popen(['timeout','--kill-after=1','600','taskset','-c','0',str(ROOT/'run_stage.sh'),str(Path(__file__)),'run'],cwd=ROOT,stdout=log,stderr=C.subprocess.STDOUT,start_new_session=True)
    C.write(OUT/'launch.json',dict(pid=q.pid,epoch=C.time.time(),source_remote=r['remote_commit']));print(q.pid)
def run():C.run()
def audit():
    p=C.read(HERE/'protocol.json');s=C.read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (DOC/'T396_COLD_AUDIT.json').exists()
    for pid in [s['pid'],C.read(OUT/'launch.json')['pid']]:assert not Path('/proc',str(pid)).exists()
    ref=C.REF.read_bytes()
    for r in p['records']:
        C.verify_prior(r);assert not (ROOT/r['original_path']).exists();z=(ROOT/r['encoded_path']).read_bytes();assert C.sha(z)==r['encoded_sha256'];b=C.decode(z,ref);assert len(b)==r['original_bytes'] and C.sha(b)==r['original_sha256']
    for name,h in {**p['source_hashes'],**p['protected_hashes']}.items():assert C.sha((ROOT/name).read_bytes())==h,name
    assert C.used()<=1024**3
    a=dict(state='PASS',files=6,raw_zip_and_original_gzip_bytes_exact=True,protected_inputs_unchanged=True,full_remote_original_readback=True,new_encoded_binaries_local_only=True,native=0,optimizer=0,released_payload_bytes=s['payload_released_bytes'],global_bytes=C.used())
    C.write(DOC/'T396_COLD_AUDIT.json',a);C.write(OUT/'audit.json',a);print('T396_COLD_AUDIT_PASS')
load_original=C.load_original
if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T396 cold bounded')));signal.alarm(600)
    assert sys.argv[1] in ['prepare','launch','run','audit'];globals()[sys.argv[1]]()

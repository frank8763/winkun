"""Reuse the audited T321 byte-exact codec for a fixed, ended cold set."""
from pathlib import Path
import sys, signal
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t321_cold_checkpoint_reserve'))
import codec as C
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t350_cold_reserve';DOC=ROOT/'stage/docs/review_20261104'
C.HERE,C.OUT,C.DOC,C.LEDGER=HERE,OUT,DOC,DOC/'T350_ENCODING_LEDGER.json'
GROUPS={'T283':['t283_capped_task_balance'],'T284':['t284_feedback_drift'],
        'T285':['t285_trajectory_constraints'],'T287':['t287_time_aligned_credit'],
        'T289':['t289_rejected_candidates'],'T322':['t322_full_network_steering']}


def prepare():
    assert not OUT.exists() and not C.LEDGER.exists();start=C.time.time();ref=C.REF.read_bytes()
    assert C.sha(ref)=='63e63e2b444867b26622a2ab0f0c4227bc38d48a7cd2b5a3b378d9b0266006df'
    protected=C.protected()
    for f in C.read(ROOT/'stage/runs/t349_consistent_demo/protocol.json')['input_hashes']:
        if Path(f).is_file():protected[str(Path(f).relative_to(ROOT))]=C.sha(Path(f).read_bytes())
    for name in ['t255_overnight','t328_wide_gap_80s','t331_policy_prefix_retry','t340_demo_supported_recovery',
                 't342_balanced_consolidation','t347_original_start_human','t348_causal_sequence_warmstart','t349_consistent_demo']:
        for f in (ROOT/'stage/runs'/name).rglob('*'):
            if f.is_file():protected[str(f.relative_to(ROOT))]=C.sha(f.read_bytes())
    records=[];skipped=[]
    for group,folders in GROUPS.items():
        for folder in folders:
            d=ROOT/'stage/runs'/folder;s=C.read(d/'summary.json');assert s['state'] in ['COMPLETE','FAILED'] and s['all_workers_stopped']
            for f in sorted(d.glob('*.pt.gz')):
                assert C.time.time()-start<240
                relative=str(f.relative_to(ROOT));assert relative not in protected
                b=f.read_bytes();r=dict(original_path=relative,original_bytes=len(b),original_sha256=C.sha(b),
                    encoded_path=relative+'.dxz',prior_receipt=f'stage/docs/long_distance_20261101/release_receipt_{group}.json',
                    prior_manifest=f'stage/docs/long_distance_20261101/source_manifest_{group}.json')
                C.verify_prior(r);z=C.encode(b,ref);assert C.decode(z,ref)==b
                if len(z)>=len(b):skipped.append(relative);continue
                r.update(encoded_bytes=len(z),encoded_sha256=C.sha(z));records.append(r)
    saving=sum(r['original_bytes']-r['encoded_bytes'] for r in records)
    assert saving>=14*1024**2,('Insufficient reserve; zero file mutation',saving)
    assert C.used()+512*1024<1024**3 and C.shutil.disk_usage(ROOT).free>=40*1024**3
    p=dict(experiment='T350_FIXED_COLD_BYTE_EXACT_RESERVE',records=records,protected_hashes=protected,
        reference=str(C.REF.relative_to(ROOT)),reference_sha256=C.sha(ref),
        source_hashes={str(f.relative_to(ROOT)):C.sha(f.read_bytes()) for f in [Path(__file__),Path(C.__file__),DOC/'T350_PLAN.md']},
        wall_seconds=600,audit_seconds=300,own_metadata_bytes=512*1024,global_bytes=1024**3,min_free_bytes=40*1024**3,
        expected_released_bytes=saving,physics_steps=0,optimizer_calls=0,Actor_updates=0,skipped_no_gain=skipped)
    OUT.mkdir();C.write(HERE/'protocol.json',p);C.write(OUT/'protocol.json',p)
    C.write(OUT/'preflight.json',dict(state='PASS',files=len(records),files_modified=0,physics_steps=0,optimizer_calls=0,
        all_original_gzip_and_raw_zip_bytes_exact=True,prior_remote_every_member_SHA_verified=True,expected_released_bytes=saving))
    print(C.json.dumps(dict(state='PREFLIGHT_PASS',files=len(records),saving=saving)),flush=True)


def stopped():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==C.os.getpid():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if args and Path(C.os.fsdecode(args[0])).name.startswith('python'):
            assert not any(str(ROOT/'stage/experiments') in C.os.fsdecode(a) for a in args[1:]),('Other Stage Python',proc.name)


def run():
    stopped();C.run()


def launch():
    assert not any((OUT/f).exists() for f in ['launch.json','status.json','summary.json'])
    assert not C.subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=C.read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==C.subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (OUT/'run.log').open('xb') as f:
        proc=C.subprocess.Popen(['timeout','--kill-after=1s','600s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(Path(__file__)),'run'],cwd=ROOT,stdout=f,stderr=C.subprocess.STDOUT,start_new_session=True)
    C.write(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=C.time.time(),remote_commit=receipt['remote_commit'],single_launch=True));print(proc.pid)


def audit():
    p=C.read(HERE/'protocol.json');s=C.read(OUT/'summary.json');dest=DOC/'T350_AUDIT.json';assert s['state']=='COMPLETE' and not dest.exists()
    for pid in [s['pid'],C.read(OUT/'launch.json')['pid']]:assert not Path('/proc',str(pid)).exists()
    start=C.time.time();ref=C.REF.read_bytes()
    for r in p['records']:
        assert C.time.time()-start<300;C.verify_prior(r);assert not (ROOT/r['original_path']).exists()
        z=(ROOT/r['encoded_path']).read_bytes();assert C.sha(z)==r['encoded_sha256'];b=C.decode(z,ref)
        assert len(b)==r['original_bytes'] and C.sha(b)==r['original_sha256']
    for name,h in {**p['source_hashes'],**p['protected_hashes']}.items():assert C.sha((ROOT/name).read_bytes())==h,name
    assert C.used()<1024**3 and C.shutil.disk_usage(ROOT).free>=40*1024**3
    C.write(dest,dict(state='PASS',files=len(p['records']),original_gzip_and_raw_zip_bytes_exact=True,
        protected_current_inputs_unchanged=True,prior_full_remote_member_SHA_receipts_verified=True,
        new_encoded_binaries_not_uploaded=True,physics_steps=0,optimizer_calls=0,global_bytes=C.used()))
    print('T350 independent byte audit PASS')


load_original=C.load_original
if __name__=='__main__':
    parser=C.argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','launch','run','audit','restore']);parser.add_argument('path',nargs='?');a=parser.parse_args()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T350 bounded codec')));signal.alarm(600)
    if a.mode=='restore':C.restore(a.path)
    else:globals()[a.mode]()

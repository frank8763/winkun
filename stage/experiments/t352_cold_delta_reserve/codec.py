"""Reuse the audited T321 byte-exact codec for a fixed, ended cold set."""
from pathlib import Path
import sys, signal
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t321_cold_checkpoint_reserve'))
import codec as C
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t352_cold_delta_reserve';DOC=ROOT/'stage/docs/review_20261104'
C.REF=ROOT/'stage/runs/t211_guarded_progress/models/original_round0.pt.gz'
C.HERE,C.OUT,C.DOC,C.LEDGER=HERE,OUT,DOC,DOC/'T352_ENCODING_LEDGER.json'
GROUPS={'T283':['t283_capped_task_balance'],'T284':['t284_feedback_drift'],
        'T285':['t285_trajectory_constraints'],'T287':['t287_time_aligned_credit'],
        'T289':['t289_rejected_candidates'],'T322':['t322_full_network_steering']}


def verify_prior(r):
    old=C.read(ROOT/r['prior_receipt']);assert old['full_remote_original_binary_and_all_members_verified']
    item=next(v for v in old['records'] if v['encoded_path']==r['original_path'])
    assert item['encoded_sha256']==r['original_sha256'] and item['encoded_bytes']==r['original_bytes']
    rec=C.read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T286.json')
    mf=(ROOT/r['prior_manifest']).read_bytes();assert rec['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert rec['manifest_sha256']==C.sha(mf)
    raw=next(v for v in C.json.loads(mf)['files'] if v['path']==r['old_raw_path'])
    assert raw['sha256']==r['old_raw_sha256']==item['original_sha256']
C.verify_prior=verify_prior

def prepare():
    assert not OUT.exists() and not C.LEDGER.exists();start=C.time.time();ref=C.REF.read_bytes()
    assert C.sha(ref)==next(v['encoded_sha256'] for v in C.read(ROOT/'stage/docs/long_distance_20261101/T286_CHECKPOINT_ENCODING_RECEIPT.json')['records'] if v['encoded_path']==str(C.REF.relative_to(ROOT)))
    protected=C.protected()
    for f in C.read(ROOT/'stage/runs/t351_recurrent_recovery/protocol.json')['input_hashes']:
        if Path(f).is_file():protected[str(Path(f).relative_to(ROOT))]=C.sha(Path(f).read_bytes())
    for name in ['t255_overnight','t328_wide_gap_80s','t331_policy_prefix_retry','t340_demo_supported_recovery',
                 't342_balanced_consolidation','t347_original_start_human','t348_causal_sequence_warmstart','t349_consistent_demo','t351_recurrent_recovery']:
        for f in (ROOT/'stage/runs'/name).rglob('*'):
            if f.is_file():protected[str(f.relative_to(ROOT))]=C.sha(f.read_bytes())
    records=[];skipped=[]
    receipt=ROOT/'stage/docs/long_distance_20261101/T286_CHECKPOINT_ENCODING_RECEIPT.json'
    prior=C.read(receipt)
    for old in prior['records']:
        f=ROOT/old['encoded_path'];relative=str(f.relative_to(ROOT))
        if not f.is_file() or f.stat().st_size<1000000 or relative in protected:continue
        assert C.time.time()-start<240 and not any(v in relative for v in ['t194_','t255_','t299_','t288_'])
        b=f.read_bytes();assert C.sha(b)==old['encoded_sha256'] and len(b)==old['encoded_bytes']
        r=dict(original_path=relative,original_bytes=len(b),original_sha256=C.sha(b),encoded_path=relative+'.dxz',
            prior_receipt=str(receipt.relative_to(ROOT)),prior_manifest='stage/docs/long_distance_20261101/source_manifest_T286.json',
            old_raw_path=old['original_path'],old_raw_sha256=old['original_sha256'])
        verify_prior(r);z=C.encode(b,ref);assert C.decode(z,ref)==b
        if len(z)>=len(b):skipped.append(relative);continue
        r.update(encoded_bytes=len(z),encoded_sha256=C.sha(z));records.append(r)
    saving=sum(r['original_bytes']-r['encoded_bytes'] for r in records)
    assert saving>=16*1024**2,('Insufficient reserve; zero file mutation',saving)
    assert C.used()+512*1024<1024**3 and C.shutil.disk_usage(ROOT).free>=40*1024**3
    p=dict(experiment='T352_FIXED_OLD_GZIP_DELTA_RESERVE',records=records,protected_hashes=protected,
        reference=str(C.REF.relative_to(ROOT)),reference_sha256=C.sha(ref),
        source_hashes={str(f.relative_to(ROOT)):C.sha(f.read_bytes()) for f in [Path(__file__),Path(C.__file__),DOC/'T352_PLAN.md']},
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
    p=C.read(HERE/'protocol.json');s=C.read(OUT/'summary.json');dest=DOC/'T352_AUDIT.json';assert s['state']=='COMPLETE' and not dest.exists()
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
    print('T352 independent byte audit PASS')


load_original=C.load_original
if __name__=='__main__':
    parser=C.argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','launch','run','audit','restore']);parser.add_argument('path',nargs='?');a=parser.parse_args()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T352 bounded codec')));signal.alarm(600)
    if a.mode=='restore':C.restore(a.path)
    else:globals()[a.mode]()

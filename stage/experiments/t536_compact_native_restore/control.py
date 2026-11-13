"""Prepare compact native restore only with a separately recorded live window."""
from pathlib import Path
import datetime,gzip,json,os,pickle,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t536_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha
AUTH=DOC/'T536_WINDOW_AUTHORIZATION.json'

def engineering_check():
    """No scene, native call, Actor optimizer, registration or launch."""
    prior=ROOT/'stage/runs/t535_minimal_snapshot';p=read(prior/'protocol.json');rows=[]
    assert not OUT.exists(),'Engineering checks must precede registration'
    # Check the actual failure-producing old disk artifact at its preserved
    # hash, including all history and RNG state that previously broke equality.
    for row in p['source_snapshots']:
        assert sha(row['path'])==row['sha256'];original=R.Compact.unpack(Path(row['path']).read_bytes())
        reduced=R.Compact.compact(original);blob=R.Compact.pack(reduced)
        restored=R.Compact.unpack(blob);R.Compact.verify_preserved(original,restored,R.state_equal)
        assert len(blob)<=16*1024
        corrupt=R.Compact.unpack(blob);corrupt['hidden'][0,0,0]+=1.
        try:R.Compact.verify_preserved(original,corrupt,R.state_equal)
        except AssertionError:pass
        else:raise AssertionError('Comparator accepted a corrupt recurrent hidden state')
        corrupt=R.Compact.unpack(blob);corrupt['native']['attrs']['_np_random'].normal()
        try:R.Compact.verify_preserved(original,corrupt,R.state_equal)
        except AssertionError:pass
        else:raise AssertionError('Comparator accepted a changed RNG state')
        assert len(restored['rows'])==len(restored['qstate'][2])==0 and restored['steps']==2400
        rows.append(dict(source_path=row['path'],source_sha256=row['sha256'],compact_bytes=len(blob),nonlogging_state_exact=True,hidden_and_RNG_corruption_rejected=True))
    result=dict(state='PASS_EXISTING_SNAPSHOT_ENGINEERING_ONLY',pid=os.getpid(),rows=rows,new_native=0,new_optimizer=0,actual_compact_native_restore_NOT_tested=True,registration_NOT_created=True,window_authorization_pending=True,source_sha256={str(f):sha(f) for f in HERE.glob('*.py')},epoch=time.time())
    f=DOC/'T536_ENGINEERING_CHECK.json';assert not f.exists();save(f,result)
    print('T536_ENGINEERING_CHECK_PASS_NO_NATIVE_OR_LAUNCH',flush=True)

def prepare():
    assert not OUT.exists();auth=read(AUTH);assert auth['user_confirmed'] and auth['scope']=='T536_ONLY_73000_PHYSICS_1MiB_ZERO_ACTOR'
    end=auth['absolute_all_stop'];assert time.time()+1080<end,'Confirmed window expired or insufficient'
    prior=ROOT/'stage/runs/t534_native_state_restore';base=read(prior/'protocol.json')
    assert read(prior/'audit.json')['state']=='PASS' and read(prior/'summary.json')['state']=='FAILED'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids'])
    for n in ['T534','T535']:assert read(ROOT/('stage/docs/long_distance_20261101/release_receipt_'+n+'.json'))['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=dict(base)
    p.update(id='T536_COMPACT_NATIVE_STATE_RESTORE',max_bytes=2**20,snapshot_bytes=32*1024,suffix_bytes=64*1024,main_seconds=480,audit_seconds=180,transport_seconds=240,source_seconds=120,absolute_all_stop=end,absolute_native_stop=end-900,absolute_learning_stop=end-1800,scope='Explicit repair of historical logging growth; same four frozen windows. Compact persisted snapshot retains all causal state, clears log-only buffers. 73000 new native physics and9commands,0Actor/Value/direction/update. All suffix arrays/integration must be bitwise equal. No extra rounds or automatic follow-on.',authorization=auth)
    assert time.time()+480<p['absolute_native_stop']
    # Keep original Stage dependency hashes, add the repaired implementation.
    for f in [*HERE.glob('*.py'),ROOT/'stage/experiments/t535_minimal_snapshot/compact_snapshot.py',DOC/'T536_PLAN.md',DOC/'T536_ENGINEERING_CHECK.json',AUTH,prior/'summary.json',prior/'audit.json']:p['inputs'][str(f)]=sha(f)
    p['prior_T534_physics_not_erased']=38900;p['mechanism_cumulative_physics_max']=111900
    assert R.used(ROOT/'stage/runs')+p['max_bytes']+65536<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
    save(DOC/'T536_BUDGET.json',dict(new_physics=73000,prior_T534_physics=38900,cumulative_if_complete=111900,commands=9,own_max_bytes=2**20,global_reserved=R.used(ROOT/'stage/runs')-R.used(OUT)+2**20,main_seconds=480,audit_seconds=180,backup_seconds=240,source_seconds=120,admission_seconds=1080,ActorAdam=0,ValueAdam=0,absolute_all_stop=end,absolute_native_stop=p['absolute_native_stop']))
    print('T536_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists()
    # The engineering witness must match the registered code files.
    checks=read(DOC/'T536_ENGINEERING_CHECK.json')
    assert checks['state']=='PASS_EXISTING_SNAPSHOT_ENGINEERING_ONLY'
    for f,h in checks['source_sha256'].items():assert sha(f)==h,f
    assert sha(AUTH)==p['inputs'][str(AUTH)]
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,compact_existing_snapshots_qualified=True,actual_native_restore_NOT_tested=True,engineering_witness_sha256=sha(DOC/'T536_ENGINEERING_CHECK.json')))
    print('T536_PREFLIGHT_PASS_NOT_LAUNCHED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

"""Register exact native state restoration; zero-native synthetic preflight."""
from pathlib import Path
import gzip,json,os,pickle,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t534_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t533_causal_vehicle_identification'
    assert read(prior/'audit.json')['state']=='PASS'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T533.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    base=read(ROOT/'stage/runs/t532_physics_forward_contract/protocol.json');end=base['absolute_all_stop'];assert time.time()+1080<end
    p=dict(id='T534_PERSISTED_NATIVE_STATE_RESTORE',source_protocol=read(base['actor_configuration'])['source_protocol'],actor_configuration=base['actor_configuration'],windows=base['windows'],holds=5,max_bytes=int(2.5*2**20),snapshot_bytes=384*1024,suffix_bytes=64*1024,main_seconds=480,audit_seconds=180,transport_seconds=240,source_seconds=120,absolute_all_stop=end,absolute_native_stop=end-900,absolute_learning_stop=end-1800,CPUQuota='100%',GPU=False,max_physics=sum((w['first']+10)*100 for w in base['windows']),max_commands=9,scope='Four fixed existing old/new success/failure windows; real native prefix then continuous5holds and persisted snapshot restore5holds. True unchanged Actor64carry/current120. All source prefix and suffix exact. No teacher/parameter search/optimizer/navigation gate or automatically following experiment.')
    assert time.time()+480<p['absolute_native_stop']
    p['inputs']=dict(base['inputs'])
    for f in [*HERE.glob('*.py'),DOC/'T534_PLAN.md',prior/'audit.json',prior/'summary.json']:p['inputs'][str(f)]=sha(f)
    p['protected_files']={w['row']['request']['model']:sha(w['row']['request']['model']) for w in p['windows']}
    assert R.used(ROOT/'stage/runs')+p['max_bytes']+65536<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
    save(DOC/'T534_BUDGET.json',dict(physics=p['max_physics'],commands=9,main_seconds=480,audit_seconds=180,backup_seconds=240,source_seconds=120,max_native_seconds=4*1+sum(w['first']*.2 for w in p['windows'])+4*1,own_max_bytes=p['max_bytes'],global_reserved=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes'],ActorAdam=0,ValueAdam=0,absolute_native_stop=p['absolute_native_stop'],absolute_all_stop=end))
    print('T534_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists()
    # Serialization is tested without creating a native scene or stepping it.
    a=dict(x=R.np.arange(20),h=R.torch.zeros((1,1,64)),frames=[R.np.zeros(12)],clock=4.8)
    b=pickle.loads(gzip.decompress(gzip.compress(pickle.dumps(a,protocol=5),mtime=0)));b['x'][0]=99
    assert a['x'][0]==0 and R.torch.equal(a['h'],b['h'])
    assert R.F.FieldSession.capture is not None and R.F.FieldSession.restore is not None
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,synthetic_disk_serialization_fixture=True,real_state_restore_NOT_tested=True,software_renderer_will_be_asserted_at_native_setup=True));print('T534_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

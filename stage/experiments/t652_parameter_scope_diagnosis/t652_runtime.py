"""One matched learning candidate per arm, field24 then conditional unique407."""
from pathlib import Path
import copy,fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t652_adapters_pg';DOC=ROOT/'stage/docs/day_20261119/t652'
sys.path.insert(0,str(ROOT/'stage/experiments/t646_offline_night'))
import t646_runtime as Shared
import t646_coverage as Coverage
import t652_learning as M
import t651_runtime as DataSource
import t651_native as Native
O,R=Shared.O,Shared.R
read,save,sha,digest,torch,np=Shared.read,Shared.save,Shared.sha,Shared.digest,Shared.torch,Shared.np


def rootplan():return read(HERE/'protocol.json')
def plan():
    root=rootplan();name=sys.argv[2];assert name in root['arms']+['reconstruction']
    child=read(OUT/name/'protocol.json');assert child['_root_sha256']==sha(OUT/'protocol.json')
    return dict(root,**child,out=str(OUT/name))


def verify(p,*args):
    root=rootplan();assert root['authorization_approved'] and time.time()<root['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert R.used(OUT)<=root['max_bytes'] and R.used(Path(p['out']))<=(root['reconstruction_bytes'] if p['arm']=='reconstruction' else root['child_bytes'])
    assert R.used(ROOT/'stage/runs')-R.used(OUT)+root['max_bytes']<root['global_bytes']
    assert R.D.B.shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in root['inputs'].items():assert sha(f)==h,f


def safe(row):
    boundary_applies='fullfield_start_pose' in row['request']['job']
    return bool(row['complete'] and row['success'] and not row['collision'] and (not boundary_applies or not row.get('field_boundary',False)))


def setup(p):
    folder=Path(p['out']);O.OUT=folder;O.DOC=DOC/p['arm'];R.OUT=folder;R.DOC=DOC/p['arm'];R.HERE=HERE
    R.verify=verify;R.M=M;O.M=M;R.setup(p);R.N.worker=Native.worker;R.N.inspect=O.N.inspect
    R.safe=safe
    original=R.gpu_guard;last=[0.]
    def gpu_guard():
        if time.time()-last[0]>=5:original();last[0]=time.time()
    R.gpu_guard=gpu_guard
    R.E.N.offline()


def data(p,guard):
    return DataSource.data(p,guard)


def data_manifest(old,groups):
    return dict(old=O.Parent.manifest(old),fresh=[dict(id=key,sha256=row['sha256'],arrays=row['array_digest']) for key,g in groups for a,row in g])


def event(folder,who):
    def emit(stage,k):
        with (folder/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([who,stage,k])+'\n')
    return emit


def gpu():
    R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    f=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f


def main():
    p=plan();verify(p);setup(p);folder=Path(p['out']);R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (folder/'launch.json').exists();save(folder/'launch.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=O.Runtime(p);error=None;proposal=None;final407=None;screen=None;retained=p['initial'];u=None
    try:
        old,new=data(p,rt.guard)
        assert data_manifest(old,new)==read(Path(p['scope_reference']['source'])/'pg/DATA_MANIFEST.json'),'Ablation data differs from T651'
        save(folder/'DATA_MANIFEST.json',data_manifest(old,new))
        assert time.time()+p['learning_seconds']<p['absolute_learning_stop'];gl=gpu()
        obj=M.make(p);shadow=M.make(p);before=M.state(obj)
        start=time.time()
        def guard():rt.guard();assert time.time()<min(start+p['learning_seconds'],p['absolute_learning_stop'])-5;R.gpu_guard()
        u=M.fit(obj,old,new,p,guard,event(folder,'live'));v=M.fit(shadow,old,new,p,guard,event(folder,'shadow'))
        assert u==v and digest(M.state(obj))==digest(M.state(shadow))
        mature=digest(M.state(obj))
        proposal=M.export(obj,p,folder/'proposal.pt.gz')
        save(folder/'LEARNING.json',dict(update=u,trained_state_hash=mature,proposal=proposal))
        M.load(obj,before);M.load(shadow,before);assert digest(M.state(obj))==digest(before)==digest(M.state(shadow))
        save(folder/'ROLLBACK.json',dict(before=digest(before),restored=digest(M.state(obj)),all_Actor_Adam=True))
        obj[0].cpu();shadow[0].cpu();del obj,shadow,old,new;torch.cuda.empty_cache();fcntl.flock(gl,fcntl.LOCK_UN);gl.close()
        # Launch native only after training tensors are freed; fourGiB cap.
        os.environ['CUDA_VISIBLE_DEVICES']=''
        rt.pool=R.N.Pool(dict(p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request(p['screen_jobs'],proposal,'screen36');rows=[r for a,r in vv];by={r['job_id']:r for r in rows}
        old_ok=all(R.safe(by[k]) for k in p['old_screen_ids'])
        gained=[r['job_id'] for r in rows if R.safe(r) and r['job_id'] not in p['before_success_ids']]
        screen=dict(old_protected_ids_retained=old_ok,field24_success=sum(R.safe(by[j['id']]) for j in p['field_jobs']),new_field_success_ids=gained,rows=rows)
        if old_ok and gained:
            assert time.time()+p['full_evaluation_reserve_seconds']<rt.deadline
            rest=[j for j in p['evaluation_jobs'] if j['id'] not in by];assert len(rest)==371
            other=rt.request(rest,proposal,'remaining371');by.update({r['job_id']:r for a,r in other})
            ordered=[by[j['id']] for j in p['evaluation_jobs']];ids={r['job_id'] for r in ordered if R.safe(r)}
            lost=sorted(set(p['before_success_ids'])-ids);gained=sorted(ids-set(p['before_success_ids']));accepted=not lost and bool(gained)
            final407=dict(success=len(ids),success_ids=sorted(ids),lost_prior=lost,gained_prior=gained,accepted=accepted)
            with gzip.open(folder/'JOINT407.json.gz','xb') as f:f.write(json.dumps(dict(model=proposal,rows=ordered,reused36=True)).encode())
            if accepted:retained=proposal
        keys=[by[j['id']] for j in p['key_jobs']]
        rt.request(p['key_jobs'],proposal,'exact20',True,keys)
        save(folder/'SCREEN.json',screen)
    except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,proposal=proposal,update=u,screen=screen,final407=final407,retained=retained,new_RL_Actor_Adam=u['Actor_Adam'] if u else 0,new_Critic=0))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=plan();verify(p);setup(p);folder=Path(p['out']);s=read(folder/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for g in read(folder/'worker_generations.json'):assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    rows=[json.loads(x) for x in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows};assert len(by)==len(rows)
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path'])
        if r['full']:O.N.inspect(p,a,r)
        else:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            ref=by[r['reference_path']];assert ref['array_hashes']==r['array_hashes'] and ref['integration_sha256']==r['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and len(rows)+6==s['commands']<=p['max_commands']
    start=time.time();deadline=min(start+p['audit_seconds']-20,p['absolute_all_stop']-60)
    def guard():assert time.time()<deadline
    old,new=data(p,guard);assert read(folder/'DATA_MANIFEST.json')==data_manifest(old,new)
    gl=gpu();obj=M.make(p);before=M.state(obj)
    def guarded():guard();R.gpu_guard()
    u=M.fit(obj,old,new,p,guarded,event(folder,'independent_history'))
    learned=read(folder/'LEARNING.json');assert u==learned['update'] and digest(M.state(obj))==learned['trained_state_hash'] and obj[0].phash()==s['proposal']['policy_hash']
    M.load(obj,before);assert digest(M.state(obj))==read(folder/'ROLLBACK.json')['restored']
    fcntl.flock(gl,fcntl.LOCK_UN);gl.close()
    calls=[json.loads(v) for v in (folder/'optimizer_calls.jsonl').read_text().splitlines()]
    expected=[[where,k] for k in range(u['Actor_Adam']) for where in ['BEFORE','AFTER']]
    for who in ['live','shadow','independent_history']:assert [v[1:] for v in calls if v[0]==who]==expected
    screen=s['screen'];assert len(screen['rows'])==36
    actual={r['job_id']:r for r in screen['rows']};assert screen['old_protected_ids_retained']==all(R.safe(actual[k]) for k in p['old_screen_ids'])
    assert screen['field24_success']==sum(R.safe(actual[j['id']]) for j in p['field_jobs'])
    assert screen['new_field_success_ids']==[r['job_id'] for r in screen['rows'] if R.safe(r) and r['job_id'] not in p['before_success_ids']]
    if s['final407']:
        joint=read_gzip(folder/'JOINT407.json.gz');ids={r['job_id'] for r in joint['rows'] if R.safe(r)}
        assert len(joint['rows'])==407 and len({r['job_id'] for r in joint['rows']})==407
        assert s['final407']['success_ids']==sorted(ids) and s['final407']['lost_prior']==sorted(set(p['before_success_ids'])-ids)
        assert s['final407']['accepted']==(not(set(p['before_success_ids'])-ids) and bool(ids-set(p['before_success_ids'])))
    assert sha(p['initial']['path'])==p['initial']['sha256'] and sha(s['proposal']['path'])==s['proposal']['sha256']
    save(folder/'audit.json',dict(state='PASS',all_original_history_and_all_Actor_Adam_rebuilt=True,new_native=0,historical_Actor_Adam=u['Actor_Adam'],exact20_verified=True,full407_conditional_checked=True))


def read_gzip(f):return json.loads(gzip.decompress(Path(f).read_bytes()))

if __name__=='__main__':globals()[sys.argv[1]]()

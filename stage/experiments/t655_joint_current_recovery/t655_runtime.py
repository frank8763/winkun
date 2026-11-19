"""Two fixed coverage supports, fit before evaluation and one finite queue."""
from pathlib import Path
import copy,fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t655_joint_current_recovery';DOC=ROOT/'stage/docs/day_20261119/t655'
sys.path.insert(0,str(ROOT/'stage/experiments/t646_offline_night'))
import t646_runtime as Shared
import t655_learning as M
import t655_native as Native
O,R=Shared.O,Shared.R
read,save,sha,digest,torch,np=Shared.read,Shared.save,Shared.sha,Shared.digest,Shared.torch,Shared.np

def rootplan():return read(HERE/'protocol.json')
def plan():
    root=rootplan();name=sys.argv[2];assert name in ['shared','joint'];c=read(OUT/name/'protocol.json')
    assert c['_root_sha256']==sha(OUT/'protocol.json');return dict(root,**c,out=str(OUT/name))

def verify(p,*args):
    root=rootplan();assert root['authorization_approved'] and time.time()<root['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert R.used(OUT)<=root['max_bytes'] and R.used(Path(p['out']))<=(root['shared_bytes'] if p['arm']=='shared' else root['child_bytes'])
    assert R.used(ROOT/'stage/runs')-R.used(OUT)+root['max_bytes']<root['global_bytes']
    assert R.D.B.shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in root['inputs'].items():assert sha(f)==h,f

def safe(row):
    boundary='fullfield_start_pose' in row['request']['job']
    return bool(row['complete'] and row['success'] and not row['collision'] and (not boundary or not row.get('field_boundary',False)))

def setup(p):
    folder=Path(p['out']);O.OUT=folder;O.DOC=DOC/p['arm'];R.OUT=folder;R.DOC=DOC/p['arm'];R.HERE=HERE
    R.verify=verify;R.M=M;O.M=M;R.setup(p);R.N.worker=Native.worker;O.N.inspect=Native.inspect;R.N.inspect=Native.inspect;R.safe=safe
    original=R.gpu_guard;last=[0.]
    def gpu_guard():
        if time.time()-last[0]>=5:original();last[0]=time.time()
    R.gpu_guard=gpu_guard;R.E.N.offline()

def shared_rows():return [json.loads(v) for v in gzip.decompress((OUT/'shared/cases.jsonl.gz').read_bytes()).decode().splitlines()]

def load_data(p,guard):
    source=read(p['old_source_protocol']);old=Shared.load(source,guard)
    report=read(OUT/'shared/summary.json');assert report['qualified_for_learning'] and read(OUT/'shared/audit.json')['state']=='PASS'
    rows=shared_rows();datasets={}
    field=[]
    q=R.L.frozen(p,p['initial']['path'])
    for row in p['field_keep_rows']:
        guard();assert sha(row['path'])==row['sha256'] and safe(row);a=R.E.N.S.arrays(row['path']);Native.inspect(p,a,row)
        g=M.features(q,a,row);g['target']=torch.tensor(a['locations']);assert float((M.location(q,g)-g['target']).abs().max())<1e-10;field.append(g)
    assert len(field)==8 and {g['id'] for g in field}==set(p['required_field_ids'])
    for method in p['methods']:
        groups=[]
        for job in p['original_jobs']:
            rr=[r for r in rows if r['role']=='old_'+method+'_fresh' and r['job_id']==job['id']];assert len(rr)==8
            groups.append((job['id'],[(R.E.N.S.arrays(r['path']),r) for r in rr]))
        for job in p['field_train_jobs']:
            rr=[r for r in rows if r['role']=='field_fresh' and r['job_id']==job['id']];assert len(rr)==4
            groups.append((job['id'],[(R.E.N.S.arrays(r['path']),r) for r in rr]))
        datasets[method]=dict(fields=field,groups=groups)
    return old,datasets

def manifest(old,data):return dict(old=O.Parent.manifest(old),methods={key:[dict(id=task,sources=[r['sha256'] for a,r in values]) for task,values in v['groups']] for key,v in data.items()},field_keep=[g['source_sha256'] for g in data['original']['fields']])

def event(folder,who,method):
    def emit(stage,k):
        with (folder/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([who,method,stage,k])+'\n')
    return emit

def gpu():
    R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    f=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f

def train_all(p,rt,folder):
    old,data=load_data(p,rt.guard);save(folder/'DATA_MANIFEST.json',manifest(old,data));gl=gpu();results={}
    start=time.time()
    def guard():rt.guard();assert time.time()<min(start+p['learning_seconds'],p['absolute_learning_stop'])-5;R.gpu_guard()
    try:
        for method in p['methods']:
            pc=dict(p,arm=method);obj=M.make(pc);shadow=M.make(pc);before=M.state(obj)
            try:
                u=M.fit(obj,old,data[method],pc,guard,event(folder,'live',method));v=M.fit(shadow,old,data[method],pc,guard,event(folder,'shadow',method))
                assert u==v and digest(M.state(obj))==digest(M.state(shadow));mature=digest(M.state(obj))
                proposal=M.export(obj,pc,folder/('proposal_'+method+'.pt.gz'))
                results[method]=dict(update=u,trained_state_hash=mature,proposal=proposal)
                save(folder/('LEARNING_'+method+'.json'),results[method])
            finally:
                M.load(obj,before);M.load(shadow,before);assert digest(M.state(obj))==digest(before)==digest(M.state(shadow))
                save(folder/('ROLLBACK_'+method+'.json'),dict(before=digest(before),restored=digest(M.state(obj))))
                obj[0].cpu();shadow[0].cpu();del obj,shadow;torch.cuda.empty_cache()
    finally:fcntl.flock(gl,fcntl.LOCK_UN);gl.close()
    return results

def main():
    p=plan();verify(p);setup(p);folder=Path(p['out']);R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(folder/'launch.json',dict(pid=os.getpid(),epoch=time.time()));rt=O.Runtime(p);error=None;trained={};results={}
    try:
        assert time.time()+p['learning_seconds']<p['absolute_learning_stop'];trained=train_all(p,rt,folder)
        os.environ['CUDA_VISIBLE_DEVICES']='';torch.cuda.empty_cache()
        rt.pool=R.N.Pool(dict(p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        # Fixed priority; no post-hoc best checkpoint selection. Both fitted
        # before the first native screen; each full407 needs full remaining time.
        for method in p['evaluation_priority']:
            model=trained[method]['proposal'];vv=rt.request(p['screen_jobs'],model,'screen_'+method);rows=[r for a,r in vv];by={r['job_id']:r for r in rows}
            old_ok=all(safe(by[k]) for k in p['old_screen_ids']);field_ok=all(safe(by[k]) for k in p['required_field_ids']);joint=None;reason='SCREEN_REJECTED'
            if old_ok and field_ok:
                if time.time()+p['full_evaluation_reserve_seconds']<rt.deadline:
                    rest=[j for j in p['evaluation_jobs'] if j['id'] not in by];assert len(rest)==371
                    by.update({r['job_id']:r for a,r in rt.request(rest,model,'remaining_'+method)})
                    ordered=[by[j['id']] for j in p['evaluation_jobs']];ids={r['job_id'] for r in ordered if safe(r)}
                    lost=sorted(set(p['before_success_ids'])-ids);accepted=not lost and set(p['required_field_ids'])<=ids
                    joint=dict(success=len(ids),success_ids=sorted(ids),lost_prior=lost,gained_prior=sorted(ids-set(p['before_success_ids'])),accepted=accepted)
                    with gzip.open(folder/('JOINT407_'+method+'.json.gz'),'xb') as f:f.write(json.dumps(dict(rows=ordered,model=model)).encode())
                    reason='FULL407_ACCEPTED' if accepted else 'FULL407_REJECTED'
                else:reason='FULL407_NOT_ADMITTED_INSUFFICIENT_COMPLETE_TIME_RESERVE'
            keys=[by[j['id']] for j in p['key_jobs']];rt.request(p['key_jobs'],model,'exact_'+method,True,keys)
            results[method]=dict(field24_success=sum(safe(by[j['id']]) for j in p['field_jobs']),old_screen_kept=old_ok,all8_mean_field_retained=field_ok,
                screen=rows,final407=joint,reason=reason,retained=model if joint and joint['accepted'] else p['safe_retained'])
            save(folder/'RESULTS.json',results)
    except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,trained=trained,results=results,actual_Adam=sum(v['update']['Actor_Adam'] for v in trained.values())))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def audit():
    p=plan();verify(p);setup(p);folder=Path(p['out']);s=read(folder/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for g in read(folder/'worker_generations.json'):assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    rows=[json.loads(v) for v in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows};assert len(rows)==len(by)
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        if row['full']:Native.inspect(p,a,row)
        else:assert all(digest(v)==row['array_hashes'][key] for key,v in a.items())
        if row.get('exact'):
            ref=by[row['reference_path']];assert ref['array_hashes']==row['array_hashes'] and ref['integration_sha256']==row['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and len(rows)+6==s['commands']<=p['max_commands']
    deadline=time.time()+p['audit_seconds']-20
    def guard():assert time.time()<deadline;R.gpu_guard()
    old,data=load_data(p,guard);assert read(folder/'DATA_MANIFEST.json')==manifest(old,data);gl=gpu()
    try:
        for method in p['methods']:
            pc=dict(p,arm=method);obj=M.make(pc);before=M.state(obj);u=M.fit(obj,old,data[method],pc,guard,event(folder,'independent_history',method));learned=s['trained'][method]
            assert u==learned['update'] and digest(M.state(obj))==learned['trained_state_hash'] and obj[0].phash()==learned['proposal']['policy_hash']
            M.load(obj,before);assert digest(M.state(obj))==read(folder/('ROLLBACK_'+method+'.json'))['restored']
            obj[0].cpu();del obj;torch.cuda.empty_cache()
            result=s['results'][method];actual={r['job_id']:r for r in result['screen']};assert len(actual)==36
            assert result['old_screen_kept']==all(safe(actual[k]) for k in p['old_screen_ids'])
            assert result['all8_mean_field_retained']==all(safe(actual[k]) for k in p['required_field_ids'])
            assert result['field24_success']==sum(safe(actual[j['id']]) for j in p['field_jobs'])
            if result['final407']:
                j=json.loads(gzip.decompress((folder/('JOINT407_'+method+'.json.gz')).read_bytes()));ids={r['job_id'] for r in j['rows'] if safe(r)}
                assert len(j['rows'])==len({r['job_id'] for r in j['rows']})==407
                assert result['final407']['success_ids']==sorted(ids) and result['final407']['accepted']==(set(p['before_success_ids'])<=ids and set(p['required_field_ids'])<=ids)
    finally:fcntl.flock(gl,fcntl.LOCK_UN);gl.close()
    calls=[json.loads(v) for v in (folder/'optimizer_calls.jsonl').read_text().splitlines()]
    for method in p['methods']:
        expected=[[where,k] for k in range(s['trained'][method]['update']['Actor_Adam']) for where in ['BEFORE','AFTER']]
        for who in ['live','shadow','independent_history']:assert [v[2:] for v in calls if v[:2]==[who,method]]==expected
    save(folder/'audit.json',dict(state='PASS',all_actual_Actor_Adam_rebuilt=True,all_history_and_exact=True,new_native=0,historical_updates=s['actual_Adam']))
if __name__=='__main__':globals()[sys.argv[1]]()

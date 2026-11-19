"""Bounded child: full histories, fresh joint blocks, exact rollback and audit."""
from pathlib import Path
import copy,fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t660_joint_night_relay';DOC=ROOT/'stage/docs/overnight_20261119'
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'));import control as Resource
sys.path.insert(0,str(ROOT/'stage/experiments/t654_verified_deviation_recovery'));import t654_runtime as Prior
import t660_learning as M
import t660_native as N
import admission
R,O,torch,np=Prior.R,Prior.O,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest

def plan():
    root=read(HERE/'protocol.json');child=read(OUT/sys.argv[2]/'protocol.json')
    assert child['_root_sha256']==sha(OUT/'protocol.json')
    return dict(root,**child,out=str(OUT/sys.argv[2]))

def verify(p,*args):
    root=read(OUT/'protocol.json')
    assert root['authorization_approved'] and time.time()<root['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert R.used(OUT)<=root['max_bytes'] and R.used(Path(p['out']))<=p['child_bytes']
    assert R.used(ROOT/'stage/runs')-R.used(OUT)+root['max_bytes']<root['global_bytes']
    assert R.D.B.shutil.disk_usage(ROOT).free>=40*2**30
    Resource.resources()
    h=read(DOC/'AFFINITY_GUARD.json');assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    for f,h in root['inputs'].items():assert sha(f)==h,f

def setup(p):
    folder=Path(p['out']);Prior.OUT,Prior.DOC,Prior.HERE,Prior.verify=folder,DOC/p['arm'],HERE,verify
    Prior.setup(p);R.N.worker=N.worker;R.N.inspect=O.N.inspect=N.inspect
    O.OUT=folder;R.OUT=folder;R.verify=verify;R.safe=Prior.safe
    N.P.frozen=N.L.frozen

def pool(rt):
    folder=Path(rt.p['out']);f=folder/'worker_generations.json';old=read(f) if f.exists() else []
    rt.pool=R.N.Pool(dict(rt.p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
    new=read(f);save(f,old+new)
    if old:rt.commands+=6

def close(rt):
    if rt.pool:rt.pool.close();rt.pool=None

def gpu(p,folder):
    R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    lock=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return lock

def free_gpu(lock,objects):
    for obj in objects:
        obj[0].cpu();obj[1].cpu()
    torch.cuda.empty_cache();fcntl.flock(lock,fcntl.LOCK_UN);lock.close();os.environ['CUDA_VISIBLE_DEVICES']=''

def event(folder,who,rnd):
    def emit(kind,stage,k):
        with (folder/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([who,rnd,kind,stage,k])+'\n')
    return emit

def fit_block(p,folder,rnd,before,groups,guard):
    lock=gpu(p,folder);objects=[]
    try:
        obj=M.make(p,'cuda');shadow=M.make(p,'cuda');objects=[obj,shadow]
        if before is not None:M.load(obj,M.read_state(before));M.load(shadow,M.read_state(before))
        initial=M.cpu_state(obj);assert digest(initial)==digest(M.cpu_state(shadow))
        start=M.save_state(folder/f'start_{rnd:02d}.pt.gz',initial,p['state_bytes'])
        refs=M.references(p,obj[0],guard)
        u=M.fit(obj,refs,groups,p,guard,event(folder,'live',rnd))
        v=M.fit(shadow,refs,groups,p,guard,event(folder,'shadow',rnd))
        assert u==v and digest(M.state(obj))==digest(M.state(shadow))
        after=M.cpu_state(obj);proposal=M.export(obj,p,folder/f'proposal_{rnd:02d}.pt.gz')
        full=M.save_state(folder/f'trained_{rnd:02d}.pt.gz',after,p['state_bytes'])
        M.load(obj,initial);M.load(shadow,initial)
        assert digest(M.cpu_state(obj))==digest(initial)==digest(M.cpu_state(shadow))
        del refs
        return dict(round=rnd,start=start,trained=full,update=u,proposal=proposal,rollback_verified=True)
    finally:free_gpu(lock,objects)

def main():
    p=plan();verify(p);setup(p);folder=Path(p['out']);R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(folder/'science_started.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=O.Runtime(p);error=None;rounds=[];best=None;current=p['initial'];before=None;rejects=0;full_done=False
    previous_screen_ids=set(p['original_screen_success_ids']);stop='FIXED_ROUNDS_COMPLETE';live=value=0
    try:
        for rnd in range(p['rounds']):
            # Complete next block reserve; time denial is a normal bounded
            # stop, not an invitation to shorten evaluation or reuse batches.
            if time.time()+p['block_science_reserve_seconds']+p['learning_seconds']>=min(rt.deadline,p['absolute_learning_stop']):
                stop='NEXT_COMPLETE_BLOCK_TIME_ADMISSION_DENIED';break
            groups=[];training_rows=[]
            if p['arm'].startswith('ppo'):
                pool(rt)
                jobs=[j for j in p['training_jobs'] for unused in range(p['episodes_per_task'])]
                seeds=[[p['night_seed'],p['arm_index'],rnd,i] for i in range(len(jobs))]
                vv=rt.request(jobs,current,f'fresh_{rnd:02d}',True,seeds=seeds)
                rt.request(jobs,current,f'exact_fresh_{rnd:02d}',False,[r for a,r in vv],seeds=seeds)
                training_rows=[r for a,r in vv]
                groups=[(j['id'],[(a,r) for a,r in vv if r['job_id']==j['id']]) for j in p['training_jobs']]
                close(rt)
            start=time.time()
            def guard():
                rt.guard();assert time.time()<min(start+p['learning_seconds'],p['absolute_learning_stop'])-5;R.gpu_guard()
            record=fit_block(p,folder,rnd,before,groups,guard)
            live+=record['update']['Actor_Adam'];value+=record['update']['Value_Adam']
            record['training_rows']=training_rows
            pool(rt)
            vv=rt.request(p['screen_jobs'],record['proposal'],f'screen_{rnd:02d}',False)
            screen=[r for a,r in vv];ids={r['job_id'] for r in screen if Prior.safe(r)}
            keys=[r for r in screen if r['job_id'] in p['key_ids']]
            rt.request([r['request']['job'] for r in keys],record['proposal'],f'exact_key_{rnd:02d}',True,keys)
            native_ok,lost_screen=admission.stable_screen(previous_screen_ids,ids)
            record.update(screen=screen,lost_screen_ids=lost_screen,full407=None,
                          research_state_kept=False,retained_all384_verified=False)
            # At most one full407 per arm, only with a screen-safe acquisition.
            gain=ids-set(p['before_success_ids'])
            if native_ok and gain and not full_done:
                if time.time()+p['full_evaluation_reserve_seconds']<rt.deadline:
                    full_done=True;by={r['job_id']:r for r in screen}
                    rest=[j for j in p['evaluation_jobs'] if j['id'] not in by]
                    zz=rt.request(rest,record['proposal'],f'rest407_{rnd:02d}',False)
                    by.update({r['job_id']:r for a,r in zz});safe_ids={k for k,r in by.items() if Prior.safe(r)}
                    lost=sorted(set(p['before_success_ids'])-safe_ids)
                    accepted=not lost and bool(safe_ids-set(p['before_success_ids']))
                    record['full407']=dict(success=len(safe_ids),success_ids=sorted(safe_ids),lost_prior=lost,
                        gained_prior=sorted(safe_ids-set(p['before_success_ids'])),accepted=accepted,
                        all387=set(p['required387_ids'])<=safe_ids)
                    native_ok=native_ok and not lost
                    if accepted:best=dict(model=record['proposal'],full407=record['full407']);record['retained_all384_verified']=True
                else:record['full407_not_run']='Complete reserved wall time unavailable; null'
            close(rt)
            if native_ok:
                before=record['trained'];current=record['proposal'];previous_screen_ids|=ids
                rejects=0;record['research_state_kept']=True
            else:rejects+=1
            record['consecutive_native_rejections']=rejects;rounds.append(record);save(folder/'ROUNDS.json',rounds)
            save(folder/'progress.json',dict(rounds=len(rounds),Actor_live=live,Value_live=value,physics=sum(rt.counts),commands=rt.commands))
            if rejects>=2:stop='TWO_CONSECUTIVE_NATIVE_REJECTIONS_NEXT_BRANCH';break
    except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        close(rt);rt.commands+=3
        calls=folder/'optimizer_calls.jsonl'
        if calls.exists():
            entries=[json.loads(x) for x in calls.read_text().splitlines()]
            live=sum(v[0]=='live' and v[2:4]==['actor','BEFORE'] for v in entries)
            value=sum(v[0]=='live' and v[2:4]==['value','BEFORE'] for v in entries)
        save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,
            physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,rounds=rounds,stop_reason=stop,
            live_Actor=live,live_Value=value,best=best,retained=best['model'] if best else p['initial'],
            full407=best['full407'] if best else None,research_model=current,no_runtime_teacher_or_router=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def audit():
    p=plan();verify(p);setup(p);folder=Path(p['out']);s=read(folder/'summary.json')
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    generations=read(folder/'worker_generations.json') if (folder/'worker_generations.json').exists() else []
    assert all(not Path('/proc',str(pid)).exists() for g in generations for pid in [g['parent'],*g['workers']])
    rows=[json.loads(x) for x in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()] if (folder/'cases.jsonl.gz').exists() else []
    by={r['path']:r for r in rows};assert len(by)==len(rows)
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        if row['full']:N.inspect(p,a,row)
        else:assert all(digest(v)==row['array_hashes'][key] for key,v in a.items())
        if row.get('exact'):
            ref=by[row['reference_path']];assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics']
    assert s['commands']==len(rows)+6+6*max(0,len(generations)-1)<=p['max_commands']
    start=time.time();last=[0.]
    def guard():
        assert time.time()<min(start+p['audit_seconds']-10,p['absolute_all_stop']-60)
        if time.time()-last[0]>5:verify(p);R.gpu_guard();last[0]=time.time()
    counts_a=counts_v=0
    next_start_hash=None;previous_screen_ids=set(p['original_screen_success_ids']);full_seen=0
    for record in s['rounds']:
        assert record['round']==counts_round(s['rounds'],record)
        if next_start_hash is not None:assert record['start']['state_hash']==next_start_hash
        if not record['research_state_kept']:next_start_hash=record['start']['state_hash']
        else:next_start_hash=record['trained']['state_hash']
        actual_ids={r['job_id'] for r in record['screen'] if Prior.safe(r)}
        assert record['lost_screen_ids']==sorted(previous_screen_ids-actual_ids)
        assert len(record['screen'])==36 and {r['job_id'] for r in record['screen']}=={j['id'] for j in p['screen_jobs']}
        if record['research_state_kept']:assert previous_screen_ids<=actual_ids;previous_screen_ids|=actual_ids
        if record['full407']:
            full_seen+=1
            rr=[r for r in rows if r['role']==f"rest407_{record['round']:02d}"]+record['screen']
            assert len(rr)==407 and len({r['job_id'] for r in rr})==407
            ids={r['job_id'] for r in rr if Prior.safe(r)};v=record['full407']
            assert v['success_ids']==sorted(ids) and v['success']==len(ids)
            assert v['lost_prior']==sorted(set(p['before_success_ids'])-ids)
            assert v['accepted']==(not v['lost_prior'] and bool(ids-set(p['before_success_ids'])))
            assert v['all387']==(set(p['required387_ids'])<=ids)
        gl=gpu(p,folder);obj=M.make(p,'cuda')
        try:
            M.load(obj,M.read_state(record['start']));refs=M.references(p,obj[0],guard)
            training=record['training_rows'];groups=[(j['id'],[(R.E.N.S.arrays(r['path']),r) for r in training if r['job_id']==j['id']]) for j in p['training_jobs']] if training else []
            result=M.fit(obj,refs,groups,p,guard,event(folder,'independent_history',record['round']))
            assert result==record['update'] and digest(M.cpu_state(obj))==record['trained']['state_hash']
            assert obj[0].phash()==record['proposal']['policy_hash']
            M.load(obj,M.read_state(record['start']));assert digest(M.cpu_state(obj))==record['start']['state_hash']
            counts_a+=result['Actor_Adam'];counts_v+=result['Value_Adam'];del refs
        finally:free_gpu(gl,[obj])
    assert counts_a==s['live_Actor'] and counts_v==s['live_Value']
    assert full_seen<=1
    if s['best']:
        b=s['best'];assert any(r['proposal']==b['model'] and r['full407']==b['full407'] and r['full407']['accepted'] for r in s['rounds'])
    calls=[json.loads(v) for v in (folder/'optimizer_calls.jsonl').read_text().splitlines()] if s['rounds'] else []
    for record in s['rounds']:
        for who in ['live','shadow','independent_history']:
            for kind,expected in [('actor',record['update']['Actor_Adam']),('value',record['update']['Value_Adam'])]:
                actual=[v[3:] for v in calls if v[:3]==[who,record['round'],kind]]
                assert actual==[[stage,k] for k in range(expected) for stage in ['BEFORE','AFTER']]
    save(folder/'audit.json',dict(state='PASS',all_complete_history_and_density=True,all_live_shadow_independent_equal=True,
        independent_Actor=counts_a,independent_Value=counts_v,full_Actor_Value_Adam_rollback=True,new_native=0))

def counts_round(rounds,record):return rounds.index(record)

def frozen_main():
    p=plan();verify(p);setup(p);folder=Path(p['out']);R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(folder/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));rt=O.Runtime(p);error=None
    try:
        pool(rt)
        jobs=p['frozen_jobs'];seeds=p.get('frozen_seeds')
        vv=rt.request(jobs,p['initial'],'frozen_fresh',p['frozen_full'],seeds=seeds)
        rt.request(jobs,p['initial'],'frozen_exact',False,[r for a,r in vv],seeds=seeds)
    except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,raw_preserved=True))
    finally:
        close(rt);rt.commands+=3
        fresh=[r for r in rt.rows if r['role']=='frozen_fresh']
        save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,
            physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,live_Actor=0,live_Value=0,
            measured=len(fresh),safe=sum(Prior.safe(r) for r in fresh),rows=fresh,model=p['initial'],
            kind=p['kind'],stochastic_coverage_is_not_deterministic_acceptance=seeds is not None))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def frozen_audit():
    p=plan();verify(p);setup(p);folder=Path(p['out']);s=read(folder/'summary.json')
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    for g in read(folder/'worker_generations.json'):assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    rows=[json.loads(x) for x in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows}
    assert len(rows)==len(by)==2*len(p['frozen_jobs'])
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        if row['full']:N.inspect(p,a,row)
        else:assert all(digest(v)==row['array_hashes'][key] for key,v in a.items())
        if row.get('exact'):
            ref=by[row['reference_path']];assert ref['array_hashes']==row['array_hashes'] and ref['integration_sha256']==row['integration_sha256']
        assert row['loaded_model_sha256']==p['initial']['sha256']
    assert sum(r['physics'] for r in rows)==s['physics'] and len(rows)+6==s['commands']
    assert s['physics']<=p['max_physics'] and s['commands']<=p['max_commands']
    save(folder/'audit.json',dict(state='PASS',all_exact_arrays_and_full_histories_checked=True,
         independent_Actor=0,independent_Value=0,new_native=0,new_optimizer=0))

if __name__=='__main__':{'main':main,'audit':audit,'frozen_main':frozen_main,'frozen_audit':frozen_audit}[sys.argv[1]]()

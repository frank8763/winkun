"""New Monday external review; all original deadlines and failure files immutable."""
from pathlib import Path
import fcntl,gzip,importlib.util,json,os,shutil,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t699_matched_value'))
import t699_common as C
DOC=ROOT/'stage/docs/day_20261123/t700';OUT=ROOT/'stage/runs/t700_plain_external_review'
OLD=C.OUT;CHILD=OLD/'known_plain';UNIT='stage-t700-review.service'
save,sha,used=C.save,C.sha,C.used

def read(path):
    value=C.read(path)
    if isinstance(value,dict) and '_review_manifests' in value:
        rows=[]
        for manifest in value['_review_manifests']:
            assert sha(manifest['path'])==manifest['sha256']
            rows.extend(json.loads(Path(manifest['path']).read_text()))
        value['inputs']={r[0]:r[1] for r in rows}
        value['input_stats']={r[0]:r[2:] for r in rows}
        assert len(value['inputs'])==len(rows)
    return value

def old_stopped():
    spec=importlib.util.spec_from_file_location('t700_previous_guard',ROOT/'stage/experiments/t699_unexecuted_continuation/source_sync.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.stopped()

def stopped():
    old_stopped()
    props=subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','-p','ActiveState'],text=True)
    assert 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props)
    if (OUT/'DRIVER_STARTED.json').exists():
        s=read(OUT/'SEALED.json');assert s['review_finished']
        assert not Path('/proc',str(read(OUT/'DRIVER_STARTED.json')['pid'])).exists()
    else:assert read(DOC/'REGISTERED.json')['state']=='REGISTERED_NOT_LAUNCHED'

def immutable(p,full=False):
    assert time.time()<p['absolute_all_stop']
    assert sha(OUT/'protocol.json')==sha(HERE/'protocol.json')
    assert used(OUT)<8*2**20 and used(ROOT/'stage/runs')<10*2**30
    assert shutil.disk_usage(ROOT).free>40*2**30
    C.resources()
    for f,h in p['inputs'].items():
        q=Path(f);assert [q.stat().st_size,q.stat().st_mtime_ns]==p['input_stats'][f],f
        if full:assert sha(q)==h,f

def register():
    old_stopped();C.resources();assert not OUT.exists()
    auth=read(DOC/'AUTHORIZATION.json');l=read(C.WEEKEND/'LEDGER.json')
    assert l['audit_seconds']+900<18000 and l['transport_seconds']+600<18000
    p=read(OLD/'protocol.json');inputs=dict(p['inputs'])
    for folder in [OLD,ROOT/'stage/runs/t699_unexecuted_continuation']:
        for f in folder.rglob('*'):
            if f.is_file():inputs[str(f)]=sha(f)
    for f in list(HERE.glob('*.py'))+[DOC/'AUTHORIZATION.json',DOC/'PLAN.md']:inputs[str(f)]=sha(f)
    manifest=[[f,h,Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f,h in sorted(inputs.items())]
    (HERE/'inputs.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
    q=dict(id='T700_NEW_EXTERNAL_REVIEW',absolute_all_stop=auth['absolute_all_stop'],audit_seconds=900,
           _input_manifest=dict(path=str(HERE/'inputs.json'),sha256=sha(HERE/'inputs.json')),
           original_protocol_sha256=sha(OLD/'protocol.json'),new_native=0,new_optimizer=0,new_Actor=0)
    OUT.mkdir();blob=json.dumps(q,separators=(',',':'))+'\n';(OUT/'protocol.json').write_text(blob);(HERE/'protocol.json').write_text(blob)
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=time.time(),inputs=len(inputs),absolute_all_stop=q['absolute_all_stop']))
    immutable(read(OUT/'protocol.json'),True)
    print('T700_REGISTERED',len(inputs))

def review():
    p=read(OUT/'protocol.json');old_stopped();immutable(p,True)
    assert not (OUT/'DRIVER_STARTED.json').exists();os.sched_setaffinity(0,{15})
    save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time()))
    start=time.time();outside=used(ROOT/'stage/runs')-used(OUT);error=None;summary=None;count=0
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T700 fixed900s audit')));signal.alarm(min(900,int(p['absolute_all_stop']-time.time()-5)))
    try:
        import t699_native as N
        import mujoco
        def forbidden(*args,**kwargs):raise AssertionError('T700 cannot integrate native or optimize')
        mujoco.mj_step=forbidden;N.torch.optim.Adam.step=forbidden
        pc=dict(read(OLD/'protocol.json'),**read(CHILD/'protocol.json'))
        # New review deadline is only an auditor guard; never write this overlay to an old protocol.
        pc['absolute_all_stop']=p['absolute_all_stop']
        last=[0.]
        def guard(*args,**kwargs):
            if time.time()-last[0]>2 or kwargs.get('force'):
                immutable(p,False);last[0]=time.time()
        N.verify=guard;N.setup(pc)
        rows=[json.loads(x) for x in gzip.decompress((CHILD/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
        by={r['path']:r for r in rows};fresh=[r for r in rows if not r.get('exact')]
        assert len(rows)==500 and len(fresh)==431 and len(by)==500
        s=read(CHILD/'summary.json');assert s['state']=='COMPLETE' and sha(s['model']['path'])==s['model']['sha256']
        first=next(r for r in rows if r['full']);arr=N.T.R.E.N.S.arrays(first['path'])
        corrupt=dict(first,array_hashes=dict(first['array_hashes'],actions='0'*64))
        try:N.BASE_INSPECT(pc,arr,corrupt)
        except AssertionError:pass
        else:raise AssertionError('Corrupt action digest accepted')
        save(OUT/'PREFLIGHT.json',dict(state='PASS',actual_original_inspector=True,corrupt_digest_rejected=True,new_native=0,new_optimizer=0))
        for row in rows:
            guard();assert used(OUT)<8*2**20;assert sha(row['path'])==row['sha256']
            assert row['policy_hash']==s['model']['policy_hash'] and row['loaded_model_sha256']==s['model']['sha256']
            arrays=N.T.R.E.N.S.arrays(row['path'])
            if row['full']:N.BASE_INSPECT(pc,arrays,row)
            else:assert all(N.digest(v)==row['array_hashes'][k] for k,v in arrays.items())
            if row.get('exact'):
                ref=by[row['reference_path']];assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
            count+=1
            if count%25==0:save(OUT/'status.json',dict(state='ACTIVE',completed=count,total=500,pid=os.getpid(),epoch=time.time()))
        ids=sorted(r['job_id'] for r in fresh if N.T.Prior.safe(r))
        assert ids==s['success_ids'] and {r['job_id'] for r in fresh}=={j['id'] for j in pc['evaluation_jobs']}
        assert sum(r['physics'] for r in rows)==s['physics']==11001515 and len(rows)+14==s['commands']==514
        assert sorted(set(pc['required403_ids'])-set(ids))==s['lost403']
        summary=dict(core_success=s['core_success'],core_total=407,total_success=len(ids),total=431,lost403=s['lost403'],
                     source_success_ids=s['source_success_ids'],supplementary_success_ids=s['supplementary_success_ids'])
        immutable(p,True);assert used(ROOT/'stage/runs')-used(OUT)==outside
    except BaseException:error=traceback.format_exc()
    finally:
        signal.alarm(0);elapsed=time.time()-start
        report=dict(state='FAILED' if error else 'PASS',error=error,review_finished=True,external_new_authorization=True,
                    original_T699_stays_FAILED=True,old_audit_stays_FAILED_EXPIRED=True,rows_checked=count,summary=summary,
                    new_physics=0,new_commands=0,new_optimizer=0,new_Actor=0,audit_seconds=elapsed,retained='T698',epoch=time.time())
        save(OUT/'audit.json',report);save(OUT/'SEALED.json',report);save(OUT/'status.json',report);save(DOC/'RESULT.json',report)
        l=read(C.WEEKEND/'LEDGER.json');key='T700_EXTERNAL_REVIEW'
        reviews=l.setdefault('external_readonly_reviews',[])
        assert not any(j['id']==key for j in reviews)
        l['audit_seconds']+=elapsed;reviews.append(dict(id=key,state=report['state'],external_readonly_review=True,out=str(OUT),accounted=dict(audit_seconds=elapsed,physics=0,commands=0,Actor_each=0)))
        save(C.WEEKEND/'LEDGER.json',l)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def sync():
    stopped();l=read(C.WEEKEND/'LEDGER.json');p=read(OUT/'protocol.json')
    assert l['transport_seconds']+600<18000 and time.time()+300<p['absolute_all_stop']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T700 source300')));signal.alarm(295)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as Old
    Old.stopped=stopped;Old.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'));import t626_sync_source
    t626_sync_source.main();stopped();signal.alarm(0)

def launch():
    stopped();p=read(OUT/'protocol.json');immutable(p,True);assert not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert time.time()+920<p['absolute_all_stop']
    cmd=['systemd-run','--user','--unit=stage-t700-review','--collect','--property=Type=exec','--property=CPUQuota=100%',
         '--property=CPUWeight=10','--property=AllowedCPUs=15','--property=Nice=19','--property=MemoryMax=2G','--property=MemorySwapMax=0',
         '--property=KillMode=control-group','--property=TimeoutStopSec=5','--property=RuntimeMaxSec=920',
         f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/review.log','--property=StandardError=inherit',
         '--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',str(ROOT/'run_stage.sh'),str(HERE/'review.py'),'review']
    c=subprocess.run(cmd,text=True,capture_output=True);assert c.returncode==0,c.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit=UNIT,source_remote=receipt['remote_commit']));print(c.stderr)

if __name__=='__main__':{'register':register,'review':review,'sync':sync,'launch':launch}[sys.argv[1]]()

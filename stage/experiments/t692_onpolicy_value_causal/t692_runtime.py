"""Single registered fit and conditional intervention job, Actor never learns."""
from pathlib import Path
import ctypes,fcntl,gc,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t692_onpolicy_value_causal';DOC=ROOT/'stage/docs/weekend_20261120_23/t692'
import t692_wait as Wait
from status_io import save
import t692_learning as L
import t692_native as N
T,M,torch,np=L.T,L.M,L.torch,L.np
read,sha,digest=L.read,L.sha,L.digest
T.OUT,T.DOC,T.HERE,T.Wait=OUT,DOC,HERE,Wait

def verify(p,*unused):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')-T.R.used(OUT)+p['max_bytes']<p['global_bytes']
    assert T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    Wait.resources();h=read(DOC/'AFFINITY_GUARD.json');assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f

T.verify=verify

def setup(p):
    T.setup(p);T.R.N.worker=N.worker;T.R.N.inspect=T.O.N.inspect=T.N.inspect=N.inspect
    N.P.frozen=N.L.frozen;assert T.R.N.Pool.__init__.__globals__['worker'] is N.worker

def event(who,arm):
    def emit(stage,k):
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([who,arm,k,'value',stage])+'\n')
    return emit

def schedule(p):
    jobs=[];seeds=[]
    for task_index,base in enumerate(p['branch_jobs']):
        for action_index,delta in enumerate(p['action_deltas']):
            for rep in range(4):
                jobs.append(dict(base,action_delta=delta,action_index=action_index,replicate=rep))
                seeds.append([69220261121,task_index,rep])
    return jobs,seeds

def causal_result(p,fresh,predictions):
    matrices={};contrasts=[]
    for job in p['branch_jobs']:
        rows=[r for r in fresh if r['job_id']==job['id']];assert len(rows)==16
        matrix=[];flags=[]
        for action in range(4):
            rr=sorted([r for r in rows if r['request']['job']['action_index']==action],key=lambda r:r['seed'][-1]);assert len(rr)==4
            matrix.append([int(T.Prior.safe(r))-int(r['collision']) for r in rr])
            flags.append([dict(safe=T.Prior.safe(r),collision=r['collision'],timeout=r['timeout'],boundary=r['field_boundary'],path=r['path']) for r in rr])
        matrices[job['id']]=dict(Y=matrix,flags=flags)
        for name,plus,minus in [('drive',0,1),('steer',2,3)]:
            difference=np.asarray(matrix[plus])-np.asarray(matrix[minus]);empirical=float(difference.mean())
            predicted=predictions[job['id']][str(plus)]['values']['aware']-predictions[job['id']][str(minus)]['values']['aware']
            blind=predictions[job['id']][str(plus)]['values']['blind']-predictions[job['id']][str(minus)]['values']['blind'];assert blind==0.
            informative=abs(empirical)>=.5;correct=predicted*empirical>0
            contrasts.append(dict(task=job['id'],head=name,paired_Y=difference.tolist(),empirical_mean_gap=empirical,predicted_gap=predicted,
                blind_gap=blind,blind_tie=True,absolute_gap_error=abs(predicted-empirical),informative=informative,correct_sign=bool(correct)))
    active=[x for x in contrasts if x['informative']]
    checks=dict(at_least2_informative=len(active)>=2,at_least2_prefixes=len({x['task'] for x in active})>=2,
        includes_field=any(x['task'].startswith('t502_') for x in active),no_informative_wrong_sign=all(x['correct_sign'] for x in active))
    return dict(gate='FINITE_ACTION_CALIBRATION_PASS' if all(checks.values()) else ('INCONCLUSIVE_NO_REQUIRED_GAPS' if not all(checks[k] for k in ['at_least2_informative','at_least2_prefixes','includes_field']) else 'FINITE_ACTION_CALIBRATION_FAILED'),
        checks=checks,matrices=matrices,contrasts=contrasts,replicates=4,CRN_not_independent_actions=True,no_Q_gradient_or_population_ranking_claim=True,no_Actor_guidance=True)

def count_commands(rows,commands):assert len(rows)+6==commands

def collect():
    p=read(OUT/'protocol.json');verify(p);setup(p);T.R.E.L.install_stop_handlers();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));rt=T.Runtime(p);error=None;results={};trained={};models={};fresh=[];gate=None;causal=None
    learnstart=time.time();last=[0.]
    def guard(phase='',key=''):
        assert time.time()<min(learnstart+p['learning_seconds'],p['absolute_learning_stop'])-10
        if time.time()-last[0]>5:rt.guard();last[0]=time.time()
    try:
        dataset=L.data(p,guard);save(OUT/'DATA_MANIFEST.json',dataset['manifest']);initial={};objects={}
        # All prescribed fixed training is completed before any heldout score.
        for arm in ['aware','blind']:
            obj=L.make(p,arm);shadow=L.make(p,arm);initial[arm]=digest(L.state(obj));assert initial[arm]==digest(L.state(shadow))
            live=L.fit(p,obj,dataset,guard,event('live',arm));other=L.fit(p,shadow,dataset,guard,event('shadow',arm))
            assert live==other and digest(L.state(obj))==digest(L.state(shadow))
            trained[arm]=live;models[arm]=L.export(obj,p,arm,OUT);objects[arm]=obj;del shadow
        assert initial['aware']==initial['blind']
        for arm,obj in objects.items():
            guard('heldout',arm);results[arm]=dict(train=L.metrics(obj,dataset,False),heldout=L.metrics(obj,dataset,True))
        regression=L.regression_gate(results);save(OUT/'LEARNING.json',dict(trained=trained,models=models,results=results,regression_gate=regression,initial_state_hashes=initial))
        predictions=L.branch_predictions(p,objects,dataset);save(OUT/'PRE_NATIVE_PREDICTIONS.json',dict(predictions=predictions,sha256=digest(predictions),epoch=time.time()))
        del objects,dataset;gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
        if regression['passed']:
            assert time.time()+p['native_reserve_seconds']<rt.deadline
            T.pool(rt);jobs,seeds=schedule(p)
            vv=rt.request(jobs,p['initial'],'branch_fresh',True,seeds=seeds);fresh=[r for a,r in vv]
            rt.request(jobs,p['initial'],'branch_exact',True,fresh,seeds=seeds)
            causal=causal_result(p,fresh,predictions);gate=causal['gate']
        else:gate='HELDOUT_ACTION_INPUT_GATE_FAILED'
        save(OUT/'RESULTS.json',dict(gate=gate,regression_gate=regression,causal=causal,no_Actor_guidance=True,full407=None))
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        T.close(rt);rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,
            gate=gate,rows=fresh,new_Actor=0,new_Value_live=sum(x['steps'] for x in trained.values()),new_Value_shadow=sum(x['steps'] for x in trained.values()),full407=None,no_model_promoted=True,retained=p['initial']))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def audit():
    p=read(OUT/'protocol.json');verify(p);setup(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    if (OUT/'worker_generations.json').exists():
        for g in read(OUT/'worker_generations.json'):assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();last=[0.]
    def guard(phase='',key=''):
        assert time.time()<min(start+p['audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>5:verify(p);last[0]=time.time()
    dataset=L.data(p,guard);assert dataset['manifest']==read(OUT/'DATA_MANIFEST.json');learning=read(OUT/'LEARNING.json');results={};objects={}
    for arm in ['aware','blind']:
        obj=L.make(p,arm);assert digest(L.state(obj))==learning['initial_state_hashes'][arm]
        info=L.fit(p,obj,dataset,guard,event('independent_history',arm));assert info==learning['trained'][arm]
        saved=M.P.checkpoint(learning['models'][arm]['path']);assert digest(L.state(obj))==saved['state_sha256']==digest(saved['state'])
        results[arm]=dict(train=L.metrics(obj,dataset,False),heldout=L.metrics(obj,dataset,True));objects[arm]=obj
    assert results==learning['results'];regression=L.regression_gate(results);assert regression==learning['regression_gate']
    predictions=L.branch_predictions(p,objects,dataset);assert predictions==read(OUT/'PRE_NATIVE_PREDICTIONS.json')['predictions']
    del objects,dataset;gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0);rows=[]
    if (OUT/'cases.jsonl.gz').exists():rows=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    by={r['path']:r for r in rows};assert len(by)==len(rows)
    for r in rows:
        guard('native_audit',r['path']);assert sha(r['path'])==r['sha256'] and r['loaded_model_sha256']==p['initial']['sha256']
        a=T.R.E.N.S.arrays(r['path']);assert r['full'];N.inspect(p,a,r)
        if r.get('exact'):
            previous=by[r['reference_path']];assert r['array_hashes']==previous['array_hashes'] and r['integration_sha256']==previous['integration_sha256']
    fresh=[r for r in rows if not r.get('exact')];assert fresh==s['rows']
    assert len(rows)==(128 if regression['passed'] else 0)
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'];count_commands(rows,s['commands']);assert s['commands']<=p['max_commands']
    causal=causal_result(p,fresh,predictions) if regression['passed'] else None;result=read(OUT/'RESULTS.json');assert result['causal']==causal and result['regression_gate']==regression
    calls=[json.loads(x) for x in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for who in ['live','shadow','independent_history']:
        for arm in ['aware','blind']:assert [(r[2],r[4]) for r in calls if r[:2]==[who,arm]]==[(k,stage) for k in range(500) for stage in ['BEFORE','AFTER']]
    assert not any(r[3]!='value' for r in calls)
    save(OUT/'audit.json',dict(state='PASS',all_original_features_split_labels_loss_and_models_exact=True,Value_independent=1000,new_Actor=0,new_native=0,
        all_actual_native_arrays_integrals_density_ownership_and_exact=True,commands_including_no_native6=True,gate=s['gate'],no_Actor_guidance=True))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':{'collect':collect,'audit':audit}[sys.argv[1]]()

"""Unique offline fit and independent full optimizer reconstruction."""
from pathlib import Path
import sys,importlib.util,time,os,fcntl,json,traceback
s=importlib.util.spec_from_file_location('t719_fit_controller',Path(__file__).resolve().parent/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
import t719_learning as L

def event(who):
    def emit(stage,k):
        with (C.OUT/'fit/fit_calls.jsonl').open('a') as f:f.write(json.dumps(dict(who=who,stage=stage,k=k,kind='actor'))+'\n')
    return emit

def fit():
    p=C.root_plan();folder=C.OUT/'fit';C.verify(p,force=True);L.torch.set_num_threads(8);C.save(folder/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));start=time.time();error=None;result=None;model=None
    lock=(C.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def guard():C.verify(p);assert time.time()<min(start+600,p['absolute_learning_stop'])-5
    try:
        obj=L.make(p);shadow=L.make(p);initial=L.state(obj);assert L.digest(initial)==L.digest(L.state(shadow));data,manifest=L.dataset(p,obj[0],guard);C.save(folder/'DATA_MANIFEST.json',manifest)
        result=L.fit(obj,data,p,guard,event('live'));other=L.fit(shadow,data,p,guard,event('shadow'));assert result==other and L.digest(L.state(obj))==L.digest(L.state(shadow));model=L.export(obj[0],p,folder/'proposal.pt.gz');joint=L.Base.write_state(folder/'actor_adam.pt.gz',L.state(obj));assert C.A.frozen(p,model['path']).phash()==obj[0].phash()
        C.save(folder/'LEARNING.json',dict(model=model,state=joint,fit=result,no_new_native_collection=True,offpolicy_weighted_MLE_not_PPO=True))
    except BaseException:error=traceback.format_exc();C.save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        C.verify(p,force=True);C.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,model=model,fit=result,physics=0,commands=0,elapsed_seconds=time.time()-start,finals={'outcome':dict(model=model)}));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def audit():
    p=C.root_plan();folder=C.OUT/'fit';C.verify(p,force=True);sm=C.read(folder/'summary.json');assert sm['state']=='COMPLETE' and not Path('/proc',str(sm['pid'])).exists();start=time.time();L.torch.set_num_threads(8)
    lock=(C.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def guard():C.verify(p);assert time.time()<min(start+300,p['absolute_all_stop'])-5
    try:
        obj=L.make(p);data,manifest=L.dataset(p,obj[0],guard);assert manifest==C.read(folder/'DATA_MANIFEST.json');result=L.fit(obj,data,p,guard,event('independent'));actual=C.read(folder/'LEARNING.json');assert result==sm['fit'] and L.digest(L.state(obj))==L.digest(L.Base.read_state(actual['state']))
        saved=L.Base.checkpoint(sm['model']['path']);assert L.digest(saved['joint_state']['actor'])==L.digest(obj[0].state_dict()) and C.sha(sm['model']['path'])==sm['model']['sha256'];assert obj[0].phash()==sm['model']['policy_hash']
        calls=[json.loads(x) for x in (folder/'fit_calls.jsonl').read_text().splitlines()]
        for who in ['live','shadow','independent']:assert [(r['stage'],r['k']) for r in calls if r['who']==who]==[(stage,k) for k in range(16) for stage in ['BEFORE','AFTER']]
        C.verify(p,force=True);C.save(folder/'audit.json',dict(state='PASS',independent_Actor_Adam=16,all72_source_features_weights_and_Actor_Adam_exact=True,new_native=0,new_learning=0))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

"""Fresh, bounded demonstration-only control; no parent-preservation loss."""
from pathlib import Path
import ast
import copy
import gzip
import io
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import multiprocessing as mp
import importlib.util

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t379_human_only_control'
DOC=ROOT/'stage/docs/narrow_20261105'
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import t377_core as S
np,torch=S.np,S.torch


def read(path):return json.loads(Path(path).read_text())
def sha(path):return S.K.C.sha(str(path))
def save(path,obj):S.save(path,obj)
def absent(pid):assert not Path('/proc',str(pid)).exists(),pid


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    old=read(p['source_protocol'])
    for f,h in old['input_hashes'].items():assert sha(f)==h,f
    assert S.K.S.used(ROOT/'stage/runs')-S.K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    budget(p)
    old.update(id='T379',folder=OUT.name,experiment='HUMAN_ONLY_MATCHED_CONTROL',out=str(OUT),
               contract=(DOC/'T379_PLAN.md').read_text(),fit_steps=3000,model_bytes=128*1024,full_bytes=384*1024,
               partial_bytes=384*1024,max_commands=2,max_physics=80000,
               final_bytes=p['final_bytes'])
    return old


def budget(p):
    assert S.K.S.used(OUT)<=p['final_bytes']
    assert S.K.S.used(ROOT/'stage/runs')<=p['global_bytes']
    assert S.shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def cache(q,data):
    result=[]
    with torch.no_grad():
        for d in data[1]:
            s=q.normalized(d['x']);h=q.core(q.encoder(s)[:,None])[0][:,0]
            result.append(dict(features=torch.cat([h,s],1),base=torch.atanh(d['b']/1.25)+q.head(h)))
    return result


def prediction(q,d,cached):
    return q.sequence(d['x'],d['b']) if cached is None else 1.25*torch.tanh(cached['base']+q.adapter(cached['features']))


def gradients(q,d,cached):
    pred=prediction(q,d,cached);first=d['first'];loss=.5*(pred[first:]-d['y'][first:]).square().mean()
    return loss,torch.autograd.grad(loss,q.trainable())


def fit(obj,data,guard,event,steps=3000,cached=True,checkpoint=None):
    q=obj.actor;frozen=S.fixed_hash(q);features=cache(q,data) if cached else [None,None];trace=[]
    for k in range(steps):
        guard();obj.opt.zero_grad(set_to_none=True);values=[]
        for d,feature in zip(data[1],features):
            pred=prediction(q,d,feature);first=d['first'];loss=.5*(pred[first:]-d['y'][first:]).square().mean()
            loss.backward();values.append(float(loss.detach()))
        norm=float(torch.nn.utils.clip_grad_norm_(q.trainable(),1.,error_if_nonfinite=True))
        event('BEFORE',k);obj.opt.step();obj.calls+=1;event('AFTER',k)
        assert S.fixed_hash(q)==frozen and all(v.grad is None for n,v in q.named_parameters() if not n.startswith('adapter.'))
        trace.append([*values,norm])
        if k==299 and checkpoint:checkpoint()
    return np.asarray(trace,dtype=np.float64)


def prepare():
    assert not OUT.exists()
    prior=ROOT/'stage/runs/t378_gradient_conflict'
    assert read(prior/'audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T378-COMPLETE.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for pid in [read(prior/'summary.json')['pid'],read(prior/'launch.json')['pid']]:absent(pid)
    final=read(prior/'results.json')['models'][1]['fixed_model_average_over_original300schedule']
    assert final['negative_gradient_interference'] and final['scalar_GD_human_descent_ratio']<0
    original=ROOT/'stage/runs/t377_full_human_observation_transfer/protocol.json'
    files=[original,ROOT/'stage/runs/t377_full_human_observation_transfer/summary.json',
           ROOT/'stage/runs/t377_full_human_observation_transfer/initial_fallback.pt.gz',
           prior/'results.json',prior/'audit.json',DOC/'T379_PLAN.md',*HERE.glob('*.py')]
    used=S.K.S.used(ROOT/'stage/runs')
    p=dict(id='T379',source_protocol=str(original),inputs={str(f):sha(f) for f in sorted(files)},
           fit_steps=3000,matched_update_checkpoint=300,live_Adam=3000,shadow_Adam=3000,audit_discarded_Adam=3000,
           preflight_discarded_Adam=8,final_bytes=int(1.5*1024**2),global_bytes=1024**3,min_free_bytes=40*1024**3,
           main_seconds=1800,final_native_reserve_seconds=300,audit_seconds=900,all_work_seconds=3300,
           max_commands=2,max_physics=80000,full_bytes=384*1024,partial_bytes=384*1024,
           model_bytes=128*1024,trace_bytes=96*1024,event_bytes=64*1024,
           new_RL_updates=0,new_native_training_collection=0,
           selection='Fixed final3000 model only;300model is a prespecified matched-update diagnostic, not native selection')
    assert used+p['final_bytes']<p['global_bytes'] and S.shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T379_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_reserved=p['final_bytes'],
         full_reservation=used+p['final_bytes'],global_limit=p['global_bytes'],worst_payload_bytes=1507328,
         live=3000,shadow=3000,audit=3000,preflight=8,max_commands=2,max_physics=80000))
    print('REGISTERED_T379_HUMAN_ONLY_FRESH_INITIALIZATION')


def preflight():
    p=read(HERE/'protocol.json');science=verify(p)
    assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    obj=S.NightLearner(science);data=S.dataset(science,obj.actor,lambda:None);report=S.contract(science,obj,data)
    state=obj.state();calls=0;rounds=[]
    for rnd in range(2):
        cached=cache(obj.actor,data)
        for d,c in zip(data[1],cached):
            a,ga=gradients(obj.actor,d,c);b,gb=gradients(obj.actor,d,None)
            assert torch.equal(a,b) and all(torch.equal(x,y) for x,y in zip(ga,gb))
        before=obj.state();shadow=S.NightLearner(science);shadow.load(before)
        a=fit(obj,data,lambda:None,lambda *_:None,steps=2,cached=True);calls+=2
        b=fit(shadow,data,lambda:None,lambda *_:None,steps=2,cached=False);calls+=2
        assert np.array_equal(a,b) and obj.hashes()==shadow.hashes()
        rounds.append(dict(round=rnd,before_state_sha256=S.digest(before),after_hashes=obj.hashes(),
                           paired_trace_sha256=S.digest(a),cached_uncached_loss_gradient_Adam_exact=True))
        if rnd==1:obj.load(before);assert S.digest(obj.state())==S.digest(before)
    obj.load(state);assert calls==8 and S.digest(obj.state())==S.digest(state)
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=0,discarded_Adam=8,source_contract=report,
         frozen_feature_cache_full_sequence_loss_gradients_and_mature_Adam_exact=True,
         whole_history_and_human_mask_preserved=True,discarded_real_data_clone_rounds=rounds))
    print('PREFLIGHT_PASS_8_DISCARDED_ADAM_ZERO_NATIVE')


def main():
    p=read(HERE/'protocol.json');science=verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();deadline=read(OUT/'launch.json')['main_deadline_epoch'];learn_end=deadline-300
    obj=S.NightLearner(science);initial=obj.state();last=0.;counts=None;worker=None;pipe=None;rows=[];replays=[];commands=0
    calls=dict(live=0,shadow=0);phase='DATASET';model=None;result={};state='FAILED'
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();assert time.time()<deadline-4;budget(p)
        save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,updates=calls,commands=commands,
             physics_steps=0 if counts is None else int(counts[0]),deadline_epoch=deadline))
    def guard():assert time.time()<learn_end;status()
    event_file=None
    def event(replica,where,k):
        if where=='AFTER':calls[replica]+=1
        event_file.write((json.dumps([replica,where,k])+'\n').encode())
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T379 main deadline')))
    signal.alarm(max(1,int(deadline-time.time())-2))
    save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]))
    try:
        data=S.dataset(science,obj.actor,guard);result['initial_fit_metrics']=S.metrics(obj.actor,data)
        source_initial=ROOT/'stage/runs/t377_full_human_observation_transfer/initial_fallback.pt.gz'
        old_state=torch.load(io.BytesIO(gzip.decompress(source_initial.read_bytes())),map_location='cpu',weights_only=False)['state']
        assert S.digest(obj.state())==S.digest(old_state)
        shadow=S.NightLearner(science);shadow.load(initial);phase='HUMAN_ONLY_FIT';status(True)
        def checkpoint():
            result['matched300_fit_metrics']=S.metrics(obj.actor,data)
            result['matched300_model']=S.export(science,obj,OUT/'matched300.pt.gz')
        with gzip.GzipFile(filename=str(OUT/'optimizer_calls.jsonl.gz'),mode='wb',mtime=0) as event_file:
            trace=fit(obj,data,guard,lambda a,k:event('live',a,k),checkpoint=checkpoint)
            other=fit(shadow,data,guard,lambda a,k:event('shadow',a,k))
        assert np.array_equal(trace,other) and obj.hashes()==shadow.hashes()
        assert (OUT/'optimizer_calls.jsonl.gz').stat().st_size<=p['event_bytes']
        buf=io.BytesIO();np.savez_compressed(buf,trace=trace);assert len(buf.getvalue())<=p['trace_bytes'];(OUT/'trace.npz').write_bytes(buf.getvalue())
        model=S.export(science,obj,OUT/'candidate.pt.gz');result['final_fit_metrics']=S.metrics(obj.actor,data)
        save(OUT/'block.json',dict(input_hash=S.digest(initial),output_hashes=obj.hashes(),trace_sha256=S.digest(trace),live=3000,shadow=3000))
        phase='NATIVE_CENTRE';status(True);assert time.time()+160<deadline
        ctx=mp.get_context('spawn');counts=ctx.Array('q',1);pipe,child=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']='1'
        worker=ctx.Process(target=S.night_worker,args=(child,0,science,counts));worker.start();child.close();os.environ['STAGE_PROCESS_CPU']='0'
        def receive():
            while not pipe.poll(.25):status()
            return pipe.recv()
        ready=receive();assert ready['ready'] and ready['affinity']==[1]
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[worker.pid],ready=ready))
        reference=None
        for role in ['centre','centre_readback']:
            commands+=1;job=science['narrow_jobs'][0]
            pipe.send(dict(job,id=role+'_'+job['id'],source_reset_id=job['id'],role=role,arm='memory',rho=0.,
                 learned=obj.actor.learned(),policy_hash=obj.actor.phash(),deadline_epoch=deadline-15,
                 readback_model=model['path'] if role=='centre_readback' else None,stochastic=False))
            item=receive();assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays']
            if not r['complete']:
                buf=io.BytesIO();np.savez_compressed(buf,**S.packed(a));assert len(buf.getvalue())<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(buf.getvalue());save(OUT/'partial_case.json',r);raise RuntimeError('Censored native evidence preserved')
            if reference is None:
                buf=io.BytesIO();np.savez_compressed(buf,**S.packed(a));assert len(buf.getvalue())<=p['full_bytes']
                path=OUT/'centre.npz';path.write_bytes(buf.getvalue());r.update(path=str(path),sha256=sha(path));reference=item;rows.append(r);save(OUT/'cases.json',rows)
            else:
                assert r['array_hashes']==reference['row']['array_hashes'] and r['integration_sha256']==reference['row']['integration_sha256']
                assert a.keys()==reference['arrays'].keys() and all(np.array_equal(v,reference['arrays'][k]) for k,v in a.items())
                r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);replays.append(r);save(OUT/'replays.json',replays)
        result['centre_safe_GOAL']=rows[0]['success'];state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,calls=calls,commands=commands))
    finally:
        if pipe is not None:
            try:pipe.send(None)
            except (OSError,EOFError):pass
        if worker is not None:
            worker.join(5)
            if worker.is_alive():worker.kill();worker.join(3)
        signal.alarm(0)
    if model is None:obj.load(initial);model=S.export(science,obj,OUT/'candidate.pt.gz')
    summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,commands=commands,
         physics_steps=0 if counts is None else int(counts[0]),updates=calls,model=model,result=result,
         all_workers_stopped=worker is None or not worker.is_alive(),old275=None,six54=None,fresh16=None,reload78=None,
         new_RL_updates=0,new_native_training_collection=0,stable_original_six=False,entered7or8=False,research_candidate_not_promoted=True)
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);budget(p);print(json.dumps(summary))


def audit():
    p=read(HERE/'protocol.json');science=verify(p);summary=read(OUT/'summary.json')
    assert summary['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    for pid in [summary['pid'],read(OUT/'launch.json')['pid'],*read(OUT/'workers.json')['workers']]:absent(pid)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    remaining=read(OUT/'launch.json')['all_work_deadline_epoch']-time.time()-20
    assert remaining>0
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T379 independent audit deadline')));signal.alarm(max(1,int(min(900,remaining))))
    obj=S.NightLearner(science);data=S.dataset(science,obj.actor,lambda:None)
    assert S.metrics(obj.actor,data)==summary['result']['initial_fit_metrics'];events=[]
    def checkpoint():
        model=summary['result']['matched300_model'];assert sha(model['path'])==model['sha256']
        original=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),map_location='cpu',weights_only=False)['state']
        assert S.digest(obj.state())==S.digest(original) and S.metrics(obj.actor,data)==summary['result']['matched300_fit_metrics']
    trace=fit(obj,data,lambda:None,lambda a,k:events.append([a,k]),cached=False,checkpoint=checkpoint)
    with np.load(OUT/'trace.npz') as z:assert np.array_equal(z['trace'],trace)
    with gzip.open(OUT/'optimizer_calls.jsonl.gz','rt') as f:recorded=[json.loads(v) for v in f]
    for replica in ['live','shadow']:assert [[v[1],v[2]] for v in recorded if v[0]==replica]==events
    model=summary['model'];assert sha(model['path'])==model['sha256']
    original=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),map_location='cpu',weights_only=False)['state']
    assert S.digest(obj.state())==S.digest(original) and S.metrics(obj.actor,data)==summary['result']['final_fit_metrics']
    r=read(OUT/'cases.json')[0];rep=read(OUT/'replays.json')[0];assert sha(r['path'])==r['sha256'];a=S.arrays(r['path'])
    assert {k:S.digest(v) for k,v in a.items()}==r['array_hashes']==rep['array_hashes']
    assert r['integration_sha256']==rep['integration_sha256'] and rep['all_arrays_hidden_history_integration_exact']
    assert summary['commands']==2 and summary['physics_steps']==2*int(a['native_counts'].sum())<=80000
    spec=importlib.util.spec_from_file_location('stage_t379_forward_audit',ROOT/'stage/experiments/t373_factual_parent_support/run.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker);checker.check_forward(obj.actor,a,r)
    assert summary['updates']==dict(live=3000,shadow=3000) and summary['elapsed_s']<1800
    report=dict(state='PASS',new_native=0,historical_discarded_Adam=3000,
         full_uncached_history_all3000losses_gradnorms_ActorAdam_and300checkpoint_exact=True,
         live_shadow_event_ledger_and_final_model_exact=True,full_native_saved_model_arrays_hidden_history_integration_exact=True,
         commands=2,physics_steps=summary['physics_steps'])
    save(OUT/'audit.json',report);save(DOC/'T379_AUDIT.json',report);signal.alarm(0);print(json.dumps(report))


def launch():
    p=read(HERE/'protocol.json');verify(p)
    assert not (OUT/'launch.json').exists() and read(OUT/'preflight.json')['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (OUT/'main.log').open('xb') as log:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage human-only learning control',
             'timeout','--signal=TERM','--kill-after=1s','1800s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'main'],
             cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,main_deadline_epoch=epoch+1800,all_work_deadline_epoch=epoch+3300,
         source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


if __name__=='__main__':
    assert sys.argv[1] in ['prepare','preflight','launch','main','audit']
    globals()[sys.argv[1]]()

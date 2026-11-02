from pathlib import Path
import core as K
import builtins,fcntl,io,gzip,json,os,shutil,signal,symtable,sys,time,traceback,copy
import numpy as np
import torch
ROOT=K.ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t293_contrastive_action';DOC=ROOT/'stage/docs/distance_20261102'


def used(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file())
def verify(p):
    for f,h in p['input_hashes'].items():assert K.sha(f)==h,f
    assert (OUT/'protocol.json').read_bytes()==(HERE/'protocol.json').read_bytes()
    assert used(OUT)<p['own_bytes'] and used(ROOT/'stage/runs')<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
def setup():torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


def preflight():
    assert not any((OUT/f).exists() for f in ['status.json','preflight.json']);p=json.loads((HERE/'protocol.json').read_text());verify(p);setup();data=K.load(p)
    for mod in [sys.modules[__name__],K]:
        st=symtable.symtable(Path(mod.__file__).read_text(),mod.__file__,'exec');missing=set()
        def walk(t):
            for s in t.get_symbols():
                if s.is_referenced() and s.is_global() and s.get_name() not in vars(mod) and not hasattr(builtins,s.get_name()):missing.add(s.get_name())
            for c in t.get_children():walk(c)
        walk(st);assert not missing,(mod.__name__,missing)
    torch.manual_seed(p['seed']);q=K.Head();x=torch.tensor(data['x'][data['train'][:16]]);g=torch.tensor(data['g'][data['train'][:16]])
    with torch.no_grad():
        original=q(x,g,True);changed=x.clone();changed[:,120:122]=7;assert torch.equal(original,q(changed,g,True))
        repeated=K.metrics(np.zeros((len(data['test']),32),np.float32),data);assert abs(repeated['equal_episode_top1']-1/32)<1e-12
    loss=torch.nn.functional.cross_entropy(q(x,g),torch.arange(16));loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in q.parameters())
    assert sum(v.numel() for v in q.parameters())==11168
    assert (500*128<2**31) and 2*256*1024+3*4559*32*4+4559*31*2+32768<p['payload_bytes']
    assert (OUT/'protocol.json').stat().st_size+120000<p['metadata_bytes']
    K.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,Actor_updates=0,optimizer_steps=0,parameters_each=11168,all_source_and_pair_semantics_verified=True,whole_family_split_preserved=True,action_blind_exact_invariance=True,top1_constant_ties_equal_chance=True,all_parameter_gradients_finite=True))
    print('T293 preflight PASS: no optimizer/native/Actor, source/action-blind/tie-aware contracts')


def main():
    assert not (OUT/'status.json').exists();p=json.loads((HERE/'protocol.json').read_text());start=time.time();setup();verify(p);lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outside=used(ROOT/'stage/runs')-used(OUT);assert outside+p['own_bytes']<p['global_bytes'];calls=[];state='COMPLETE';models={};opt={};metrics={};exports={};initial=None;last=0.;holder={}
    def guard(phase):
        nonlocal last
        assert time.time()<start+p['wall_seconds']-p['eval_reserve_seconds']
        if time.time()-last>5:
            last=time.time();verify(p);assert used(ROOT/'stage/runs')-used(OUT)==outside
            K.save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=time.time(),main_epoch=start,absolute_deadline_epoch=start+p['wall_seconds'],optimizer_calls=len(calls),native_steps=0,Actor_updates=0))
    def event(name,step):
        calls.append(dict(head=name,step=step));assert len(calls)<=1000
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(calls[-1])+'\n')
    def alarm(signum,frame):raise TimeoutError('T293 absolute bounded wall')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds'])
    try:
        K.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],native_envs_created=0));data=K.load(p)
        models,opt,losses,initial=K.train(data,p,guard,event,holder);scores={};neg=None
        for name,q in models.items():
            scores[name],n=K.score_arrays(q,data,p,name=='blind')
            if neg is not None:assert np.array_equal(neg,n)
            neg=n;metrics[name]=K.metrics(scores[name],data)
        # A permutation is a dependence description only, not a causal branch test.
        perm=copy.copy(data);perm['x']=data['x'].copy();ids=data['test'];perm['x'][ids,120:122]=data['x'][np.roll(ids,1),120:122]
        blind,_=K.score_arrays(models['blind'],perm,p,True);assert np.array_equal(blind,scores['blind'])
        shuffled,_=K.score_arrays(models['action'],perm,p,False)
        buf=io.BytesIO();np.savez_compressed(buf,action=scores['action'],blind=scores['blind'],action_permuted=shuffled,
            negatives=neg.astype(np.uint16),test=data['test'].astype(np.uint32),losses=losses,**data['normalization'])
        blob=buf.getvalue();assert len(blob)<p['evidence_bytes'];(OUT/'evidence.npz').write_bytes(blob)
        K.save(OUT/'metrics.json',dict(**metrics,action_permutation=K.metrics(shuffled,data),screen=K.screen(metrics['action'],metrics['blind'])))
    except BaseException:
        state='FAILED';K.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),optimizer_invocations=len(calls),native_steps=0,Actor_updates=0))
        models=holder.get('models',{});opt=holder.get('opt',{});initial=holder.get('initial')
    for name,q in models.items():
        payload=dict(model=q.state_dict(),optimizer=opt[name].state_dict(),steps=sum(x['head']==name for x in calls),normalization=data['normalization'],protocol_sha256=K.sha(HERE/'protocol.json'),initial_hash=initial)
        b=io.BytesIO();torch.save(payload,b);blob=gzip.compress(b.getvalue(),compresslevel=6,mtime=0);assert len(blob)<p['model_bytes']
        f=OUT/(name+'_head.pt.gz');assert not f.exists();f.write_bytes(blob);exports[name]=dict(path=str(f),sha256=K.sha(f),model_hash=K.digest(payload['model']),optimizer_hash=K.digest(payload['optimizer']),steps=payload['steps'])
    signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();verify(p)
    result=dict(state=state,optimizer_calls=len(calls),head_steps={n:sum(x['head']==n for x in calls) for n in ['action','blind']},Actor_updates=0,native_steps=0,physics_steps=0,all_workers_stopped=True,models=exports,
        metrics=metrics,screen=K.screen(metrics['action'],metrics['blind']) if state=='COMPLETE' else None,elapsed_s=time.time()-start,own_bytes=used(OUT),global_bytes=used(ROOT/'stage/runs'),automatic_next_Actor_training=False,seven_or_eight_started=False,
        limitation='Conditional relative future endpoint density-ratio pilot with time and original120 behavior-goal context, not discounted occupancy/first-hit safeQ or full CRL navigation. Noise negatives are not unreachable labels. Same consumed96 trajectories/26395pairs not independent samples, matched action-blind heads no causal intervention claim. NoActor/native, no automatic pilot extension.')
    K.save(OUT/'summary.json',result);K.save(OUT/'status.json',result);print(json.dumps(result,default=K.clean),flush=True)


def audit():
    dest=DOC/'T293_AUDIT.json';assert not dest.exists();p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text());assert s['state']=='COMPLETE';setup();verify(p)
    for pid in [json.loads((OUT/'workers.json').read_text())['parent'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live/reused PID',pid))
    start=time.time();count=[]
    def guard(phase):assert time.time()-start<p['audit_seconds']
    def alarm(signum,frame):raise TimeoutError('T293 audit deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['audit_seconds']);data=K.load(p);models,opt,losses,initial=K.train(data,p,guard,lambda n,i:count.append((n,i)))
    with np.load(OUT/'evidence.npz') as z:
        assert np.array_equal(z['losses'],losses) and np.array_equal(z['test'],data['test'])
        for k,v in data['normalization'].items():assert np.array_equal(z[k],v)
        for name,q in models.items():
            meta=s['models'][name];assert K.sha(meta['path'])==meta['sha256'];payload=torch.load(io.BytesIO(gzip.decompress(Path(meta['path']).read_bytes())),map_location='cpu',weights_only=False)
            assert K.digest(q.state_dict())==K.digest(payload['model'])==meta['model_hash'] and K.digest(opt[name].state_dict())==K.digest(payload['optimizer'])==meta['optimizer_hash']
            assert payload['steps']==500 and payload['initial_hash']==initial
            scores,neg=K.score_arrays(q,data,p,name=='blind');assert np.array_equal(scores,z[name]) and np.array_equal(neg,z['negatives']);assert K.metrics(scores,data)==s['metrics'][name]
        perm=copy.copy(data);perm['x']=data['x'].copy();ids=data['test'];perm['x'][ids,120:122]=data['x'][np.roll(ids,1),120:122]
        shuffled,_=K.score_arrays(models['action'],perm,p,False);blind,_=K.score_arrays(models['blind'],perm,p,True)
        assert np.array_equal(shuffled,z['action_permuted']) and np.array_equal(blind,z['blind'])
        raw=json.loads((OUT/'metrics.json').read_text());assert K.metrics(shuffled,data)==raw['action_permutation'] and K.screen(s['metrics']['action'],s['metrics']['blind'])==s['screen']==raw['screen']
    assert len(count)==1000 and s['head_steps']==dict(action=500,blind=500);signal.alarm(0)
    K.save(dest,dict(state='PASS',native_added=0,Actor_updates=0,disposable_shadow_head_optimizer_steps=1000,all_models_Adam_loss_normalization_predictions_negatives_exact=True,all_sources_SHA_verified=True,all_original_processes_absent=True,screen=s['screen'],elapsed_s=time.time()-start,summary_sha256=K.sha(OUT/'summary.json')))
    print('T293 independent audit PASS: two500-step heads/Adam and heldout evidence exact')


if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:preflight()
    elif sys.argv[1:]==['--audit']:audit()
    else:assert not sys.argv[1:];main()

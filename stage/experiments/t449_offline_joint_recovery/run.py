"""Finite offline Stage relay: tail imitation, validated recovery, own anchored PPO."""
from pathlib import Path
import ast,contextlib,copy,gzip,importlib.util,io,json,math,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t449_offline_joint_recovery';DOC=ROOT/'stage/docs/overnight_20261109'
sys.path.insert(0,str(ROOT/'stage/experiments/t446_plain_joint'))
import t446_learning as P
spec=importlib.util.spec_from_file_location('stage449_runtime448',ROOT/'stage/experiments/t448_full_gradient_control/run.py');W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)
R=W.R;Z=R.Z;torch,np,E,L=P.torch,P.np,P.E,P.L
read,save,sha,digest=P.read,P.save,P.sha,P.digest
def used(folder=None):return sum(f.stat().st_size for f in (Path(folder) if folder else ROOT/'stage/runs').rglob('*') if f.is_file())
def verify(p,full=False):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert used(OUT)<=p['max_bytes'] and used()-used(OUT)+p['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (p['inputs'] if full else p['protected_files']).items():assert sha(f)==h,f
def setup_runtime():
    R.OUT=OUT;R.verify=verify;R.M.OUT=OUT;R.M.verify=verify
def initial(p):
    obj=P.make(p);P.load(obj,P.checkpoint(p['night_initial']['path'])['joint_state']);assert digest(P.state(*obj))==p['night_initial']['joint_hash'];return obj
def dataset(p):return P.dataset(p,p['teacher_rows'])
def tail_step(obj,groups):
    q,v,ao,vo=obj;ao.zero_grad(set_to_none=True);total=0.
    for task in range(10):
        selected=[g for g in groups if g['task']==task];weight=1/16 if task<8 else 1/4
        for off in range(0,len(selected),8):
            chunk=selected[off:off+8];x=torch.nn.utils.rnn.pad_sequence([g['x'] for g in chunk]);y=torch.nn.utils.rnn.pad_sequence([g['y'] for g in chunk]);loc=q.sequence(x);loss=0.
            for i,g in enumerate(chunk):
                per=P.action_loss(loc[:len(g['x']),i],y[:len(g['x']),i]).mean(-1)
                # Half ordinary episode risk, half empirical upper20% decision risk.
                loss=loss+(.5*per.mean()+.5*per.topk(max(1,math.ceil(.2*len(per)))).values.mean())*weight/len(selected)
            loss.backward();total+=float(loss.detach())
    norm=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True));ao.step();return dict(loss=total,gradient_norm=norm)
def mix_step(obj,groups,order,corrections,k):
    q,v,ao,vo=obj;ao.zero_grad(set_to_none=True);selected=[groups[i] for i in order[k%len(order)]]
    x=torch.nn.utils.rnn.pad_sequence([g['x'] for g in selected]);y=torch.nn.utils.rnn.pad_sequence([g['y'] for g in selected]);loc=q.sequence(x)
    losses=[P.action_loss(loc[:len(g['x']),i],y[:len(g['x']),i]).mean() for i,g in enumerate(selected)]
    original=sum(losses[:8])/16+sum(losses[8:])/4;g=corrections[k%len(corrections)]
    recovery=P.action_loss(q.sequence(g['x']),g['y']).mean();loss=.5*(original+recovery);loss.backward()
    norm=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True));ao.step();return dict(loss=float(loss.detach()),original=float(original.detach()),recovery=float(recovery.detach()),gradient_norm=norm)
def anchored_ppo(obj,arrays,p,groups,order,rnd,guard,event):
    q,v,ao,vo=obj;device=next(q.parameters()).device;batch=[];errors=[]
    for i,a in enumerate(arrays):
        assert not np.asarray(a['ownership']).any() and np.asarray(a['stochastic']).all()
        assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        x=torch.tensor(a['states'],device=device);raw=torch.tensor(a['raw_actions'],device=device);atom=torch.tensor(a['atom'],device=device);old=torch.tensor(a['logp'],device=device)
        with torch.no_grad():
            loc=q.sequence(x);er=float((loc-loc.new_tensor(a['locations'])).abs().max());lp=float((L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max());assert er<1e-10 and lp<1e-9
            vv=v(torch.tensor(np.concatenate([a['states'],a['next_states'][-1:]]),device=device)).cpu().numpy()
        mask=np.ones((len(x),1),bool);mask[-1]=False
        adv,target=L.physical_gae(a['rewards'][:,None],vv[:-1,None],vv[1:,None],a['native_counts'].astype(np.int64)[:,None],mask,mask,gamma_native=p['gamma_native'],lambda_native=p['lambda_native'])
        batch.append(dict(x=x,raw=raw,atom=atom,old=old,adv=adv[:,0],target=loc.new_tensor(target[:,0]),group=i//2));errors.append([er,lp])
    for group in [0,1]:
        selected=[b for b in batch if b['group']==group];mean=np.mean([b['adv'].mean() for b in selected]);sd=math.sqrt(max(0,np.mean([np.mean(b['adv']**2) for b in selected])-mean**2))+1e-8
        for b in selected:b['adv']=b['old'].new_tensor((b['adv']-mean)/sd)
    trace=[]
    for k in range(p['epochs']):
        guard();ao.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True);pg=vl=0.
        for b in batch:
            ratio=(L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp();loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/4;loss.backward();pg+=float(loss.detach())
            loss=.5*(v(b['x'])-b['target']).square().mean()/4;loss.backward();vl+=float(loss.detach())
        selected=[groups[i] for i in order[rnd*p['epochs']+k]];x=torch.nn.utils.rnn.pad_sequence([g['x'] for g in selected]);y=torch.nn.utils.rnn.pad_sequence([g['y'] for g in selected]);loc=q.sequence(x)
        losses=[P.action_loss(loc[:len(g['x']),i],y[:len(g['x']),i]).mean() for i,g in enumerate(selected)];anchor=sum(losses[:8])/16+sum(losses[8:])/4;anchor.backward()
        an=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True));vn=float(torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();vo.step();event('AFTER',k)
        with torch.no_grad():kl=sum(float(((z:=(L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in batch)/4
        trace.append(dict(epoch=k,policy_loss=pg,value_loss=vl,anchor=float(anchor.detach()),actor_norm=an,value_norm=vn,KL=kl))
        if kl>p['target_kl']:break
    return dict(trace=trace,density_errors=errors,Actor_Adam=len(trace),V_Adam=len(trace))

def worker(pipe,index,p,counts):
    E.N.offline();R.C.ban_learning();os.sched_setaffinity(0,{index});torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.set_out(p['out'])
    folder=Path(p['out'])/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'console.log').open('a');stdout,stderr=sys.stdout,sys.stderr;s=w=None;expert=None
    def guard(physical=False):E.L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<p['max_physics']
    def charge():counts[index+1]+=1;assert sum(counts)<=p['max_physics']
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            module,native,create,w,_=E.N.S.P.C.B.setup(folder/'native',E.N.science(p));original=copy.deepcopy(module.ACTIVE);E.L.install_stop_handlers();pipe.send(dict(state='READY',pid=os.getpid(),cpu=index))
            while True:
                req=pipe.recv()
                if req is None:break
                q=P.frozen(p,req['model']);q.sample=req.get('seed') is not None;q.rng=None if req.get('seed') is None else np.random.default_rng(np.random.SeedSequence(req['seed']));j=req['job']
                module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m']);before=counts[index+1]
                s=E.L.Session(j,native,create,w,guard,charge,q);s.hidden=torch.zeros((1,1,64),dtype=torch.float64);E.L.install_stop_handlers();labels=[]
                if req.get('cancel_probe'):
                    def cancel(physical=False):
                        if physical:raise E.L.StageStop('T449 zero-native cancellation')
                    s.guard=cancel
                    try:s.step();raise AssertionError('Cancellation missing')
                    except E.L.StageStop:pass
                    assert not s.rows and s.steps==0;s.close();s=None;pipe.send(dict(state='CANCEL_PASS',physics=0));continue
                query=req.get('query',False)
                if query:
                    assert req.get('seed') is None
                    if expert is None:expert=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False)
                while not s.done():
                    if query:
                        x=s.packet().copy()
                        with torch.no_grad():command=expert.base_actions(torch.tensor(x[None])).float()
                        labels.append(command[0].numpy().copy())
                        if len(s.rows)>=p['teacher_first']:s.step(command[0].numpy(),owner=1,actor_active=True)
                        else:s.step(owner=0)
                        assert np.array_equal(s.rows[-1][0],x)
                    else:s.step(owner=0)
                a=E.payload(s)
                if query:a['teacher_actions']=np.asarray(labels,np.float32)
                r=s.result();r.update(policy_hash=q.phash(),physics=counts[index+1]-before,array_digest=digest(a),array_hashes={k:digest(v) for k,v in a.items()},integration_sha256=digest(a['final_native_integration']),pid=os.getpid(),cpu=index,teacher=query,teacher_first=p['teacher_first'] if query else None)
                if req.get('full_path'):
                    path=Path(req['full_path']);path.parent.mkdir(parents=True,exist_ok=True);blob=E.N.R.blob(E.N.S.packed(a));assert len(blob)<=p['full_bytes']
                    if path.exists():assert path.read_bytes()==blob
                    else:path.write_bytes(blob)
                s.close();s=None;pipe.send((a,r))
    except BaseException:
        save(folder/'failure.json',dict(error=traceback.format_exc(),physics=counts[index+1]))
        if s is not None and s.rows:
            try:E.N.write_npz(folder/'partial.npz',E.N.S.packed(E.payload(s)),p['full_bytes'])
            except BaseException:save(folder/'partial_save_error.json',dict(error=traceback.format_exc()))
        try:pipe.send(dict(state='FAILED',error=traceback.format_exc()))
        except BaseException:pass
        raise
    finally:
        if s:s.close()
        if w:w.close()
        sys.stdout,sys.stderr=stdout,stderr;log.close();pipe.close()

class Pool(W.Pool):
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];ctx=R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(8)];assert all(v['state']=='READY' for v in self.ready)
            path=Path(p['out'])/'worker_generations.json';old=read(path) if path.exists() else [];old.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time()));save(path,old)
        except BaseException:self.close();raise

class Runtime(R.Runtime):
    def request(self,jobs,model,role,seeds=None,full=False,references=None,query=False):
        self.role=role;result=[]
        for offset in range(0,len(jobs),8):
            js=jobs[offset:offset+8];self.guard();assert sum(self.counts)+sum(j['case']['step_cap'] for j in js)<=self.p['max_physics'];assert self.commands+len(js)+16<=self.p['max_commands'];self.commands+=len(js)
            requests=[]
            for i,j in enumerate(js):
                path=(references[offset+i]['path'] if references is not None else str(OUT/'episodes'/f'{len(self.rows)+i:04d}.npz')) if full else None
                requests.append(dict(model=model['path'],job=j,seed=None if seeds is None else seeds[offset+i],full_path=path,query=query))
            for i,((a,r),j,req) in enumerate(zip(self.pool.request(requests),js,requests)):
                r.update(job_id=j['id'],role=role,seed=req['seed'],request=req,loaded_model_sha256=sha(model['path']));assert r['policy_hash']==model['policy_hash'] and int(a['native_counts'].sum())==r['physics']
                if references is not None:
                    previous=references[offset+i];assert r['array_digest']==previous['array_digest'] and r['integration_sha256']==previous['integration_sha256'];r.update(reference_path=previous['path'],exact=True,independent_new_sample=False);self.replays.append(r);log='replays.jsonl'
                else:
                    path=Path(req['full_path']) if full else OUT/'episodes'/f'{len(self.rows):04d}.npz';path.parent.mkdir(exist_ok=True)
                    if full:assert digest(E.N.S.arrays(path))==r['array_digest']
                    else:E.N.write_npz(path,{k:a[k] for k in ['actions','native_counts','end_flags']},self.p['compact_bytes'])
                    r.update(path=str(path),sha256=sha(path),full=full);self.rows.append(r);log='cases.jsonl'
                with (OUT/log).open('a') as f:f.write(json.dumps(r)+'\n')
                result.append((a,r))
        return result

def safe_ids(rows):return sorted(r['job_id'] for a,r in rows if r['success'] and not r['collision'])
def evaluate(rt,p,model,role):
    rows=rt.request(p['main_jobs']+p['old275_jobs'],model,role)
    return dict(model=model,wide54=R.M.tally([r for a,r in rows[:54]]),old275=R.M.tally([r for a,r in rows[54:]]),success_ids=safe_ids(rows)),[r for a,r in rows]
def admit(candidate,current):return set(current['success_ids'])<=set(candidate['success_ids'])
def events(phase,which):
    def event(where,k):
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([phase,which,where,k])+'\n')
    return event
def fit_fixed(rt,p,phase,obj,shadow,groups,order,corrections):
    trace=[];steps=p['tail_steps'] if phase=='tail' else p['recovery_steps']
    for k in range(steps):
        rt.role=phase.upper()+'_FIT';rt.guard(True);a=b=None
        for which,objects in [('live',obj),('shadow',shadow)]:
            event=events(phase,which);event('BEFORE',k);v=tail_step(objects,groups) if phase=='tail' else mix_step(objects,groups,order,corrections,k);event('AFTER',k)
            if which=='live':a=v
            else:b=v
        assert a==b;trace.append(a)
        if k%32==0:assert digest(P.state(*obj))==digest(P.state(*shadow));save(OUT/'fit_progress.json',dict(phase=phase,steps=k+1,latest=a,epoch=time.time()))
    assert digest(P.state(*obj))==digest(P.state(*shadow));save(OUT/f'{phase}_trace.json',trace);return trace

def try_gate(rt,p,current,baseline):
    if len(current['success_ids'])!=329:return dict(state='NOT_ADMITTED_DEVELOPMENT_FAILURE',fresh16=None,reload78=None,six_gate=False)
    # Reserve the entire independent78 full arrays, fresh compact, model copy,
    # and metadata before consuming any navigation holdout.
    required=78*p['full_bytes']+16*p['compact_bytes']+p['model_bytes']+1024**2
    if used(OUT)+required>p['max_bytes'] or time.time()+2400>rt.deadline:
        return dict(state='DEVELOPMENT_ALL_SAFE_GATE_RESOURCE_ADMISSION_DEFERRED',required_bytes=required,fresh16=None,reload78=None,six_gate=False)
    fresh=rt.request(p['fresh_jobs'],current['model'],'FRESH16');ft=R.M.tally([r for a,r in fresh])
    if ft['success']!=16:return dict(state='FRESH_FAILED',fresh16=ft,reload78=None,six_gate=False)
    # New worker processes, new serialized-model path, original physics and h0.
    rt.pool.close();rt.commands+=8;copy_path=OUT/'independent_readback.pt.gz';shutil.copyfile(current['model']['path'],copy_path);assert sha(copy_path)==current['model']['sha256']
    model=dict(current['model'],path=str(copy_path));rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard);rt.commands+=8
    byid={r['job_id']:r for r in baseline};byid.update({r['job_id']:r for a,r in fresh});jobs=p['old8_jobs']+p['main_jobs']+p['fresh_jobs']
    reloaded=rt.request(jobs,model,'INDEPENDENT_RELOAD78',full=True)
    for a,r in reloaded:
        previous=byid[r['job_id']];assert r['array_hashes']==previous['array_hashes'] and r['array_digest']==previous['array_digest'] and r['integration_sha256']==previous['integration_sha256']
        assert r['success'] and not r['collision']
    return dict(state='NATIVE_GATES_PASS_PENDING_INDEPENDENT_AUDIT_AND_BACKUP',fresh16=ft,reload78=dict(valid=78,exact=78,full_arrays_saved=True),six_gate=False,native_gate=True)

def main():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists();setup_runtime();E.N.offline();E.L.install_stop_handlers();rt=Runtime(p)
    results=[];rl=[];state='FAILED';error=None;gate=None;current=copy.deepcopy(p['night_initial_result']);baseline=[r for r in map(json.loads,Path(p['night_initial_cases_path']).read_text().splitlines()) if r['role']==p['night_initial_role']];correction_rows=[]
    try:
        data=dataset(p);groups=P.packs(data,'cuda');order=P.schedule(data,max(p['recovery_steps'],p['rounds']*p['epochs']));rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for phase in ['tail','recovery']:
            if len(current['success_ids'])==329:break
            if phase=='recovery':
                query=rt.request(p['query_jobs'],current['model'],'RECOVERY_QUERY',full=True,query=True);rt.request(p['query_jobs'],current['model'],'RECOVERY_QUERY_EXACT',full=True,references=[r for a,r in query],query=True)
                correction_rows=[r for a,r in query if r['success'] and not r['collision']];save(OUT/'corrections.json',correction_rows)
                if not correction_rows:results.append(dict(phase=phase,state='SKIPPED_NO_VALIDATED_RECOVERY',accepted=False));continue
            obj=initial(p);P.load(obj,P.checkpoint(current['model']['path'])['joint_state']);shadow=P.make(p);P.load(shadow,P.state(*obj))
            before=digest(P.state(*obj));corrections=P.packs([dict(id=r['job_id'],task=10,x=(a:=E.N.S.arrays(r['path']))['states'],y=a['teacher_actions']) for r in correction_rows],'cuda')
            fit_fixed(rt,p,phase,obj,shadow,groups,order,corrections);model=P.export(obj,OUT/f'{phase}.pt.gz',p['model_bytes']);candidate,rows=evaluate(rt,p,model,'FINAL_'+phase)
            accepted=admit(candidate,current);entry=dict(phase=phase,state='COMPLETE',initial_model=current['model'],initial_joint_hash=before,candidate=candidate,accepted=accepted,correction_rows=correction_rows if phase=='recovery' else [])
            results.append(entry);save(OUT/'phases.json',results)
            if accepted:current=candidate;baseline=rows
            del obj,shadow;torch.cuda.empty_cache()
        if len(current['success_ids'])!=329:
            obj=P.make(p);P.load(obj,P.checkpoint(current['model']['path'])['joint_state'])
            for g in obj[2].param_groups:g['lr']=p['actor_lr']
            shadow=P.make(p);P.load(shadow,P.state(*obj));initial_model=current['model'];panel_ids={j['id'] for j in p['panel_jobs']};panel_safe=set(current['success_ids'])&panel_ids;rejects=0
            for rnd,jobs in enumerate(p['night_training_jobs']):
                rt.guard(True);model=initial_model if rnd==0 else rl[-1]['retained_model'];before=P.state(*obj);assert obj[0].phash()==model['policy_hash']
                seeds=[p['seed']+rnd*10+i for i in range(4)];values=rt.request(jobs,model,f'PPO_TRAIN_{rnd}',seeds,full=True);rt.request(jobs,model,f'PPO_EXACT_{rnd}',seeds,full=True,references=[r for a,r in values])
                P.load(shadow,before);a=anchored_ppo(obj,[a for a,r in values],p,groups,order,rnd,lambda:rt.guard(True),events(f'ppo{rnd}','live'));b=anchored_ppo(shadow,[a for a,r in values],p,groups,order,rnd,lambda:rt.guard(True),events(f'ppo{rnd}','shadow'));assert a==b and digest(P.state(*obj))==digest(P.state(*shadow))
                candidate=P.export(obj,OUT/f'ppo_{rnd}.pt.gz',p['model_bytes']);panel=rt.request(p['panel_jobs'],candidate,f'PPO_PANEL_{rnd}');ss=set(safe_ids(panel));accepted=panel_safe<=ss
                if accepted:panel_safe=ss;rejects=0;retained=candidate
                else:P.load(obj,before);P.load(shadow,before);rejects+=1;retained=model
                rl.append(dict(round=rnd,model_before=model,state_before_hash=digest(before),proposal=candidate,fit=a,accepted=accepted,retained_model=retained,retained_joint_hash=digest(P.state(*obj)),panel_success_ids=sorted(ss),train_rows=[r for a,r in values]));save(OUT/'rounds.json',rl)
                if rejects>=2:break
            candidate,rows=evaluate(rt,p,rl[-1]['retained_model'],'FINAL_anchored_PPO');accepted=admit(candidate,current)
            results.append(dict(phase='anchored_PPO',state='COMPLETE',initial_model=initial_model,candidate=candidate,accepted=accepted));save(OUT/'phases.json',results)
            if accepted:current=candidate;baseline=rows
            del obj,shadow;torch.cuda.empty_cache()
        gate=try_gate(rt,p,current,baseline);state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,physics=sum(rt.counts),commands=rt.commands,epoch=time.time()))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands+p['preflight_commands'],phases=results,rounds=rl,selected=current,gate=gate,error=error,all_native_workers_stopped=True,eight_started=False))
        save(OUT/'status.json',dict(state=state,epoch=time.time()));R.M.fcntl.flock(rt.lock,R.M.fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error
    print('T449_MAIN_COMPLETE',flush=True)

def cold_candidates(protected,required):
    remote={};doc=ROOT/'stage/docs/long_distance_20261101'
    for file in sorted(doc.glob('release_receipt_*.json')):
        r=read(file)
        if r.get('state')!='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' or 'group' not in r:continue
        mf=doc/f"source_manifest_{r['group']}.json"
        if not mf.exists() or sha(mf)!=r['manifest_sha256']:continue
        for row in read(mf)['files']:remote[row['path']]=(row,file,mf)
    candidates=[]
    for f in (ROOT/'stage/runs').rglob('*'):
        if not f.is_file() or f.suffix not in ['.json','.jsonl','.log','.txt'] or f.stat().st_size<10000 or str(f) in protected or f.with_name(f.name+'.cold.gz').exists() or OUT in f.parents:continue
        rel=str(f.relative_to(ROOT))
        if rel not in remote:continue
        r,receipt,mf=remote[rel];raw=f.read_bytes()
        if len(raw)!=r['bytes'] or sha(f)!=r['sha256']:continue
        blob=gzip.compress(raw,compresslevel=9,mtime=0);saving=len(raw)-len(blob)
        if saving<=0:continue
        candidates.append(dict(original=rel,encoded=rel+'.cold.gz',original_bytes=len(raw),original_sha256=sha(f),encoded_bytes=len(blob),encoded_sha256=Z.sha_bytes(blob),saving=saving,receipt=str(receipt.relative_to(ROOT)),receipt_sha256=sha(receipt),manifest=str(mf.relative_to(ROOT)),manifest_sha256=sha(mf)))
    candidates.sort(key=lambda r:(-r['saving'],r['original']));selected=[];saving=0
    for r in candidates:
        selected.append(r);saving+=r['saving']
        if saving>=required:break
    assert saving>=required and len(selected)<=500,(saving,required,len(selected));return selected,saving

def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t448_full_gradient_control';assert read(prior/'audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T448.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert not Path('/proc',str(read(prior/'driver_started.json')['pid'])).exists();base=read(prior/'protocol.json');s=read(prior/'summary.json')
    chosen=max(base['arms'],key=lambda name:(s['results'][name]['wide54']['success']+s['results'][name]['old275']['success'],s['results'][name]['old275']['success'],-base['arms'].index(name)))
    p=copy.deepcopy(base);p.update(id='T449_OFFLINE_TAIL_RECOVERY_ANCHORED_PPO',out=str(OUT),night_initial=s['results'][chosen]['model'],night_initial_result=s['results'][chosen],
        night_initial_cases_path=str(prior/'cases.jsonl'),night_initial_role='FINAL_'+chosen,
        tail_steps=1024,recovery_steps=1500,rounds=8,epochs=4,teacher_first=5,seed=202611094490,
        max_bytes=48*2**20,full_bytes=448*1024,compact_bytes=4*1024,model_bytes=int(1.5*2**20),max_physics=40000000,max_commands=2000,
        main_seconds=18000,learning_seconds=14400,audit_seconds=5400,backup_seconds=600,all_seconds=40000,
        absolute_learning_stop=1794279600.,absolute_native_stop=1794282300.,absolute_all_stop=1794283400.,
        preflight_commands=24,preflight_BC_discarded_Adam=8,preflight_PPO_discarded_Actor_Adam=8,preflight_PPO_discarded_V_Adam=8,
        fresh16='conditional registered16 unconsumed +/-0.0125m XY and +/-0.5deg from wide80s original centre',reload78='conditional new processes and serialized-model path, full arrays exact',
        six_gate=False,eight_started=False,user_authorization='Nov9 user: check and begin tonight offline unattended relay, old rules Nov10 08:30 hardstop.',
        training='Tail upper20%+mean balanced BC1024; successful actual-student-prefix safeT255 recoveries8+8exact and mixed BC1500; own4-episode PPO+original219demo anchor,8rounds max32Actor32V live and same shadow.',
        parent_comparison='T448 full chosen310/329 not better than historical321/329; no promotion. Each whole phase must preserve ALL initial full329 success IDs or fallback.',
        mechanism_scope='Joint finite engineering relay, no single-module causal/novelty/safety claims; BC and PPO counts separate; teacher mixtures never onpolicy.',
        original_T448_and_all_ended_directories_NOT_restarted=True)
    # Absolute BJT Nov10 07:00 learning,07:45 native,08:30 all, derived explicitly.
    import datetime
    tz=datetime.timezone(datetime.timedelta(hours=8))
    for key,h,m in [('absolute_learning_stop',7,0),('absolute_native_stop',7,45),('absolute_all_stop',8,30)]:p[key]=datetime.datetime(2026,11,10,h,m,tzinfo=tz).timestamp()
    byid={j['id']:j for j in p['old275_jobs']+p['main_jobs']};p['query_jobs']=[byid[f'dev_task6_{i:02d}'] for i in [0,3,6,9,15,18,21,24]]
    p['panel_jobs']=p['old8_jobs']+p['query_jobs']+p['teacher_jobs'];assert len({j['id'] for j in p['panel_jobs']})==24
    p['night_training_jobs']=[[p['query_jobs'][r],p['old8_jobs'][[0,1,2,3,4,5,7,0][r]],byid[f'wide117_80_dev6_LEFT_{[3,24,9,18,1,7,20,26][r]:02d}'],byid[f'wide117_80_dev6_RIGHT_{[3,24,9,18,1,7,20,26][r]:02d}']] for r in range(8)]
    p['fresh_jobs']=[]
    import itertools
    for side in ['LEFT','RIGHT']:
        center=byid[f'wide117_80_dev6_{side}_13']
        for i,(dx,dy,yaw) in enumerate(itertools.product([-.0125,.0125],[-.0125,.0125],[-.5,.5])):
            j=copy.deepcopy(center);j.update(id=f't449_fresh6_{side}_{i}',source_reset_id=f't449_fresh6_{side}_{i}',delta_x_m=dx,delta_y_m=dy,delta_yaw_deg=yaw);p['fresh_jobs'].append(j)
    p['worst_physics']=3*sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])+2*sum(j['case']['step_cap'] for j in p['query_jobs'])+2*sum(j['case']['step_cap'] for js in p['night_training_jobs'] for j in js)+8*sum(j['case']['step_cap'] for j in p['panel_jobs'])+sum(j['case']['step_cap'] for j in p['fresh_jobs'])+sum(j['case']['step_cap'] for j in p['old8_jobs']+p['main_jobs']+p['fresh_jobs'])
    assert p['worst_physics']<=p['max_physics']
    protected=set(base['inputs'])|set(base['constructor_reads'])|{str(prior/f) for f in ['protocol.json','summary.json','audit.json','cases.jsonl']}|{p['night_initial']['path']}
    cold,saving=cold_candidates(protected,max(1,used()+p['max_bytes']+1024**2-2**30));p['cold_storage']=cold;p['constructor_reads']=sorted(protected)
    p['protected_files']={**base['protected_files'],p['night_initial']['path']:p['night_initial']['sha256']}
    files=protected|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'PLAN.md'),str(DOC/'SOURCES.md')}
    for r in cold:files|={str(ROOT/r['receipt']),str(ROOT/r['manifest'])}
    assert not {str(ROOT/r['original']) for r in cold}&files;p['inputs']={f:sha(f) for f in sorted(files)}
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',max_bytes=p['max_bytes'],cold_files=len(cold),cold_payload_saving=saving,global_full_after_encoding=used()-used(OUT)-saving+p['max_bytes'],max_physics=p['max_physics'],worst_physics=p['worst_physics'],max_commands=p['max_commands'],maximum_pipe_commands=1500,
        main_seconds=p['main_seconds'],learning_seconds=p['learning_seconds'],audit_seconds=p['audit_seconds'],all_stop_BJT='2026-11-10 08:30:00',maximum_BC_live=2524,maximum_BC_shadow=2524,maximum_PPO_live_Actor=32,maximum_PPO_live_V=32,maximum_independent_historical_Adam=2588,
        own_reserve='32training full*448KiB=14MiB;8recovery full*448KiB=3.5MiB;10models*1.5MiB=15MiB;1400compact*4KiB~5.47MiB;metadata10MiB. Fullgate only if whole78full+model+metadata fits unused48MiB and2400s BEFORE fresh16. No capacity enlargement.'))
    assert 32*p['full_bytes']+8*p['full_bytes']+10*p['model_bytes']+1400*p['compact_bytes']+10*2**20<=p['max_bytes']
    print('T449_REGISTERED',len(cold),saving,p['worst_physics'],flush=True)
def encode():
    p=read(OUT/'protocol.json');Z.OUT=OUT;Z.encode(p)
def storage_audit():
    p=read(OUT/'protocol.json');Z.OUT=OUT;Z.audit(p);verify(p,True)
def load_original(relative):
    """Decode an explicitly cold-stored text to its original bytes without disk restore."""
    Z.OUT=OUT;return Z.load_original(relative)

def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'storage_audit.json')['state']=='PASS' and not (OUT/'preflight.json').exists();setup_runtime();data=dataset(p);groups=P.packs(data,'cuda');order=P.schedule(data,1500)
    obj=initial(p);shadow=initial(p);before=P.state(*obj);results={};times={}
    for phase in ['tail','recovery']:
        P.load(obj,before);P.load(shadow,before);start=time.monotonic();trace=[]
        corrections=[groups[0]]
        for k in range(2):
            a=tail_step(obj,groups) if phase=='tail' else mix_step(obj,groups,order,corrections,k);b=tail_step(shadow,groups) if phase=='tail' else mix_step(shadow,groups,order,corrections,k);assert a==b;trace.append(a)
        torch.cuda.synchronize();times[phase]=(time.monotonic()-start)/2;assert digest(P.state(*obj))==digest(P.state(*shadow));results[phase]=trace
    P.load(obj,before);P.load(shadow,before);synthetic=R.synthetic(obj,p)
    a=anchored_ppo(obj,synthetic,p,groups,order,0,lambda:None,lambda *_:None);b=anchored_ppo(shadow,synthetic,p,groups,order,0,lambda:None,lambda *_:None);assert a==b and digest(P.state(*obj))==digest(P.state(*shadow))
    mem=io.BytesIO();torch.save(dict(joint_state=P.state(*obj),policy_hash=obj[0].phash(),joint_hash=digest(P.state(*obj)),architecture='plain120_128_64GRU_2',single_actor=True),mem)
    mature_model_bytes=len(gzip.compress(mem.getvalue(),mtime=0));assert mature_model_bytes<=p['model_bytes'],(mature_model_bytes,p['model_bytes'])
    P.load(obj,before);assert digest(P.state(*obj))==digest(before)
    expert=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False);g=next(g for g in data if g['id']=='dev_task6_00')
    with torch.no_grad():labels=expert.base_actions(torch.tensor(g['x'][:10])).float().numpy()
    assert np.array_equal(labels,g['y'][:10]),'Safe teacher must match original T255 action function'
    xx=torch.tensor(data[0]['x'][:12]);q=P.frozen(p,p['night_initial']['path']);h=None;loc=[]
    with torch.no_grad():
        for x in xx:_,m,h,_=q.step(x[None],h);loc.append(m[0])
        err=float((torch.stack(loc)-q.sequence(xx)).abs().max());assert err<1e-10
    estimate=times['tail']*1024+times['recovery']*1500;assert estimate<8000
    counts=R.M.mp.get_context('spawn').RawArray('q',9);folder=OUT/'preflight_native';folder.mkdir();pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:responses=pool.request([dict(model=p['night_initial']['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs']]);assert all(v['state']=='CANCEL_PASS' for v in responses) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=24,BC_discarded_Adam=8,PPO_discarded_Actor_Adam=a['Actor_Adam']+b['Actor_Adam'],PPO_discarded_V_Adam=a['V_Adam']+b['V_Adam'],mature_restore_and_shadow_exact=True,teacher_first10_original_actions_exact=True,CPU_history_error=err,estimated_BC_live_shadow_seconds=estimate,mature_two_Adam_serialized_model_bytes=mature_model_bytes,synthetic_correction_NOT_navigation_data=True,all_workers_stopped=True))
    print('T449_PREFLIGHT_PASS',estimate,flush=True)

def audit_full_history(p,r,a):
    q=P.frozen(p,r['request']['model']);q.rng=None if r['seed'] is None else np.random.default_rng(np.random.SeedSequence(r['seed']));q.reset_noise();h=None
    with torch.no_grad():
        for i,x in enumerate(a['states']):
            q.sample=bool(a['stochastic'][i]);hb=np.zeros(64) if h is None else h[0,0].numpy().copy();command,loc,h,_=q.step(torch.tensor(x[None]),h)
            assert np.array_equal(hb,a['hidden_before'][i])
            for k,v in q.tape[-1].items():assert np.array_equal(v,a[k][i]),(i,k)
            if not a['ownership'][i]:assert np.array_equal(command[0].numpy(),a['actions'][i])
            else:assert r['teacher'] and np.array_equal(a['teacher_actions'][i],a['actions'][i])
            assert np.abs(E.N.K.H.rebuild(x,a['poses'][i],a['history_poses'][i],a['task_goal_xy'],a['history_goals'][i])-x).max()<=2e-6
            n=int(a['native_counts'][i]);z=E.N.K.R.GAMMA**n;f=a['end_flags'][i];d0=np.linalg.norm(a['poses'][i,:2]-a['task_goal_xy']);d1=np.linalg.norm(a['next_poses'][i,:2]-a['task_goal_xy'])
            reward=-.000002*(1-z)/(1-E.N.K.R.GAMMA)+.03*d0-z*.03*d1*(not f.any())
            if f.any():reward+=E.N.K.R.GAMMA**(n-1)*(float(f[1])-float(f[0])-.5*float(f[2]))
            assert abs(reward-a['rewards'][i])<1e-12
    assert np.array_equal(h[0,0].numpy(),a['last_hidden_after'])
    if r['teacher']:assert not a['ownership'][:p['teacher_first']].any() and a['ownership'][p['teacher_first']:].all() and not a['stochastic'].any()

def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();ids=[s['pid']]+[pid for g in read(OUT/'worker_generations.json') for pid in g['workers']];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    data=dataset(p);groups=P.packs(data,'cuda');order=P.schedule(data,max(p['recovery_steps'],p['rounds']*p['epochs']));adam=0
    for entry in s['phases']:
        if entry['phase'] not in ['tail','recovery'] or entry['state']!='COMPLETE':continue
        obj=P.make(p);P.load(obj,P.checkpoint(entry['initial_model']['path'])['joint_state']);assert digest(P.state(*obj))==entry['initial_joint_hash'];trace=[]
        corr=P.packs([dict(id=r['job_id'],task=10,x=(a:=E.N.S.arrays(r['path']))['states'],y=a['teacher_actions']) for r in entry['correction_rows']],'cuda')
        for k in range(p['tail_steps'] if entry['phase']=='tail' else p['recovery_steps']):trace.append(tail_step(obj,groups) if entry['phase']=='tail' else mix_step(obj,groups,order,corr,k));adam+=1
        assert trace==read(OUT/f"{entry['phase']}_trace.json") and digest(P.state(*obj))==entry['candidate']['model']['joint_hash'];del obj
    if s['rounds']:
        first=s['rounds'][0];obj=P.make(p);P.load(obj,P.checkpoint(first['model_before']['path'])['joint_state'])
        for g in obj[2].param_groups:g['lr']=p['actor_lr']
        for r in s['rounds']:
            before=P.state(*obj);assert digest(before)==r['state_before_hash'];arrays=[E.N.S.arrays(v['path']) for v in r['train_rows']];out=anchored_ppo(obj,arrays,p,groups,order,r['round'],lambda:None,lambda *_:None);adam+=out['Actor_Adam']+out['V_Adam'];assert out==r['fit'] and digest(P.state(*obj))==r['proposal']['joint_hash']
            if not r['accepted']:P.load(obj,before)
            assert digest(P.state(*obj))==r['retained_joint_hash']
    rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()];replays=[json.loads(x) for x in (OUT/'replays.jsonl').read_text().splitlines()] if (OUT/'replays.jsonl').exists() else []
    generations=read(OUT/'worker_generations.json');assert sum(r['physics'] for r in rows+replays)==s['physics'] and s['physics']<=p['max_physics'] and s['commands']==p['preflight_commands']+16*len(generations)+len(rows)+len(replays)<=p['max_commands']
    expert=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False)
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert int(a['native_counts'].sum())==r['physics'];assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        if r['full']:
            assert digest(a)==r['array_digest'] and all(digest(a[k])==v for k,v in r['array_hashes'].items())
            audit_full_history(p,r,a)
            if r['teacher']:
                with torch.no_grad():lab=expert.base_actions(torch.tensor(a['states'])).float().numpy()
                assert np.array_equal(lab,a['teacher_actions'])
    bypath={r['path']:r for r in rows}
    for r in replays:assert r['array_digest']==bypath[r['reference_path']]['array_digest'] and r['integration_sha256']==bypath[r['reference_path']]['integration_sha256']
    for e in s['phases']:
        if e.get('candidate'):assert sha(e['candidate']['model']['path'])==e['candidate']['model']['sha256']
    ev=[json.loads(x) for x in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for e in s['phases']:
        if e['phase'] in ['tail','recovery'] and e['state']=='COMPLETE':
            count=p['tail_steps'] if e['phase']=='tail' else p['recovery_steps']
            for which in ['live','shadow']:assert [v[2:] for v in ev if v[:2]==[e['phase'],which]]==[[where,k] for k in range(count) for where in ['BEFORE','AFTER']]
    for r in s['rounds']:
        for which in ['live','shadow']:assert [v[2:] for v in ev if v[:2]==[f"ppo{r['round']}",which]]==[[where,k] for k in range(r['fit']['Actor_Adam']) for where in ['BEFORE','AFTER']]
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Adam=adam,physics=s['physics'],commands=s['commands'],all_full_arrays_and_actual_teacher_labels_and_density_rebuilt=True,all_live_Actor_V_Adam_reconstructed=True,compact329_full120_not_reconstructed=True,gate=s['gate']))
    print('T449_AUDIT_PASS',adam,flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();now=time.time();end=min(now+p['all_seconds'],p['absolute_all_stop']);assert now+p['main_seconds']+p['audit_seconds']+900<end
    with (OUT/'driver_console.log').open('xb') as f:c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T449','--why=Registered offline finite scientific relay','timeout','--signal=TERM','--kill-after=2',str(int(end-now)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,learning_stop=min(now+p['learning_seconds'],p['absolute_learning_stop']),native_stop=min(now+p['main_seconds'],p['absolute_native_stop']),all_stop=end,source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json')))
    print('T449_ACTUALLY_LAUNCHED',c.pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

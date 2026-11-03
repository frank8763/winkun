from t315_core import *
import argparse,signal

def preflight():
    assert not any((OUT/n).exists() for n in ['preflight.json','status.json','launch.json'])
    p=read(HERE/'protocol.json');verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    A=torch.tensor([[1.,2.,0.,1.],[0.,1.,2.,-1.],[1.,0.,-1.,2.]],dtype=torch.float64);g=torch.tensor([.2,-.3,.4,.1],dtype=torch.float64)
    M=A.T@A/3+p['fisher_damping']*torch.eye(4,dtype=torch.float64)
    x,h=cg(lambda v:M@v,g,10,1e-24);assert torch.max(torch.abs(x-torch.linalg.solve(M,g)))<1e-10
    items,fitted=baseline_batch(p);q=K.policy(p);before=K.S.digest(q.state_dict());v=flat(q);put(q,v);assert before==K.S.digest(q.state_dict())
    parent=copy.deepcopy(q).eval().requires_grad_(False);ax,ay,offset=K.T.anchors(p,parent)
    rng=np.random.default_rng(p['seed']);indices=[rng.integers(0,len(i['returns']),16) for i in items['mc']]
    old_indices=np.concatenate([rng.integers(offset[k],offset[k+1],16) for k in range(8)])
    batch,target=fitted['mc'];g,gr=full_gradient(p,q,batch,target,'mc',ax,ay,old_indices,lambda:None)
    ref=read(DOC/'T314_FINDINGS_GRADIENT_REFERENCE.json');relative=abs(float(g.norm())-ref['mc_full_gradient_norm'])/max(1.,ref['mc_full_gradient_norm']);assert relative<=1e-5
    # Verify the exact ratio1 first-order identity against the actual clip loss.
    params=[z for z in q.parameters() if z.requires_grad];a=batch[0]['arrays'];t=target[0];scale=max(.05,float(np.sqrt(np.mean([np.mean(i['returns']**2) for i in batch]))))
    coefficient=torch.tensor(t['mc']*t['prefix']/scale)
    value=K.lp(q,a);ratio=(value-torch.tensor(a['logp'])).exp();clipped=torch.minimum(ratio*coefficient,ratio.clamp(.8,1.2)*coefficient).mean()
    def vector(loss):
        gg=torch.autograd.grad(loss,params,allow_unused=True)
        return torch.cat([torch.zeros_like(z).reshape(-1) if v is None else v.detach().reshape(-1) for z,v in zip(params,gg)]).double()
    direct=vector(clipped);identity=vector(torch.dot(K.lp(q,a),coefficient)/len(coefficient));error=float((direct-identity).norm())/max(1.,float(direct.norm()))
    assert error<=1e-5 and before==K.S.digest(q.state_dict()) and all(z.grad is None for z in params)
    K.C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,Adam_calls=0,Actor_parameter_proposals=0,autograd_calls=19,synthetic_SPD_CG_exact=True,
        full_gradient_norm_reference_relative_error=relative,real_trajectory_clip_identity_relative_error=error,all_full_batch_logp_errors_zero=True,pack_restore_Actor_exact=True,protocol_sha256=K.C.sha(HERE/'protocol.json'),
        full_worstcase_reserved_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']))
    print('T315 full-gradient preflight PASS: all density/clip identity/reference,0native0Adam0parameterproposal')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');start=time.time();meter=None;events=[];models={};state='FAILED';result=None
    def emit(kind,payload):
        r=dict(kind=kind,payload=payload);events.append(r)
        with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')
        K.C.save(OUT/'status.json',dict(state=kind,pid=os.getpid(),main_epoch=start,updated_epoch=time.time(),deadline_epoch=start+p['wall_seconds'],events=len(events),native_steps=0))
    def guard():
        assert time.time()<start+p['wall_seconds']-15
        if meter:meter.hold()
    def save_model(q,arm,name,accepted):
        payload=dict(policy=q.state_dict(),arm=arm,method=name,offline_only=True,accepted=accepted,protocol_sha256=K.C.sha(HERE/'protocol.json'))
        b=io.BytesIO();torch.save(payload,b);blob=gzip.compress(b.getvalue(),compresslevel=6,mtime=0);assert len(blob)<=p['model_bytes'];meter.charge_payload(len(blob))
        f=OUT/(arm+'_'+name+'_offline_policy.pt.gz');assert not f.exists();f.write_bytes(blob)
        r=dict(path=str(f),sha256=K.C.sha(f),policy_hash=K.S.digest(q.state_dict()),bytes=len(blob));models[arm+'_'+name]=r;return r
    def alarm(*_):raise TimeoutError('T315 bounded offline deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds'])
    try:
        verify(p);assert read(OUT/'preflight.json')['state']=='PASS';torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        meter=K.Budget(ROOT,OUT,p,start_epoch=start);emit('STARTING',dict(no_native=True,finite_offline_proposals=True))
        result=science(p,guard,emit,save_model);K.C.save(OUT/'metrics.json',result);meter.checkpoint();state='COMPLETE'
    except BaseException:K.C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),events=len(events),original_runs_untouched=True))
    finally:
        signal.alarm(0)
        if meter:meter.close()
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,deadline_epoch=start+p['wall_seconds'],all_workers_stopped=True,workers_spawned=0,
        commands=0,physics_steps=0,new_V_Adam_calls=0,historical_Adam_calls=0,new_full_gradient_Adam_calls=sum(e['kind']=='new_full_gradient_Adam_complete' for e in events),
        offline_parameter_proposals=sum(e['kind']=='offline_parameter_proposal' for e in events),models=models,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'),
        empirical_fisher_viable_both_arms=None if result is None else result['empirical_fisher_viable_both_arms'],Adam_viable_both_arms=None if result is None else result['Adam_viable_both_arms'],
        old275_six54_fresh16_reload78_executed=False,six_metre_stable=False,deployed_Actor_changed=False,seven_eight_entered=False,automatic_native_training=False)
    K.C.save(OUT/'summary.json',s);K.C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)

def audit():
    dest=DOC/'T315_AUDIT.json';assert not dest.exists();p=read(HERE/'protocol.json');s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Process still present',pid))
    verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();events=[]
    def guard():assert time.time()<start+p['audit_seconds']
    def save(q,arm,name,accepted):
        ref=s['models'][arm+'_'+name];f=Path(ref['path']);assert K.C.sha(f)==ref['sha256']
        saved=torch.load(io.BytesIO(gzip.decompress(f.read_bytes())),map_location='cpu',weights_only=False)
        assert K.S.digest(saved['policy'])==K.S.digest(q.state_dict())==ref['policy_hash'] and saved['accepted']==accepted
        return ref
    result=science(p,guard,lambda kind,payload:events.append(dict(kind=kind,payload=payload)),save)
    assert result==read(OUT/'metrics.json') and events==[json.loads(v) for v in (OUT/'events.jsonl').read_text().splitlines() if json.loads(v)['kind']!='STARTING']
    assert s['historical_Adam_calls']==0 and s['new_full_gradient_Adam_calls']==2 and s['offline_parameter_proposals']==result['total_parameter_proposals']<=32
    K.C.save(dest,dict(state='PASS',native_added=0,disposable_reconstruction_of_new_Adam_calls=2,offline_parameter_proposals_reconstructed=s['offline_parameter_proposals'],
        all_score_Fisher_CG_trials_objectives_and_model_SHA_exact=True,all_source_and_input_SHA_exact=True,all_original_processes_absent=True,
        summary_sha256=K.C.sha(OUT/'summary.json'),metrics_sha256=K.C.sha(OUT/'metrics.json'),elapsed_s=time.time()-start))
    print('T315 independent audit PASS: all fullgrad/Adam/Fisher/trials/models exact,0native')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','main','audit']);args=parser.parse_args();globals()[args.mode]()

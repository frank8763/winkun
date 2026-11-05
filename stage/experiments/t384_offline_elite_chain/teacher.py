from night_common import *

def main(outpath):
    offline();out=Path(outpath);p=read(out/'protocol.json');configure(out);base=science(p)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not (out/'summary.json').exists();start=time.time();deadline=min(start+p['main_seconds'],p['native_deadline'])
    physical=commands=0;last=0.;phase='START';state='FAILED';s=w=None;results=[];pred=compact(p);reference=None;active=None
    def guard(physical=False):
        nonlocal last
        assert time.time()<deadline-10 and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            last=time.time();capacity(out,p['final_bytes']);save(out/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=total(),commands=commands,deadline=deadline))
    def total():return physical
    def charge():
        nonlocal physical
        physical+=1;assert physical<=p['max_physics']
    def command():
        nonlocal commands
        commands+=1;guard()
    def persist():
        if s and s.rows:write_npz(out/'mixed.npz',S.packed(s.mixed()),512*1024)
    def query(k,j,controls,rep,snapshot,old=None):
        nonlocal active
        command();s.restore(snapshot);hds=[];first=None;active=dict(k=k,j=j,start_rows=len(s.rows),physics_start=physical)
        if rep==0:pred['controls'][k,j]=controls;write_npz(out/'predictions.npz',pred,1792*1024)
        for t in range(20):
            row,hd=s.step(controls[t],owner=1,actor_active=False);hds.append(hd)
            if t==0:first=(copy.deepcopy(row),copy.deepcopy(w.get(s.env)),K.R.integration(s.env).copy())
            if rep==0:pred['next_poses'][k,j,t]=row[5];pred['counts'][k,j,t]=row[2];pred['flags'][k,j,t]=row[6]
            else:assert equal(pred['next_poses'][k,j,t],row[5]) and pred['counts'][k,j,t]==row[2] and equal(pred['flags'][k,j,t],np.asarray(row[6],np.uint8))
            if s.done():break
        n=t+1;values=dict(controls=controls,lengths=np.uint8(n),costs=np.float64(R.cost(controls,n,s.info.get('collision'),float(s.env._goal_state()[1]))),remaining_m=np.float64(s.env._goal_state()[1]),
            trace_sha256=np.frombuffer(bytes.fromhex(digest(np.asarray(hds,np.uint8))),np.uint8),integration_sha256=np.frombuffer(bytes.fromhex(digest(K.R.integration(s.env))),np.uint8))
        if rep==0:
            for key,v in values.items():pred[key][k,j]=v
            write_npz(out/'predictions.npz',pred,1792*1024)
        else:
            for key,v in values.items():assert equal(pred[key][k,j],v),(k,j,key)
        if old is not None:
            for key,v in values.items():assert equal(old[key][54,j],v),('T382 first query exact',key,j)
            assert equal(old['counts'][54,j],pred['counts'][k,j]) and equal(old['flags'][54,j],pred['flags'][k,j]) and equal(old['next_poses'][54,j],pred['next_poses'][k,j])
        active=None;return first
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Teacher external stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Teacher deadline')));signal.alarm(max(1,int(deadline-start)-5))
    save(out/'workers.json',dict(parent=os.getpid(),workers=[]))
    try:
        native,create,w,job=R.setup(base);oldactual=S.arrays(OLD/'mixed.npz');oldpred=load_npz(OLD/'predictions.npz');human=S.arrays(base['demonstration'])['actions']
        assert len(oldactual['states'])==94 and not oldactual['end_flags'].any()
        for rep in range(2):
            phase=f'PASS{rep}_HISTORICAL_PREFIX94';command();s=R.Session(job,native,create,w,guard,charge,R.actor(base))
            for i in range(94):
                row,hd=s.step(None if i<40 else oldactual['actions'][i],owner=0 if i<40 else 1)
                assert equal(hd,oldactual['hold_sha256'][i])
                for z,name in enumerate(NAMES):assert equal(np.asarray(row[z]),oldactual[name][i])
            assert hashes(s.mixed())==hashes(oldactual) and s.steps==9400
            assert bytes.fromhex(C.snapshot_hash(s))==oldpred['snapshot_sha256'][54].tobytes()
            snapshot=s.capture();start_hash=C.snapshot_hash(s);nominal=oldpred['controls'][54,0].copy();stop='MAX_REPLANS';accepted=attempts=0
            if rep==0:persist()
            for k in range(p['max_replans']):
                phase=f'PASS{rep}_{p["mechanism"]}_REPLAN{k}';snapshot=s.capture();ph=np.frombuffer(bytes.fromhex(C.snapshot_hash(s)),np.uint8)
                if rep==0:pred['snapshot_sha256'][k]=ph
                else:assert equal(ph,pred['snapshot_sha256'][k])
                if k==0:controls=oldpred['controls'][54,:5].copy()
                else:
                    if p['mechanism']=='HUMAN_ANCHOR':nominal=refseq(human,94+k)
                    controls,_=C.proposal(nominal,54+k,p)
                attempts+=1
                for j in range(5):query(k,j,controls[j],rep,snapshot,oldpred if k==0 else None)
                selected=choose(pred['costs'][k],pred['flags'][k],pred['lengths'][k]);s.restore(snapshot)
                if selected is None:stop='NO_FEASIBLE_TEACHER';break
                if rep==0:pred['selected'][k]=selected
                else:assert pred['selected'][k]==selected
                chosen=controls[selected].copy();first=query(k,5,chosen,rep,snapshot);s.restore(snapshot)
                n=int(pred['lengths'][k,5]);assert not pred['flags'][k,5,n-1,0] and (n==20 or pred['flags'][k,5,n-1,1]),'Selected exact validation contract failed'
                command();row,hd=s.step(chosen[0],owner=1);accepted+=1
                assert all(equal(np.asarray(a),np.asarray(b)) for a,b in zip(row,first[0])) and all(equal(a,b) for a,b in zip(w.get(s.env),first[1])) and equal(K.R.integration(s.env),first[2])
                if rep==0:persist();write_npz(out/'predictions.npz',pred,1792*1024)
                else:assert len(s.rows)<=len(reference['states']) and equal(hd,reference['hold_sha256'][len(s.rows)-1])
                if s.done():stop='ACTUAL_TERMINAL';break
                nominal=np.concatenate([chosen[1:],human[min(94+k+20,len(human)-1):min(94+k+20,len(human)-1)+1]])
            a=s.mixed();actual=s.result()
            if rep==0:persist();reference=copy.deepcopy(a)
            else:assert hashes(a)==hashes(reference) and all(equal(v,reference[key]) for key,v in a.items())
            results.append(dict(pass_index=rep,mechanism=p['mechanism'],prefix94_historical_exact=True,prefix_snapshot=start_hash,attempts=attempts,accepted=accepted,stop=stop,mixed=actual,
                mixed_hashes=hashes(a),prediction_hashes=hashes(pred),first_locked_candidate_set_exact=True,independent_sample=rep==0))
            save(out/'passes.json',results);s.close();s=None
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if s:
            try:
                partial=dict(packet=s.packet(),integration=K.R.integration(s.env),history_poses=w.get(s.env)[0],history_goals=w.get(s.env)[1],history_times=w.get(s.env)[2],steps=np.asarray(s.steps),all_physics=np.asarray(physical))
                if active:
                    branch=s.rows[active['start_rows']:]
                    for j,key in enumerate(NAMES):partial['branch_'+key]=np.asarray([row[j] for row in branch])
                    history=s.hist[active['start_rows']:]+[w.get(s.env)]
                    partial.update(branch_history_poses=np.asarray([v[0] for v in history]),branch_history_goals=np.asarray([v[1] for v in history]),branch_history_times=np.asarray([v[2] for v in history]))
                write_npz(out/'interruption.npz',partial,64*1024)
            except BaseException:save(out/'partial_error.json',dict(error=traceback.format_exc()))
        save(out/'failure.json',dict(error=error,phase=phase,active=active,physics=physical,commands=commands))
    finally:
        if s:s.close()
        if w:w.close()
        signal.alarm(0)
        summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=physical,commands=commands,Actor=0,optimizer=0,passes=results,all_workers_stopped=True,
            teacher_safe_GOAL=bool(state=='COMPLETE' and results[0]['mixed']['success']),autonomous_six_stable=False,old275=None,originalsix54=None,fresh16=None,reload78=None,
            files={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name not in ['main.log','status.json']})
        save(out/'summary.json',summary);save(out/'status.json',dict(state=state,updated_epoch=time.time(),physics=physical,commands=commands));capacity(out,p['final_bytes']);print(json.dumps(dict(state=state,physics=physical,commands=commands)))
    assert state=='COMPLETE','Engineering/administrative failure stops entire chain'

def audit(outpath):
    offline();out=Path(outpath);p=read(out/'protocol.json');configure(out);base=science(p);s=read(out/'summary.json');stopped(out);assert s['state']=='COMPLETE' and not (out/'audit.json').exists()
    a=S.arrays(out/'mixed.npz');pred=load_npz(out/'predictions.npz');old=S.arrays(OLD/'mixed.npz');oldp=load_npz(OLD/'predictions.npz');human=S.arrays(base['demonstration'])['actions'];f=s['passes'][0];g=s['passes'][1]
    assert all(equal(a[key][:94],old[key]) for key in NAMES) and f['stop']==g['stop'] and f['accepted']==g['accepted'] and f['attempts']==g['attempts']
    nominal=oldp['controls'][54,0];accepted=0;cmd=1
    for k in range(f['attempts']):
        if k==0:controls=oldp['controls'][54,:5]
        else:
            if p['mechanism']=='HUMAN_ANCHOR':nominal=refseq(human,94+k)
            controls,_=C.proposal(nominal,54+k,p)
        assert equal(controls,pred['controls'][k,:5]);cmd+=5
        for j,n in enumerate(pred['lengths'][k]):
            n=int(n)
            if not n:continue
            assert not pred['flags'][k,j,:n-1].any() and 0<int(pred['counts'][k,j,:n].min())<=100
            assert pred['costs'][k,j]==R.cost(pred['controls'][k,j],n,pred['flags'][k,j,n-1,0],pred['remaining_m'][k,j])
        selected=choose(pred['costs'][k],pred['flags'][k],pred['lengths'][k])
        if selected is None:assert f['stop']=='NO_FEASIBLE_TEACHER' and k==f['attempts']-1 and pred['selected'][k]==-1;break
        assert pred['selected'][k]==selected and equal(pred['controls'][k,5],controls[selected]);cmd+=2
        assert a['ownership'][94+accepted] and equal(a['actions'][94+accepted],controls[selected,0]);accepted+=1
        nominal=np.concatenate([controls[selected,1:],human[min(94+k+20,len(human)-1):min(94+k+20,len(human)-1)+1]])
    assert accepted==f['accepted'] and len(a['states'])==94+accepted
    physics=2*(9400+int(pred['counts'].sum())+int(a['native_counts'][94:].sum()))
    assert physics==s['physics']<=p['max_physics'] and 2*cmd==s['commands']<=p['max_commands']
    assert hashes(a)==f['mixed_hashes']==g['mixed_hashes'] and hashes(pred)==f['prediction_hashes']==g['prediction_hashes']
    q=R.actor(base);h=None;goal=np.asarray(read(R.SOURCE/'cases.json')[0]['goal_xy'])
    for i,x in enumerate(a['states']):
        assert np.abs(K.H.rebuild(x,a['poses'][i],a['history_poses'][i],goal,a['history_goals'][i])-x).max()<=2e-6
        before=np.zeros(16) if h is None else h[0,0].numpy().copy()
        with torch.no_grad():mode,_,h,_=q.step(torch.tensor(x[None]),h)
        assert equal(before,a['hidden_before'][i])
        for key,v in q.tape[-1].items():assert equal(v,a['actor_proposal_'+key][i])
        if not a['ownership'][i]:assert equal(mode[0].numpy(),a['actions'][i])
        d0=np.linalg.norm(a['poses'][i,:2]-goal);d1=np.linalg.norm(a['next_poses'][i,:2]-goal);n=int(a['native_counts'][i]);z=K.R.GAMMA**n;flags=a['end_flags'][i];terminal=bool(flags.any())
        rw=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not terminal)
        if terminal:rw+=K.R.GAMMA**(n-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
        assert abs(float(a['rewards'][i])-rw)<1e-12
    assert equal(h[0,0].numpy(),a['last_hidden_after'])
    for path,hsh in s['files'].items():assert sha(out/path)==hsh
    save(out/'audit.json',dict(state='PASS',native=0,Actor=0,optimizer=0,physics=physics,commands=s['commands'],all_input_and_actual_hashes_exact=True,all_selection_cost_nominal_and_two_pass_reconstruction_exact=True,teacher_safe_GOAL=s['teacher_safe_GOAL'],autonomous_six_stable=False));capacity(out,p['final_bytes'])
    print('TEACHER_AUDIT_PASS')
if __name__=='__main__':globals()[sys.argv[1]](sys.argv[2])

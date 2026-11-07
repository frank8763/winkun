"""Zero-native reconstruction of labels, likelihoods, updates and accounting."""
from t404_core import *

def teacher_audit(q,p,c):
    folder=OUT/'teachers'/str(c['cycle']);data=N.load_npz(folder/'search.npz');assert digest(data)==c['search_hash'] and q.phash()==c['policy_hash']
    mean=np.zeros((16,2));sd=np.broadcast_to(np.asarray(p['teacher_sd']),(16,2)).copy();best=None;bestid=None;cost=float('inf');nquery=0
    for tr in c['traces']:
        choices=teacher_candidates(mean,sd,best,c['cycle'],tr['generation'],p)
        for idx in tr['ids']:
            theta=choices[idx%12];u=teacher_controls(theta,N.S.arrays(REFERENCE)['actions'],c['first'],p);n=int(data['lengths'][idx]);assert n>0
            assert equal(theta,data['coefficients'][idx]) and equal(u,data['controls'][idx]) and not data['flags'][idx,:n-1].any() and data['flags'][idx,n-1].any()
            actual=objective(u,n,data['flags'][idx,n-1],data['remaining_m'][idx]);assert actual==data['costs'][idx]
            if bestid is None or (actual,idx)<(cost,bestid):best=theta.copy();bestid=idx;cost=actual
            nquery+=1
        elite=sorted(tr['ids'],key=lambda i:(float(data['costs'][i]),i))[:3];assert elite==tr['elite'] and bestid==tr['best_id']
        if not tr['goal_found']:mean=.5*mean+.5*data['coefficients'][elite].mean(0);sd=np.maximum(np.asarray(p['teacher_min_sd']),.5*sd+.5*data['coefficients'][elite].std(0))
        assert equal(mean,np.asarray(tr['mean_after'])) and equal(sd,np.asarray(tr['sd_after']))
    assert nquery==c['queries']<=24 and c['query_exact_repeats']==nquery and c['training_labels_allowed']==c['full_teacher_safe'] and not c['short_bridge_GOAL_required']
    charged=c['first']*100+2*int(data['counts'].sum());commands=1+2*nquery
    if c['full_teacher_safe']:
        a=N.S.arrays(c['full_path']);assert sha(c['full_path'])==c['full_sha256'];audit_array(q,a,p)
        assert digest(a)==c['full_hash'] and a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        n=int(data['lengths'][bestid]);assert len(a['states'])==c['first']+n and equal(a['actions'][c['first']:],data['controls'][bestid,:n])
        charged+=2*int(a['native_counts'][c['first']:].sum());commands+=2
    assert charged==c['physics'] and commands==c['commands']

def main():
    N.offline();p=plan();base=verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    for pid in read(OUT/'workers.json')['workers']:assert not Path('/proc',str(pid)).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q,opt=learner(base,p);old=old_groups(q,p);calls=0
    rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()];reps=[json.loads(x) for x in (OUT/'replays.jsonl').read_text().splitlines()]
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=N.S.arrays(r['path']);assert int(a['native_counts'].sum())==r['physics_steps'] and bool(a['end_flags'][-1,1])==r['success']
        if r['role'].startswith('diagnostic_') and r.get('full'):audit_array(q,a,p)
    for r in reps:assert r['exact'] and r['array_digest'] in {v['array_digest'] for v in rows}
    cohorts=[];blocks={x['tag']:x for x in s['update_blocks']}
    for ci,c in enumerate(s['teachers']):
        before=state(q,opt);teacher_audit(q,p,c)
        if not c['training_labels_allowed']:assert ci==len(s['cycles']);break
        cohorts.append(c);cycle=s['cycles'][ci];opt=optimizer(q,p)
        a=L.warm_fit(q,opt,correction_groups(q,cohorts,p),old,p,lambda:None,lambda *_:None,p['warm_steps']);calls+=p['warm_steps'];acq=blocks[cycle['acquire']]
        assert a==acq['fit'] and L.statehash(q,opt)==acq['proposed_model']['hashes']
        new=targets(q,cohorts);opt=optimizer(q,p);b=consolidate(q,opt,old,new,p,lambda:None,lambda *_:None);calls+=p['consolidation_steps'];con=blocks[cycle['consolidate']]
        assert b==con['fit'] and L.statehash(q,opt)==con['proposed_model']['hashes']
        if not cycle['supervised_accepted']:L.load(q,opt,before)
        else:
            opt=optimizer(q,p);beforeRL=state(q,opt);batch=[]
            for k in range(4):
                row=next(v for v in rows if v['role']==f'train_{ci}_{k}');a=N.S.arrays(row['path']);audit_array(q,a,p,c['first'],[p['seed'],ci,k]);batch.append(a)
            fit=L.fit(q,opt,batch,c['first'],p,correction_groups(q,cohorts,p),old,lambda:None,lambda *_:None);calls+=p['epochs'];rl=blocks[cycle['RL']]
            assert fit==rl['fit'] and L.statehash(q,opt)==rl['proposed_model']['hashes']
            if not cycle['RL_accepted']:L.load(q,opt,beforeRL)
        assert L.statehash(q,opt)==cycle['retained_hashes']
    assert L.statehash(q,opt)==s['result']['model']['hashes'] and s['updates']==dict(live=calls,shadow=calls)
    commands=[json.loads(x) for x in (OUT/'commands.jsonl').read_text().splitlines()];assert len(commands)==s['commands'] and sum(x['physics'] for x in commands)==s['physics'] and not any(x['error'] for x in commands)
    assert s['commands']+read(OUT/'preflight.json')['zero_physics_commands']<=p['max_commands'] and s['physics']<=p['max_physics']
    events=[json.loads(x) for x in (OUT/'optimizer_calls.jsonl').read_text().splitlines()] if (OUT/'optimizer_calls.jsonl').exists() else []
    assert sum(x[1]=='live' and x[2]=='AFTER' for x in events)==calls and sum(x[1]=='shadow' and x[2]=='AFTER' for x in events)==calls
    if s['result']['stable_six']:
        for k,n in [('old275',275),('main54',54),('fresh16',16)]:assert s['result'][k]['success']==s['result'][k]['valid']==n
        assert s['result']['reload78']['exact']==78
    for r in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/r['path'])==r['sha256']
    save(OUT/'audit.json',dict(state='PASS',native=0,new_optimization=0,historical_discarded_Adam=calls,all_training_history_hidden_density_rewards_exact=True,all_teacher_queries_exact_repeats=True,full_teacher_label_admission_no_pretraining_student_GOAL_gate=True,Actor_and_Adam_exact=True,physics=s['physics'],commands=s['commands'],stable_six=s['result']['stable_six']));N.manifest(OUT);print('T404_INDEPENDENT_AUDIT_PASS')
if __name__=='__main__':main()

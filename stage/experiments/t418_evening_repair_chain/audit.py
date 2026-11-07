"""Independent zero-native reconstruction of each child and its retained state."""
from repair_core import *
import repair_core as C

def jsonl(path):
    path=Path(path);return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
def uid(entry):
    r,c=entry['row'],entry['case'];return r['policy_hash'],r['job_id'],c['teacher_kind'],r['array_digest']
def model(q,opt,record):
    assert sha(record['path'])==record['sha256']
    stored=torch.load(io.BytesIO(gzip.decompress(Path(record['path']).read_bytes())),weights_only=False,map_location='cpu')
    actual=L.statehash(q,opt)
    assert actual==record['hashes']==stored['hashes']
    assert {key:digest(value) for key,value in stored['state'].items()}==actual
    assert stored['policy_hash']==record['policy_hash']==q.phash()
def source_model(q,opt,entry):
    path=entry['source_student_model'];assert sha(path)==entry['source_student_sha256']
    stored=torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),weights_only=False,map_location='cpu')
    assert stored['hashes']==L.statehash(q,opt) and stored['policy_hash']==q.phash()
    assert {key:digest(value) for key,value in stored['state'].items()}==stored['hashes']
def roles(p,role):
    return [role+'_'+suffix+'_'+j['id'] for suffix,jobs in [('old8',p['old8_jobs']),('left5',p['left5_jobs']),('right',[p['right_job']])] for j in jobs]
def joint(rows,p,role,q,record):
    lookup={r['role']:r for r in rows};values=[lookup[name] for name in roles(p,role)]
    assert all(r['policy_hash']==q.phash() and r['seed'] is None and not r['teacher'] for r in values)
    result=dict(old8=tally(values[:8]),left5=tally(values[8:13]),right1=tally(values[13:]),policy_hash=q.phash(),success_ids=sorted(r['job_id'] for r in values if r['success']),all14=all(r['success'] for r in values))
    assert result==record;return set(result['success_ids']),values
def event_rows(tag,count):
    return [[tag,which,where,k] for which in ['live','shadow'] for k in range(count) for where in ['BEFORE','AFTER']]

def main(protocol_file):
    global OUT
    N.offline();p=read(protocol_file);OUT=Path(p['out']);set_out(OUT)
    assert sha(Path(p['root_out'])/'protocol.json')==p['root_protocol_sha256']
    assert (HERE/'protocol.json').read_bytes()==(Path(p['root_out'])/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    s=read(OUT/'summary.json');assert not (OUT/'audit.json').exists()
    for file in ['main_started.json','workers.json']:
        v=read(OUT/file)
        for pid in [v[k] for k in ['pid','parent'] if type(v.get(k)) is int]+v.get('workers',[]):assert not Path('/proc',str(pid)).exists(),pid
    for row in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    start=time.time()
    def guard():assert time.time()<start+p['audit_seconds']-10
    rows=jsonl(OUT/'cases.jsonl');reps=jsonl(OUT/'replays.jsonl');commands=jsonl(OUT/'commands.jsonl')
    for row in rows:
        guard();assert sha(row['path'])==row['sha256'];a=N.S.arrays(row['path'])
        assert int(a['native_counts'].sum())==row['physics_steps']==row['physics']
        assert bool(a['end_flags'][-1,1])==row['success'] and bool(a['end_flags'][:,0].any())==row['collision'] and bool(a['end_flags'][-1,2])==row['timeout']
        assert a['end_flags'][-1].any() and not a['end_flags'][:-1].any()
        if row['full']:
            assert digest(a)==row['array_digest'] and digest(a['final_native_integration'])==row['integration_sha256']
            assert equal(a['states'][1:],a['next_states'][:-1])
    for row in reps:
        if row['role'].startswith('reload78_'):
            match=[r for r in rows if r['job_id']==row['job_id'] and r['role'].startswith(('main54_','old275_','fresh16_'))]
        else:match=[r for r in rows if r['role']==row['role'].replace('_exact_','_',1)]
        assert len(match)==1,(row['role'],len(match))
        for key in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert row[key]==match[0][key],key
        assert row['exact'] and row['independent_sample'] is False
    partials=[str(f) for f in OUT.rglob('partial*.npz')]
    ledger=dict(cases=len(rows),exact_replays=len(reps),command_records=len(commands),complete_physics=sum(r['physics_steps'] for r in rows+reps),summary_physics=s['physics'],summary_commands=s['commands'],partials=partials)
    ledger.update(unledgered_physics=s['physics']-ledger['complete_physics'],unledgered_commands=s['commands']-len(commands))
    if s['state']!='COMPLETE':
        save(OUT/'audit.json',dict(state='FAILED_PRODUCER_PARTIAL_PRESERVED',native=0,historical_discarded_Adam=0,raw_ledger=ledger));raise AssertionError('Incomplete producer preserved')
    assert not partials and ledger['unledgered_physics']==ledger['unledgered_commands']==0
    assert len(commands)==4+len(rows)+len(reps) and [r['id'] for r in commands]==list(range(1,len(commands)+1))
    assert sum(r['physics'] for r in commands)==s['physics'] and all(r['error'] is None for r in commands)
    assert s['physics']<=p['max_physics'] and s['commands']<=p['max_commands'] and s['elapsed_s']<=p['main_seconds']
    prior=p['prior_day_actual'];assert prior['physics']+s['physics']<=50_000_000 and prior['commands']+s['commands']<=2600 and prior['main_wall_s']+s['elapsed_s']<=20000
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);base=N.science(p);q,opt=learner(base,p);fixed=original_hash(q)
    initial_ids,initial_rows=joint(rows,p,'initial',q,s['result']['initial']);selected=initial_ids.copy();expected=roles(p,'initial');expected_reps=[]
    initial_values=[]
    for row in initial_rows:
        assert row['full'];a=N.S.arrays(row['path']);audit_array(q,a,p,0,None);assert not a['ownership'].any();initial_values.append((a,row))
    assert q.phash()==p['initial_expected_policy_hash'] and initial_ids==set(p['initial_expected_success_ids'])
    if p.get('initial_control_rows'):
        for old,new in zip(p['initial_control_rows'],initial_rows):
            for key in ['job_id','policy_hash','array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert old[key]==new[key],key
    g=build_groups(q,initial_values,p);assert max(float(old_kl(q,item,p)) for item in g['anchors'])<1e-26
    sources=copy.deepcopy(p.get('query_sources',[]));assert len({uid(z) for z in sources})==len(sources)
    def refresh():g['corrections']=[checked_query(q,z,p) for z in sources]
    refresh();events=[];calls=0;accepted_any=False;rejections=0;new_queries=0
    for rnd,record in enumerate(s['rounds']):
        guard();assert record==read(OUT/'blocks'/str(rnd)/'result.json') and record['round']==rnd and rnd<p['max_rounds'];before=L.state(q,opt)
        assert len(record['controls'])==(2 if p['phase']!='rl' and rnd==0 else 0)
        for entry in record['controls']+record['queries']:
            row,case=entry['row'],entry['case'];source_model(q,opt,entry);assert row['policy_hash']==q.phash() and row['teacher'] and row['seed'] is None
            assert sha(row['path'])==row['sha256'];a=N.S.arrays(row['path']);audit_query(q,a,case,p)
            assert case['teacher_kind'] in ['safe_reset','left_carry'] and case['first']==(0 if case['source_control'] else p['teacher_first'])
            expected.append(row['role'])
            if row['success']:
                expected_reps.append(row['role'].rsplit('_'+row['job_id'],1)[0]+'_exact_'+row['job_id'])
            if not case['source_control']:
                prior_entry=next((z for z in sources if uid(z)==uid(entry)),None)
                assert row['duplicate_of']==(None if prior_entry is None else prior_entry['row']['path'])
                assert row['unique_training_coverage']==(prior_entry is None)
                if row['success'] and prior_entry is None:sources.append(copy.deepcopy(entry));new_queries+=1
        refresh();bc=record['bc'];rl=record['rl']
        if bc is not None:
            assert p['phase']!='rl' and any(z['row']['success'] for z in record['queries'])
            actual=fit_bc(q,opt,g,p,guard,lambda *_:None);calls+=p['bc_steps'];events+=event_rows('BC'+str(rnd),p['bc_steps']);assert actual==bc['fit'];model(q,opt,bc['proposed_model'])
            safe,_=joint(rows,p,'BC_'+str(rnd)+'_proposal',q,bc['checks']);expected+=roles(p,'BC_'+str(rnd)+'_proposal')
            keep=initial_ids<=safe and selected<=safe and bool(safe-initial_ids);assert keep==bc['accepted']
            if keep:selected=safe;accepted_any=True
            else:L.load(q,opt,before)
            assert L.statehash(q,opt)==bc['retained_hashes']
        else:assert p['phase']=='rl' or not any(z['row']['success'] for z in record['queries'])
        eligible=p['phase']=='rl' or (bc is not None and bc['accepted'] and bc['checks']['old8']['success']==8)
        assert (rl is not None)==eligible
        if rl is not None:
            opt=torch.optim.Adam(params(q),lr=p['lr']);postbc=L.state(q,opt);job=p['left5_jobs'][3]
            assert rl['job']==job and rl['seeds']==[[p['seed'],p.get('phase_seed_offset',0),rnd,k] for k in range(4)]
            batch=[];batch_rows=[]
            for k in range(4):
                row=next(z for z in rows if z['role']==f'train_{rnd}_{k}');assert row['policy_hash']==q.phash() and row['full'] and not row['teacher'] and row['seed']==rl['seeds'][k]
                a=N.S.arrays(row['path']);audit_array(q,a,p,0,rl['seeds'][k]);assert not a['ownership'].any();batch.append(a);batch_rows.append(row);expected.append(row['role']);expected_reps.append(f'train_{rnd}_exact_{k}')
            actual=fit_ppo(q,opt,batch,g,p,rnd,guard,lambda *_:None);calls+=p['epochs'];events+=event_rows(rnd,p['epochs']);assert actual==rl['fit'] and tally(batch_rows)==rl['sampling'];model(q,opt,rl['proposed_model'])
            safe,_=joint(rows,p,'RL_'+str(rnd)+'_proposal',q,rl['checks']);expected+=roles(p,'RL_'+str(rnd)+'_proposal');keep=actual['numeric_accept'] and initial_ids<=safe and selected<=safe;assert keep==rl['accepted']
            if keep:selected=safe;accepted_any=True
            else:L.load(q,opt,postbc)
            assert L.statehash(q,opt)==rl['retained_hashes']
        success=bool((bc and bc['accepted']) or (rl and rl['accepted']));rejections=0 if success else rejections+1
        assert record['consecutive_rejections']==rejections and record['selected_success_ids']==sorted(selected)
        assert original_hash(q)==fixed
        if rejections>=2:assert rnd==len(s['rounds'])-1 and s['reason']=='TWO_CONSECUTIVE_REJECTIONS'
    assert events==jsonl(OUT/'optimizer_calls.jsonl') and s['updates']==dict(live=calls,shadow=calls)
    assert read(OUT/'query_sources.json')==sources==s['query_sources']
    result=s['result'];assert result['any_learning_accepted']==accepted_any and result['new_verified_query_count']==new_queries
    model(q,opt,result['model']);safe,_=joint(rows,p,'final_joint',q,result['final_joint']);assert initial_ids<=safe and selected<=safe;expected+=roles(p,'final_joint')
    centres=[r for r in rows if r['role'].startswith('final_centres_')];assert [r['job_id'] for r in centres]==[j['id'] for j in p['centres']] and tally(centres)==result['main_centres'];expected.extend(r['role'] for r in centres);expected_reps.extend('final_centres_exact_'+j['id'] for j in p['centres'])
    dev=[r for r in rows if r['role'].startswith('main54_')];old=[r for r in rows if r['role'].startswith('old275_')];fresh=[r for r in rows if r['role'].startswith('fresh16_')];reloads=[r for r in reps if r['role'].startswith('reload78_')]
    if result['final_joint']['all14']:
        assert [r['job_id'] for r in dev]==[j['id'] for j in p['main_jobs']] and [r['job_id'] for r in old]==[j['id'] for j in base['final_old_jobs']]
        assert tally(dev)==result['main54'] and tally(old)==result['old275'];expected.extend(r['role'] for r in dev+old)
    else:assert not dev and not old and result['main54'] is result['old275'] is None
    if fresh:
        assert all(r['success'] for r in dev+old) and [r['job_id'] for r in fresh]==[j['id'] for j in p['fresh_jobs']] and tally(fresh)==result['fresh16']
        for f,h in p['fresh_ledger'].items():assert sha(f)==h
        expected.extend(r['role'] for r in fresh)
    else:assert result['fresh16'] is None
    stable=bool(fresh and all(r['success'] for r in fresh) and len(reloads)==78);assert stable==result['stable_wide_six']
    if stable:
        assert result['reload78']==dict(exact=78,independent_sample=False) and [r['job_id'] for r in reloads]==[j['id'] for j in p['reload_jobs']];expected_reps.extend(r['role'] for r in reloads)
    else:assert not reloads and result['reload78'] is None
    for r in centres+dev+old+fresh:assert r['policy_hash']==q.phash()
    for r in reps:
        if r['role'].startswith(('final_centres_exact_','reload78_')):assert r['loaded_model_sha256']==result['model']['sha256'] and r['policy_hash']==q.phash()
    assert [r['role'] for r in rows]==expected and [r['role'] for r in reps]==expected_reps
    capacity(OUT,p['final_bytes'])
    save(OUT/'audit.json',dict(state='PASS',native=0,new_training=0,historical_discarded_Adam=calls,elapsed_s=time.time()-start,raw_ledger=ledger,
        exact_Actor_Adam_live_shadow_rejection_and_final_readback=True,own_initial_success_anchor_zero_KL=True,
        actual_safe_reset_and_left_carried_teacher_histories_exact=True,query_labels_deduplicated_and_all_physical_repeats_charged=True,
        student_sampling_onpolicy_density_hidden_reward_exact=True,safe_base_head_adapter_frozen=True,stable_wide_six=stable,
        old275=result['old275'],main54=result['main54'],fresh16=result['fresh16'],reload78=result['reload78'],seven_eight_entered=False))
    N.manifest(OUT);print('T418_CHILD_AUDIT_PASS',p['phase'],calls,flush=True)
if __name__=='__main__':
    assert len(sys.argv)==2;main(sys.argv[1])

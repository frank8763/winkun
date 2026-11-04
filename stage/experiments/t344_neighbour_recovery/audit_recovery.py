"""Additive audit recovery and mature-Adam snapshot isolation contract.

Original producer/audit/protocol/raw fields remain immutable. No native work.
"""
from t344_core import *

class IsolatedLearner(Learner):
    def load(self,s):return super().load(copy.deepcopy(s))

def register():
    assert not (HERE/'audit_recovery_protocol.json').exists()
    failure=read(DOC/'T344_ORIGINAL_AUDIT_FAILURE.json');assert failure['original_historical_Adam_completed']==8
    s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and s['all_workers_stopped']
    paths=[HERE/'audit_recovery.py',HERE/'audit.py',HERE/'run.py',HERE/'t344_core.py',HERE/'protocol.json',DOC/'T344_AUDIT_RECOVERY_PLAN.md',DOC/'T344_ORIGINAL_AUDIT_FAILURE.json']
    paths += [f for f in OUT.rglob('*') if f.is_file() and f.suffix not in ['.log','.pyc']]
    # This registration itself remains source-side, not part of the raw run.
    protocol=dict(scope='Additive failed-audit reconstruction and mature snapshot contract only',new_native=0,new_navigation_updates=0,
        additional_disposable_historical_Adam=16,previous_failed_audit_completed_Adam=8,total_disposable_historical_audit_Adam=24,
        wall_seconds=1200,absolute_deadline=read(OUT/'launch.json')['all_work_deadline_epoch'],supplemental_output_bytes=128*1024,
        input_hashes={str(f):K.C.sha(f) for f in paths})
    save(HERE/'audit_recovery_protocol.json',protocol);print('T344 additive audit recovery registered0native/16additionalhistoricalAdam,total24explicit')

def main():
    rp=read(HERE/'audit_recovery_protocol.json');p=read(HERE/'protocol.json');verify(p);assert not (DOC/'T344_AUDIT_RECOVERY.json').exists()
    for f,h in rp['input_hashes'].items():assert K.C.sha(f)==h,f
    s=read(OUT/'summary.json');w=read(OUT/'workers.json');l=read(OUT/'launch.json');assert all(not Path('/proc',str(pid)).exists() for pid in [s['pid'],l['pid'],*w['workers']])
    start=time.time();deadline=min(start+rp['wall_seconds'],rp['absolute_deadline']);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def guard():assert time.time()<deadline
    rows=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()];replays=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    blocks=[json.loads(v) for v in (OUT/'blocks.jsonl').read_text().splitlines()];calls=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    lookup={r['id']:r for r in rows};data={}
    for r in rows:
        guard();assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);data[r['id']]=a
        assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'] and r['complete'] and not r.get('error')
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        if r['full_arrays_saved']:
            assert np.array_equal(a['states'][1:],a['next_states'][:-1])
            for k,x in enumerate(a['states']):
                assert np.max(np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],np.array(r['goal_xy']),a['history_goals'][k])-x))<=2e-6
                n=int(a['native_counts'][k]);d0=float(np.linalg.norm(np.asarray(r['goal_xy'])-a['poses'][k,:2]));d1=float(np.linalg.norm(np.asarray(r['goal_xy'])-a['next_poses'][k,:2]));disc=K.R.GAMMA**n;contact,goal,timeout=a['end_flags'][k].astype(bool);terminal=contact or goal or timeout
                reward=-.000002*(1-disc)/(1-K.R.GAMMA)+.03*d0-disc*.03*d1*(not terminal)+(K.R.GAMMA**(n-1)*(float(goal)-float(contact)-.5*float(timeout)) if terminal else 0.)
                assert abs(reward-float(a['rewards'][k]))<2e-12
    assert len(rows)+len(replays)+5==s['commands']==92 and sum(r['physics_steps'] for r in [*rows,*replays])==s['physics_steps']<=p['max_physics']
    assert len(replays)==9 and all(r['full_arrays_and_integration_exact'] for r in replays)
    source=next(r for r in rows if r['role']=='support_regeneration');support=dict(row=source,arrays=data[source['id']])
    obj=IsolatedLearner(p);explained=[];contract=None;historical_calls=0
    for b in blocks:
        guard();actual=obj.hashes()
        if b['input_hashes']!=actual:
            assert b['round']==1 and actual==blocks[0]['output_hashes'] and b['input_hashes']['actor']==actual['actor']
            assert b['input_hashes']['optimizer']==b['output_hashes']['optimizer']
            explained.append(dict(round=b['round'],raw_recorded_input=b['input_hashes'],independently_reconstructed_input=actual,
                raw_field_left_unchanged=True,explanation='before.optimizer alias was mutated by shadow; post-update logging captured the final optimizer'))
        else:assert actual==b['input_hashes']
        if b['round']==1:
            # Demonstrate original alias without any optimizer step.
            snapshot=obj.state();snap_hash=digest(snapshot);legacy=Learner(p);legacy.load(snapshot)
            original_shares=any(x.data_ptr()==snapshot['optimizer']['state'][k][name].data_ptr() for k,v in legacy.opt.state_dict()['state'].items() for name,x in v.items() if isinstance(x,torch.Tensor))
            assert original_shares
            safe_snapshot=obj.state();unchanged=digest(safe_snapshot);a=IsolatedLearner(p);bclone=IsolatedLearner(p);a.load(safe_snapshot);bclone.load(safe_snapshot);other=bclone.hashes()
            first=next(v for state in a.opt.state.values() for v in state.values() if isinstance(v,torch.Tensor))
            with torch.no_grad():first.add_(1.)
            assert digest(safe_snapshot)==unchanged and bclone.hashes()==other
            a.load(safe_snapshot);assert a.hashes()==bclone.hashes()==obj.hashes()
            try:
                with torch.no_grad():next(a.actor.correction.parameters()).add_(1.)
                raise InterruptedError('mature synthetic interruption,0Adam')
            except InterruptedError:a.load(safe_snapshot)
            assert a.hashes()==obj.hashes() and digest(safe_snapshot)==unchanged and digest(snapshot)==snap_hash
            contract=dict(original_optimizer_tensor_alias_reproduced=True,isolated_snapshot_and_other_clone_unchanged=True,mature_Actor_Adam_reject_interrupt_exact=True,additional_contract_Adam_steps=0)
        batch=[dict(row=lookup[i],arrays=data[i]) for i in b['training_ids']]
        for item in batch:
            aa=item['arrays'];x=torch.tensor(aa['states'])
            with torch.no_grad():draw=obj.actor.draw(x,torch.tensor(aa['innovations']),False)
            assert all(np.array_equal(v.numpy(),aa[k]) for k,v in draw.items())
        successes=[dict(row=lookup[i],arrays=data[i]) for i in b['success_ids']];events=[]
        tr=T.supported_update(p,obj,batch,targets(p,obj.actor,support,successes),b['round'],guard,lambda row:events.append(row));historical_calls+=len(tr)
        assert digest(tr)==b['trace_sha256'] and obj.hashes()==b['output_hashes']
        for replica in ['live','shadow']:
            rec=[{k:v for k,v in r.items() if k!='replica'} for r in calls if r['replica']==replica]
            assert rec[b['round']*16:(b['round']+1)*16]==events
    assert historical_calls==16 and len(explained)==1 and contract is not None
    model=read(OUT/'models.json')['final'];assert K.C.sha(model['path'])==model['sha256']
    v=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu');assert digest(v['state'])==digest(obj.state())
    indexed={(r['role'],r['job_id']):r for r in rows}
    for r in rows:
        if r['role']=='model_readback27':
            role='final_wide5' if r['job_id'].startswith('wide_dev') else 'final_original2' if r['job_id'].startswith('dev6_') else 'final_old20';ref=indexed[(role,r['job_id'])]
            assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    for f,h in rp['input_hashes'].items():assert K.C.sha(f)==h,f
    save(DOC/'T344_AUDIT_RECOVERY.json',dict(state='ADDITIVE_RECONSTRUCTION_PASS_ORIGINAL_AUDIT_REMAINS_FAILED',original_audit_pass=False,
        all16_actual_updates_independently_reconstructed=True,all_source_raw_density_history_reward_model_Adam_exact=True,all9_native_replays_exact=True,model_readback27_exact=True,
        raw_metadata_error_explained_not_overwritten=explained,next_implementation_snapshot_contract=contract,new_native=0,new_training_updates=0,
        additional_disposable_historical_Adam=16,previous_failed_audit_Adam=8,total_historical_audit_Adam=24,
        commands=s['commands'],physics_steps=s['physics_steps'],stable_six=False,elapsed_s=time.time()-start,source_protocol_sha256=K.C.sha(HERE/'audit_recovery_protocol.json')))
    print('T344 additive audit recovery PASS0native/16additionalhistoricalAdam,total24; originalfailure/rawfields preserved')

if __name__=='__main__':
    if sys.argv[1]=='register':register()
    else:main()

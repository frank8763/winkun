"""Two fixed frozen proposal transformations, no new data/Adam/training grid."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t285_trajectory_constraints'
sys.path.insert(0,str(ROOT/'stage/experiments/t282_capped_balance'))
from balanced_learning import *
from t280_worker import generate,LimitedLog
sys.path.insert(0,str(ROOT/'stage/experiments/t278_fixed_goal_starts'))
from t278_guard import Budget
from t285_projection import project_displacement,flat_params,install,jacobian_rows,synthetic_contract
import ast,builtins,contextlib,gzip,io,json,multiprocessing as mp,os,shutil,signal,symtable,time,traceback


def read_arrays(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def prepare_data(p,plan):
    items=[]
    for row in plan['historical_training']:
        a=read_arrays(row['path']);assert all(S.digest(v)==row['array_hashes'][k] for k,v in a.items())
        items.append(dict(row=row,arrays=a))
    return L.training_batch(items,p)


def objective(p,data,ax,ay,offset,base,balance):
    rng=np.random.default_rng(base['seed']);ids=np.concatenate([rng.integers(offset[t],offset[t+1],32) for t in range(8)])
    with torch.no_grad():
        lp,_=L.likelihood(p,data);delta=lp-data['oldlp'];ratio=delta.exp();adv=data['advantage']
        term=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)*data['weights']
        tasks=[term[data['task']==t].sum()/4/base['surrogate_horizon_decisions'] for t in range(4)]
        surrogate=torch.stack([s*f*balance['common_scale'] for s,f in zip(tasks,balance['factors'])]).mean()
        retention=base['retention_weight']*(p.step(ax[ids])[0]-ay[ids]).square().mean()
        mse,mx=anchor_measure(p,ax,ay,offset)
        return dict(objective=float(surrogate-retention),surrogate=float(surrogate),retention=float(retention),
            conditional_k3=float((delta.exp()-1-delta).mean()),max_abs_log_ratio=float(delta.abs().max()),
            balanced_anchor_MSE=float(mse),anchor_max=float(mx))


def actions(p,states):
    with torch.no_grad():return np.stack([p.step(torch.tensor(x[None]))[0][0].numpy() for x in states])


def build_candidate(base,plan,indices,states):
    parent=policy(base);original=policy(base);original.load_state_dict(L.read(plan['rejected_candidate'])['policy'])
    assert S.digest(original.state_dict())==plan['rejected_policy_hash']
    initial=flat_params(parent);delta=flat_params(original)-initial
    j=jacobian_rows(parent,states[indices]);direction,stats=project_displacement(j,delta,plan['relative_eigenvalue_cutoff'])
    q=policy(base);install(q,initial+direction)
    actual_delta=flat_params(q)-initial
    stats.update(indices=indices,parameter_count=int(initial.numel()),actual_float32_displacement_norm=float(torch.linalg.vector_norm(actual_delta)),
        projected_float64_sha=S.digest(direction.numpy()),actual_float32_displacement_sha=S.digest(actual_delta.numpy()),jacobian_sha=S.digest(j.numpy()),
        candidate_policy_hash=S.digest(q.state_dict()))
    pa=actions(parent,states);old=actions(original,states);new=actions(q,states)
    stats.update(full_parent_trajectory_action_error_before_max=float(np.linalg.norm(old-pa,axis=1).max()),
        full_parent_trajectory_action_error_after_max=float(np.linalg.norm(new-pa,axis=1).max()),
        selected_states_error_before_max=float(np.linalg.norm(old[indices]-pa[indices],axis=1).max()),
        selected_states_error_after_max=float(np.linalg.norm(new[indices]-pa[indices],axis=1).max()))
    return q,stats


def main():
    assert not (OUT/'status.json').exists()
    plan=json.loads((HERE/'protocol.json').read_text());base=json.loads(Path(plan['source_protocol']).read_text())
    start=time.time();counts=mp.Array('q',1);meter=None;raw=[];results={};exports={};log=None;state='COMPLETE'
    def status(phase):C.save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=time.time(),main_epoch=start,absolute_deadline_epoch=start+plan['wall_seconds'],commands=0 if meter is None else meter.commands,physics_steps=int(counts[0]),optimizer_calls=0,navigation_training_updates=0))
    def alarm(signum,frame):raise TimeoutError('T285 absolute cutoff')
    def payload(path,blob,limit):
        assert not path.exists() and len(blob)<=limit;meter.charge_payload(len(blob));path.write_bytes(blob)
        return dict(path=str(path),sha256=C.sha(path),bytes=len(blob))
    def roll(job,q,arm,role,full=False):
        meter.begin(job['case']['step_cap']);status(arm+'_'+role)
        job=dict(job,id=f'{arm}_{role}_{job["id"]}',source_reset_id=job['id'],mode='deterministic',seed=0,aux_eta=0.)
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):item=generate(job,q,native,create,start+plan['wall_seconds']-10,counts,0,int(counts[0]))
        meter.physics=int(counts[0]);r=item['row'];a=item['arrays'];r.update(arm=arm,role=role,policy_hash=S.digest(q.state_dict()))
        saved=a if full else {k:a[k] for k in ['actions','native_counts','end_flags','rewards','environment_rewards','aux_costs']}
        buf=io.BytesIO();np.savez_compressed(buf,**saved);r.update(payload(OUT/(job['id']+'.npz'),buf.getvalue(),plan['full_episode_bytes'] if full else plan['compact_episode_bytes']))
        r['saved_array_keys']=list(saved);raw.append(r);C.save(OUT/'cases.json',raw)
        assert r['complete'] and not r['error'],'Partial evidence retained; no continuation'
        return item
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    signal.signal(signal.SIGALRM,alarm);signal.alarm(plan['wall_seconds'])
    try:
        assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        meter=Budget(ROOT,OUT,plan,start_epoch=start);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));status('FIXED_PROJECTION')
        parent=policy(base);data=prepare_data(parent,plan);ax,ay,offset=anchors(base,parent)
        before=objective(parent,data,ax,ay,offset,base,plan['balance']);C.save(OUT/'parent_objective.json',before)
        states=read_arrays(plan['parent_trajectory'])['states'];models={}
        for arm,ids in plan['arms'].items():
            q,stats=build_candidate(base,plan,ids,states);stats['measurement']=objective(q,data,ax,ay,offset,base,plan['balance'])
            m=stats['measurement'];stats['numeric_gate']=all(math.isfinite(v) for v in m.values()) and m['objective']>before['objective']+1e-10 and m['conditional_k3']<=base['max_conditional_k3'] and m['max_abs_log_ratio']<=base['max_abs_log_ratio']
            stats.update(old20_executed=0,old20_safe=0,diagnostic_six_m_executed=0,diagnostic_six_m_safe=0,local_executed=0,local_safe=0)
            buf=io.BytesIO();torch.save(dict(policy=q.state_dict(),source_policy_hash=base['policy_hash'],source_rejected_policy_hash=plan['rejected_policy_hash'],frozen_displacement_projection=True,optimizer=None),buf)
            exports[arm]=payload(OUT/(arm+'.pt.gz'),gzip.compress(buf.getvalue(),compresslevel=6,mtime=0),plan['model_bytes'])
            qq=policy(base);qq.load_state_dict(L.read(exports[arm]['path'])['policy']);assert S.digest(qq.state_dict())==stats['candidate_policy_hash'];models[arm]=qq
            results[arm]=stats;C.save(OUT/'projections.json',results);meter.checkpoint()
        (OUT/'native').mkdir(exist_ok=True);log=LimitedLog(OUT/'native.log')
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):native,create=S.native_setup(OUT/'native')
        for arm,q in models.items():
            if not results[arm]['numeric_gate']:continue
            first=roll(plan['critical_job'],q,arm,'critical',True);repeat=roll(plan['critical_job'],q,arm,'critical_replay',True)
            assert first['row']['array_hashes']==repeat['row']['array_hashes'] and first['row']['integration_sha256']==repeat['row']['integration_sha256']
            assert all(np.array_equal(v,repeat['arrays'][k]) for k,v in first['arrays'].items())
            results[arm].update(old20_executed=1,old20_safe=int(safe(first['row'])),critical_replay_exact=True)
            if not safe(first['row']):C.save(OUT/'projections.json',results);continue
            for job in plan['remaining_guard_jobs']:
                item=roll(job,q,arm,'old_guard');results[arm]['old20_executed']+=1;results[arm]['old20_safe']+=int(safe(item['row']))
            if results[arm]['old20_safe']==20:
                for role,jobs in [('diagnostic_six_m',plan['six_m_centers']),('local',plan['local_centers'])]:
                    for job in jobs:
                        item=roll(job,q,arm,role);results[arm][role+'_executed']+=1;results[arm][role+'_safe']+=int(safe(item['row']))
            C.save(OUT/'projections.json',results)
        assert sum(r['physics_steps'] for r in raw)==int(counts[0])<=plan['max_physics']
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    except BaseException:
        state='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physical_steps=int(counts[0])))
    finally:
        signal.alarm(0)
        if log is not None:log.close()
        if meter is not None:meter.close()
    summary=dict(state=state,arms=results,models=exports,commands=0 if meter is None else meter.commands,physics_steps=int(counts[0]),optimizer_calls=0,navigation_training_updates=0,
        elapsed_s=time.time()-start,all_workers_stopped=True,main_epoch=start,absolute_deadline_epoch=start+plan['wall_seconds'],own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),
        full275_54_16_78_not_executed=True,model_promoted=False,automatic_next_training=False,
        limitation='Two posthoc consumed-data frozen projection arms of one identical historical Adam displacement. No fresh learning samples, independent seed replication, full navigation gate or novelty/causality/stability claim. Native failure ends that arm without new anchor/rank/step-size grid.')
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)


def preflight():
    p=json.loads((HERE/'protocol.json').read_text());assert not (OUT/'status.json').exists()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    table=symtable.symtable(Path(__file__).read_text(),str(__file__),'exec');missing=set()
    def walk(t):
        for s in t.get_symbols():
            if s.is_referenced() and s.is_global() and s.get_name() not in globals() and not hasattr(builtins,s.get_name()):missing.add(s.get_name())
        for child in t.get_children():walk(child)
    walk(table);assert not missing,missing
    base=json.loads(Path(p['source_protocol']).read_text());q=policy(base);original_hash=S.digest(q.state_dict());install(q,flat_params(q))
    assert S.digest(q.state_dict())==original_hash==base['policy_hash']
    assert all(len(ids)==len(set(ids))==9 for ids in p['arms'].values())
    used=S.used(ROOT/'stage/runs');assert used-S.used(OUT)+p['final_bytes']<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes'] and p['max_commands']*28000<=p['max_physics']
    C.save(OUT/'preflight.json',dict(state='PASS',protocol_sha256=C.sha(HERE/'protocol.json'),all_nested_global_names_resolve=True,actual_model_parameter_roundtrip_exact=True,full_reserved_global_bytes=used-S.used(OUT)+p['final_bytes'],**synthetic_contract()))
    print('T285 PREFLIGHT PASS: zero native/optimizer; analytic rank-deficient projection')


if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:preflight()
    else:assert not sys.argv[1:];main()

from t278_native import *
import t278_native as N
from t278_guard import Budget
import io,json,os,subprocess,time,traceback
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t278_fixed_goal_starts'


def main():
    assert not (OUT/'status.json').exists(),'No repeat/state/radius/seed grid'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.time();meter=None;rows=[];geometry=[];reason='COMPLETE';pending=None;env=None
    def status(name):C.save(OUT/'status.json',dict(state=name,pid=os.getpid(),physics_steps=0 if meter is None else meter.physics,
        commands=0 if meter is None else meter.commands,optimizer_calls=0,updated_epoch=time.time()))
    def append(name,row):
        text=json.dumps(row)+'\n';assert len(text.encode())<8192
        with (OUT/name).open('a') as f:f.write(text)
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
        meter=Budget(ROOT,OUT,plan,start_epoch=start);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),CPU_affinity=sorted(os.sched_getaffinity(0))))
        p=policy(plan).eval().requires_grad_(False);assert S.digest(p.state_dict())==plan['policy_hash']
        N.R_NATIVE,create=S.native_setup(OUT/'native');status('STARTING')
        for job in plan['jobs']:
            meter.begin(28000);env,obs,g=initial(create,job);geometry.append(g);append('initials.jsonl',g)
            if not g['valid']:
                env.close();env=None;append('invalid.jsonl',dict(id=job['id'],reason='Native full vehicle initial contact',physics_steps=0));status('INVALID_'+job['id']);continue
            snap=OUT/(job['id']+'_snapshot.npz');meter.charge_payload(plan['snapshot_bytes'])
            C.write_snapshot(snap,C.capture(env,obs));assert snap.stat().st_size<=plan['snapshot_bytes']
            for repeat in [False,True]:
                if repeat:
                    meter.begin(28000);env,obs,_=R.task_env(create,job['base_job']);obs=C.restore(env,C.read_snapshot(snap))
                    packet=L.packet(env,obs,0,28000,np.zeros(2,dtype=np.float32),0);assert S.digest(packet)==g['initial120_sha256']
                row,a=rollout(env,obs,p,meter,job['id']+('_reload' if repeat else ''));env=None;row.update(job_id=job['id'],side=job['side'],start_remaining_m=job['remaining_distance_m'],replay=repeat)
                # Persist the completed or partial unique rollout BEFORE any
                # completeness assertion. At most one partial replay is saved.
                if not repeat or not row['complete'] or row['error']:
                    buf=io.BytesIO();np.savez_compressed(buf,**a);blob=buf.getvalue();assert len(blob)<=plan['episode_bytes']
                    dest=OUT/(row['id']+'.npz');assert not dest.exists();meter.charge_payload(len(blob));dest.write_bytes(blob);row.update(path=str(dest),sha256=C.sha(dest))
                append('cases.jsonl',row);rows.append(row);status(row['id'])
                if not row['complete'] or row['error']:raise RuntimeError('Saved administrative prefix; stop without replacing missing outcomes')
                if repeat:
                    assert row['array_hashes']==pending['array_hashes'] and row['integration_sha256']==pending['integration_sha256']
                    with np.load(pending['path']) as z:assert all(np.array_equal(v,z[k]) for k,v in a.items())
                else:pending=row
            meter.checkpoint()
        assert S.digest(p.state_dict())==plan['policy_hash']
    except BaseException:
        reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=0 if meter is None else meter.physics,
            commands=0 if meter is None else meter.commands,original_unique_and_partial_payloads_preserved=True))
    if env is not None:env.close();env=None
    if meter is not None:meter.close()
    unique=[r for r in rows if not r['replay']]
    result=dict(state=reason,physics_steps=0 if meter is None else meter.physics,commands=0 if meter is None else meter.commands,
        optimizer_calls=0,Actor_updates=0,all_workers_stopped=True,all_native_envs_closed=True,elapsed_s=time.time()-start,
        full_global_scan_calls=0 if meter is None else meter.full_scans,invalid_initial_states=sum(not g['valid'] for g in geometry),
        unique_valid_attempts=len(unique),safe=sum(r['complete'] and r['success'] and not r['collision'] for r in unique),
        exact_reloads=sum(r['replay'] and r['complete'] for r in rows),administratively_censored=sum(not r['complete'] for r in rows),
        unique_results=[{k:r[k] for k in ['job_id','side','start_remaining_m','complete','success','collision','timeout','remaining_m']} for r in unique],
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),safe_T255_policy_hash=plan['policy_hash'],
        original6m_start_navigation_gate=False,automatic_next_training=False,
        interpretation='Frozen reset-only shorter-start capability at true6m endpoint; no training/forgetting, no full6m success or reverse-curriculum efficacy proved.')
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()

"""Eight unique old-trajectory reconstructions with all physics charged."""
from t277_native import *
from t277_labels import fixed_targets,first_hit_record
import io,json,os,shutil,subprocess,time,traceback
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t277_physical_hindsight'


def main():
    assert not (OUT/'status.json').exists(),'Never restart or add old seed trajectories'
    p=json.loads((HERE/'protocol.json').read_text());start=time.time();deadline=start+p['wall_seconds']
    physics=commands=0;rows=[];labels=[];state='COMPLETE';inflight=False
    def status(name):C.save(OUT/'status.json',dict(state=name,pid=os.getpid(),physics_steps=physics,commands=commands,optimizer_calls=0,updated_epoch=time.time()))
    def guard():
        assert time.time()<deadline and physics<=p['max_physics'] and commands<=p['max_commands']
        assert S.used(OUT)<p['final_bytes'] and S.used(ROOT/'stage/runs')<p['global_bytes']
        assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    def on_step():
        nonlocal physics
        physics+=1
    try:
        for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
        assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        status('STARTING');native,create=S.native_setup(OUT/'native')
        for ref in p['references']:
            guard();assert commands+1<=8 and physics+ref['job']['case']['step_cap']<=p['max_physics']
            commands+=1;inflight=True
            data,source,row=replay(ref,native,create,deadline,guard,on_step);inflight=False
            buf=io.BytesIO();np.savez_compressed(buf,**data);blob=buf.getvalue()
            assert len(blob)<=ref['physics_steps']*17+p['per_payload_overhead_bytes']
            path=OUT/(ref['id']+'.npz');assert not path.exists();path.write_bytes(blob)
            row.update(path=str(path),sha256=C.sha(path));text=json.dumps(row)+'\n';assert len(text.encode())<=4096
            with (OUT/'cases.jsonl').open('a') as f:f.write(text)
            rows.append(row)
            ends=source['native_counts'].astype(np.int64).cumsum()
            for chosen in fixed_targets(source):
                record=first_hit_record(data['physical_xy'],data['physical_flags'],source['poses'][0,:2],source['next_poses'][chosen,:2],ends,int(chosen))
                record.update(source_id=ref['id']);assert len(json.dumps(record).encode())<=p['max_label_bytes'];labels.append(record)
            status(ref['id'])
        assert len(rows)==8 and len(labels)==64 and physics==sum(r['physics_steps'] for r in p['references'])
        C.save(OUT/'hindsight_first_hits.json',labels)
    except BaseException:
        state='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight=inflight))
    eligible=[r for r in labels if r['eligible']]
    result=dict(state=state,physics_steps=physics,commands=commands,optimizer_calls=0,Actor_updates=0,
        all_workers_stopped=True,all_native_envs_closed=True,elapsed_s=time.time()-start,
        exact_old_trajectories=len(rows),original_goal_safe=sum(r['success'] and not r['collision'] for r in rows),
        future_goal_requests=len(labels),eligible_future_labels=len(eligible),invalid_future_labels=len(labels)-len(eligible),
        interior_hold_first_hits=sum(r['inside_original_hold'] for r in eligible),first_hits_before_chosen_future=sum(r['first_before_selected_future'] for r in eligible),
        inflight=inflight,own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),
        independent_new_navigation_episodes=0,ready_onpolicy_or_Actor120_relabel_replay=False,six_m_navigation_gate=False,
        interpretation='Exact old command replay with native first-hit witnesses for new training-goal labels. Original6m failures unchanged; no new navigation success or Actor learning.')
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()

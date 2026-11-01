"""One fixed same-state direction test; six commands including exact repeats."""
from t273_native import *
import os,shutil,traceback
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t273_value_direction_probe'


def main():
    assert not (OUT/'status.json').exists(),'No restart/additional radii or states'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.time();deadline=start+plan['wall_seconds']
    physics=commands=0;rows=[];reason='COMPLETE';native=create=p=None
    def status(label):C.save(OUT/'status.json',dict(state=label,pid=os.getpid(),physics_steps=physics,commands=commands,optimizer_calls=0,updated_epoch=time.time()))
    def guard():
        assert time.time()<deadline and physics<=plan['max_physics'] and commands<=plan['max_commands']
        assert S.used(OUT)<plan['final_bytes'] and S.used(ROOT/'stage/runs')<plan['global_bytes']
        assert shutil.disk_usage(ROOT).free>=plan['min_free_bytes']
    def on_step():
        nonlocal physics
        physics+=1
    def case(name,proposal):
        nonlocal commands
        guard();assert commands+1<=plan['max_commands'] and physics+10000<=plan['max_physics'];commands+=1
        row,a=simulate(plan,proposal,p,native,create,deadline,guard,on_step)
        buf=io.BytesIO();np.savez_compressed(buf,**a);blob=buf.getvalue();assert len(blob)<=plan['episode_bytes']
        path=OUT/(name+'.npz');assert not path.exists();path.write_bytes(blob)
        row.update(id=name,path=str(path),sha256=C.sha(path));rows.append(row)
        with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        status(name);return row,a
    try:
        for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
        assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        status('STARTING');p=policy(plan).eval().requires_grad_(False);assert S.digest(p.state_dict())==plan['policy_hash']
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],single_native_process=True))
        native,create=S.native_setup(OUT/'native')
        for proposal in ['control','up','down']:
            first,a=case(proposal,proposal);repeat,b=case(proposal+'_exact_repeat',proposal)
            assert first['array_hashes']==repeat['array_hashes'] and first['integration_sha256']==repeat['integration_sha256']
            assert all(np.array_equal(v,b[k]) for k,v in a.items())
            if proposal=='control':
                assert first['array_hashes']==plan['reference']['array_hashes'] and first['integration_sha256']==plan['reference']['integration_sha256']
                C.save(OUT/'restoration_gate.json',dict(state='PASS',control_and_repeat_exact_original_T269=True,no_new_intervention_before_gate=True))
        assert S.digest(p.state_dict())==plan['policy_hash']
    except BaseException:
        reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands))
    unique={r['proposal']:r for r in rows if not r['id'].endswith('_exact_repeat')}
    ranking=False
    if len(unique)==3:
        a,b,c=[unique[k]['actual_score16'] for k in ['up','control','down']];ranking=a>=b>=c and a>c
    result=dict(state=reason,physics_steps=physics,commands=commands,optimizer_calls=0,Actor_updates=0,
        all_workers_stopped=True,all_native_envs_closed=True,elapsed_s=time.time()-start,
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),unique_actions=len(unique),exact_replays=len(rows)-len(unique),
        actual_score16_by_action={k:r['actual_score16'] for k,r in unique.items()},
        empirical_direction_screen_passed=ranking,automatic_Actor_coupling=False,six_m_navigation_gate=False,
        limitation='One known left6m context at36s; precomputed one-hold diagnostic actions and frozen Actor continuation, not a new deployed policy. Ties are not ranking evidence; behavior-mixture forecasts are not deterministic-continuation guarantees.')
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()

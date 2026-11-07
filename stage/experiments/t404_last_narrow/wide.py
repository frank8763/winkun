"""Independent, conditional wide-scene frozen admission; never calls learning."""
from t404_core import *
WIDE=ROOT/'stage/runs/t405_wide_admission'

def main():
    p=plan();base=verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not s['result']['stable_six'] and read(OUT/'audit.json')['state']=='PASS' and not WIDE.exists()
    for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    WIDE.mkdir();(WIDE/'episodes').mkdir();wp=dict(id='T405_CONDITIONAL_WIDE117_80_FROZEN_ADMISSION',source_protocol_sha256=sha(OUT/'protocol.json'),source_final_model=s['result']['model'],width_m=1.17,shift_x_m=.6,cap=40000,models=['last_narrow_frozen_candidate','T255_function_zero_residual_control'],new_Actor=0,new_Adam=0,max_physics=320000,max_commands=10,own_bytes=p['wide_reserved_bytes'],main_seconds=900,task_change='separate wider task; no claim of original narrow pass',deployment=p['deployment']);save(WIDE/'protocol.json',wp)
    N.offline();N.configure(WIDE);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();physics=commands=0;rows=[];replays=[];w=ss=None;status='FAILED';phase='SETUP';last=0.;q=actor(base,p)
    def forbidden(*a,**k):raise AssertionError('T405 has zero training')
    torch.optim.Adam.step=forbidden;torch.Tensor.backward=forbidden
    def guard(physical=False):
        nonlocal last
        L.check_stop();assert time.time()<start+895 and physics<=320000 and commands<=10
        if not physical and time.time()-last>3:
            assert N.K.S.used(WIDE)<=p['wide_reserved_bytes'] and N.K.S.used(ROOT/'stage/runs')<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
            last=time.time();save(WIDE/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=physics,commands=commands))
    def charge():
        nonlocal physics
        physics+=1
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(WIDE/'launch.json',dict(pid=os.getpid(),epoch=start,model_sha256=sha(s['result']['model']['path']),parent_audit_sha256=sha(OUT/'audit.json')))
    try:
        m,native,create,w,_=N.S.P.C.B.setup(WIDE/'native',base);commands+=1;original=copy.deepcopy(m.ACTIVE)
        for kind in wp['models']:
            q=actor(base,p,s['result']['model']['path']) if kind=='last_narrow_frozen_candidate' else L.Actor(base,p).eval();q.requires_grad_(False);fixed=digest(q.state_dict())
            if kind=='T255_function_zero_residual_control':assert digest(q.base.state_dict())==base['safe_policy_hash']
            for j0 in p['centres']:
                j=copy.deepcopy(j0);j['id']='wide117_80_'+j0['source_reset_id'];j['circle_shift_x_m']=.6;j['expected_initial120_sha256']=None
                reference=None
                for rep in range(2):
                    phase=kind+'_'+j['id']+f'_rep{rep}';commands+=1;q.sample=False;q.reset_noise();m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=.6
                    ss=L.Session(j,native,create,w,guard,charge,q);L.install_stop_handlers()
                    while not ss.done():ss.step(owner=0)
                    a=payload(ss);row=dict(ss.result(),role=phase,job_id=j['id'],model=kind,policy_hash=q.phash(),array_digest=digest(a),integration_sha256=digest(a['final_native_integration']))
                    if rep==0:
                        f=WIDE/'episodes'/(str(len(rows))+'.npz');N.write_npz(f,N.S.packed(a) if kind=='last_narrow_frozen_candidate' else {k:a[k] for k in ['actions','native_counts','end_flags']},p['full_bytes'] if kind=='last_narrow_frozen_candidate' else 5*1024);row.update(path=str(f),sha256=sha(f),full=kind=='last_narrow_frozen_candidate');reference=row;rows.append(row)
                    else:
                        for k in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert row[k]==reference[k]
                        row.update(exact=True,independent_sample=False);replays.append(row)
                    assert digest(q.state_dict())==fixed;ss.close();ss=None
        status='COMPLETE'
    except BaseException:save(WIDE/'failure.json',dict(phase=phase,error=traceback.format_exc(),physics=physics,commands=commands))
    finally:
        if ss and ss.rows:
            try:N.write_npz(WIDE/'partial.npz',N.S.packed(payload(ss)),p['full_bytes'])
            except BaseException:pass
        if ss:ss.close()
        if w:w.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        save(WIDE/'summary.json',dict(state=status,pid=os.getpid(),physics=physics,commands=commands,elapsed_s=time.time()-start,cases=rows,replays=replays,new_Actor=0,new_Adam=0,all_workers_stopped=True,all_native_sessions_closed=True,full_six_gate=False,seven_eight_entered=False,next='Choose wide6m baseline under separate full protocol; do not treat2centres as full-map stability'));N.manifest(WIDE)
    assert status=='COMPLETE'
def audit():
    p=plan();base=verify(p);s=read(WIDE/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (WIDE/'audit.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for r in s['cases']:
        assert sha(r['path'])==r['sha256'];a=N.S.arrays(r['path']);assert int(a['native_counts'].sum())==r['physics_steps']
        if r['full']:
            q=actor(base,p,read(OUT/'summary.json')['result']['model']['path']);audit_array(q,a,p);assert digest(a)==r['array_digest']
    assert len(s['cases'])==len(s['replays'])==4 and s['commands']==9 and s['physics']==sum(r['physics_steps'] for r in [*s['cases'],*s['replays']])<=320000
    for r in read(WIDE/'local_manifest.json')['files']:assert sha(ROOT/r['path'])==r['sha256']
    save(WIDE/'audit.json',dict(state='PASS',native=0,optimizer=0,all_four_serialized_native_repeats_exact=True,full_candidate_arrays_hidden_history_exact=True,full_six_gate=False));N.manifest(WIDE);print('T405_WIDE_FROZEN_AUDIT_PASS')
if __name__=='__main__':globals()[sys.argv[1]]()

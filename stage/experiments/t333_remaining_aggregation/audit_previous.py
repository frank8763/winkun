from t333_learning import *

def main():
    old=ROOT/'stage/runs/t332_recovery_aggregation';p=read(old/'protocol.json');s=read(old/'summary.json')
    assert s['state']=='FAILED' and s['commands']==s['physics_steps']==0 and s['supervised_Adam_calls']==600
    for pid in [s['pid'],read(old/'launch.json')['pid'],*read(old/'workers.json')['workers']]:assert not Path('/proc/'+str(pid)).exists()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h
    dest=DOC/'T332_PARTIAL_AUDIT.json';assert not dest.exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=model(p);opt=optimizer(q,p,'bc');d=bc_data(p,q,'control')
    logs=[json.loads(line) for line in (old/'optimizer_calls.jsonl').read_text().splitlines()]
    for replica in ['live','shadow']:
        q=model(p);opt=optimizer(q,p,'bc');rows=[]
        fit(p,q,opt,lambda:None,lambda row:rows.append(dict(row,replica=replica)),'control',d)
        assert rows==[r for r in logs if r['replica']==replica]
    patch=read_patch(old/'control_candidate.pt.gz')
    assert K.S.digest(state(q))==K.S.digest(patch['residual']) and K.S.digest(q.state_dict())==patch['policy_hash']
    assert K.S.digest(opt.state_dict())==patch['optimizer_hash']
    assert K.C.sha(old/'control_candidate.pt.gz')==K.C.sha(old/'partial.pt.gz')
    save(dest,dict(state='PASS_PARTIAL_EVIDENCE_ONLY',original_summary_state='FAILED',
        native_added=0,original_native_commands=0,original_physics_steps=0,
        original_supervised_Adam_calls=600,disposable_historical_reconstruction_Adam_calls=600,
        live_shadow_all_logs_and_full_control_Actor_Adam_exact=True,all_processes_absent=True,
        recovery_arm_not_trained=True,all_54_native_cases_not_executed=True,
        control_patch_sha256=K.C.sha(old/'control_candidate.pt.gz'),control_policy_hash=patch['policy_hash'],
        failure='KeyError source_reset_id before first native command',not_six_success=True))
    print('T332 partial600Adam/0native independent audit PASS')

if __name__=='__main__':main()

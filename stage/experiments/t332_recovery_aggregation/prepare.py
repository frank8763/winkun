from t332_learning import *

def main():
    assert not OUT.exists()
    assert read(DOC/'T331_AUDIT.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T331.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=read(ROOT/'stage/experiments/t329_success_residual/protocol.json')
    p.update(experiment='T332_PAIRED_RECOVERY_DATA_AGGREGATION',arms=['control','recovery'],
        recovery_demo=str(ROOT/'stage/runs/t331_policy_prefix_retry/attempt_01.npz'),human_start_decision=150,
        bc_steps=300,max_commands=108,max_physics=4320000,wall_seconds=2700,training_seconds=2100,
        final_reserve_seconds=600,audit_seconds=1800,final_bytes=6*1024**2,
        payload_bytes=3*1024**2,metadata_bytes=2*1024**2,control_equals_T329=False,
        full_old_retention_rows=28629,training_kind='SUPERVISED_CORRECTION_PREPARATION_NOT_RL')
    a=arrays(p['recovery_demo']);assert a['end_flags'][-1].tolist()==[0,1,0]
    assert K.C.sha(p['recovery_demo'])=='e332fbc2858899e9f560b0fb11da6c92f7b087a4d3d2aff55bacf506e282291a'
    assert len(a['states'][150:])==168
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}
    files|={p['recovery_demo'],str(receipt),str(DOC/'T331_AUDIT.json'),str(DOC/'T332_PLAN.md')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',fully_reserved_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'])))

if __name__=='__main__':main()

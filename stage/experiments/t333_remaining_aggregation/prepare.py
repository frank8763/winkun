from t333_learning import *

def main():
    assert not OUT.exists();old=ROOT/'stage/runs/t332_recovery_aggregation';p=read(old/'protocol.json');s=read(old/'summary.json')
    assert s['commands']==s['physics_steps']==0 and s['supervised_Adam_calls']==600
    proof=DOC/'T332_PARTIAL_AUDIT.json';assert read(proof)['state']=='PASS_PARTIAL_EVIDENCE_ONLY'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T332.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p.update(experiment='T333_REMAINING_ONLY_T332_RECOVERY',previous_run=str(old),
        cached_control_patch=str(old/'control_candidate.pt.gz'),
        absolute_deadline_epoch=s['deadline_epoch'],absolute_training_deadline_epoch=s['main_epoch']+p['training_seconds'],
        final_bytes=p['final_bytes']-K.S.used(old),prior_supervised_Adam_calls=600,remaining_supervised_Adam_calls=600)
    assert time.time()<p['absolute_training_deadline_epoch']-300
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}
    files|={p['cached_control_patch'],str(old/'summary.json'),str(old/'optimizer_calls.jsonl'),str(proof),str(receipt),str(DOC/'T333_PLAN.md')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED_REMAINING_ONLY',absolute_deadline=p['absolute_deadline_epoch'],combined_own_bytes=p['final_bytes']+K.S.used(old))))

if __name__=='__main__':main()

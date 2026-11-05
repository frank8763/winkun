"""Check exact whole-history membership of the parent's actual successful runs."""
from t372_core import *


def main():
    start=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Protection support60s')));signal.alarm(60)
    torch.set_num_threads(1);p=read(HERE/'protocol.json');q=NightActor(p).eval().requires_grad_(False);old,g,factual=dataset(p,q,lambda:None)
    protected=[dict(id=v['id'],states_sha256=digest(v['x'].numpy()),decisions=len(v['x'])) for v in [g,*old,*factual]]
    lookup={v['states_sha256']:v['id'] for v in protected};rows=[r for r in read(p['baseline_cases_path']) if r['role']=='initial' and r['success']]
    assert len(rows)==14;items=[]
    for r in rows:
        assert K.C.sha(r['path'])==r['sha256']
        items.append(dict(job_id=r['job_id'],factual_parent_states_sha256=r['array_hashes']['states'],whole_history_in_training_protection=lookup.get(r['array_hashes']['states']),baseline_full_states_saved=r['full_arrays_saved']))
    result=dict(state='READONLY_SUPPORT_AUDIT_COMPLETE',new_native=0,gradient_calls=0,parameter_fits=0,optimizer=0,elapsed_s=time.time()-start,actual_parent_successes=14,protection_histories=len(protected),actual_success_whole_histories_matched=sum(v['whole_history_in_training_protection'] is not None for v in items),actual_success_full_states_saved_in_baseline=sum(v['baseline_full_states_saved'] for v in items),items=items,protocol_sha256=K.C.sha(HERE/'protocol.json'),baseline_sha256=p['baseline_cases_sha256'],scope='Whole-history exact SHA membership only, not proof that all individual states are absent, reference states are useless, or this mismatch uniquely caused failure. Parent outputs were correctly evaluated on those other recorded histories.')
    save(DOC/'T372_PROTECTION_SUPPORT.json',result);signal.alarm(0);print(json.dumps(result),flush=True)


if __name__=='__main__':main()

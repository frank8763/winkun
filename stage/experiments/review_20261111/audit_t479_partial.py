"""Independent partial-evidence admission review, no native or new policy.

Reconstruct exactly the historical 64 discarded Adam calls which the normal
T479 audit did not reach. Preserve T479 FAILED summary and every original file.
The 47 native substeps absent from the final partial decision arrays remain an
explicit evidence gap; no assertion here fabricates their full reconstruction.
"""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t479_balanced_success_sil'))
import t479_runtime as T
L,P,E,M,R,torch,np=T.L,T.P,T.E,T.M,T.R,T.torch,T.np


def main():
    start=time.time();pdir=ROOT/'stage/runs/t479_balanced_success_sil';doc=ROOT/'stage/docs/review_20261111'
    plan=T.read(doc/'T479_PARTIAL_REVIEW_PROTOCOL.json')
    assert plan['native']==0 and plan['max_historical_discarded_Adam']==64 and time.time()<plan['all_stop_epoch']
    output=doc/'T479_PARTIAL_REVIEW.json';assert not output.exists()
    p=T.read(pdir/'protocol.json');s=T.read(pdir/'pair/summary.json');cap=T.read(pdir/'capture/summary.json')['result']
    assert s['state']=='FAILED' and T.read(pdir/'capture/audit.json')['state']=='PASS'
    for f,h in p['inputs'].items():assert T.sha(f)==h,f
    for name in ['launch.json','driver_started.json','SEALED.json']:
        d=T.read(pdir/name);ids=d.get('scientific_pids',[])+([d['pid']] if d.get('pid') else [])
        assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T479.json'
    rc=T.read(receipt);assert rc['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    manifest=T.read(receipt.with_name('source_manifest_T479.json'));archived={r['path']:r for r in manifest['files']}
    assert T.sha(receipt.with_name('source_manifest_T479.json'))==rc['manifest_sha256']
    rows=T.rows(pdir/'pair')
    for r in rows:
        assert T.sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        assert all(T.digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert int(a['native_counts'].sum())==r['physics']
        assert a['end_flags'][-1].tolist()==[int(r['collision']),int(r['success']),int(r['timeout'])]
        rel=str(Path(r['path']).relative_to(ROOT));assert archived[rel]['sha256']==r['sha256']
    refs=L.references(p,cap['success_rows']);objects={arm:L.make(p) for arm in ['A','B']}
    memory=L.success_memory(p,next(objects['A'][0].parameters()).device)
    def guard():assert time.time()<min(start+plan['max_seconds'],plan['all_stop_epoch'])-5
    update=L.fit_pair(objects,[E.N.S.arrays(r['path']) for r in cap['samples']],p,refs,memory,guard,lambda *_:None)
    assert update['matched_epochs']==16
    raw_events=[json.loads(x) for x in (pdir/'pair/optimizer_calls.jsonl').read_text().splitlines()]
    for which in ['live','shadow']:
        for arm in ['A','B']:assert [x[3] for x in raw_events if x[:3]==[which,arm,'AFTER']]==list(range(16))
    models={}
    for arm,obj in objects.items():
        f=pdir/f'pair/proposal_{arm}.pt.gz';saved=P.checkpoint(f)
        assert T.digest(P.state(*obj))==saved['joint_hash']==T.digest(saved['joint_state'])
        assert obj[0].phash()==saved['policy_hash'];models[arm]=dict(sha256=T.sha(f),joint_hash=saved['joint_hash'])
    rollback=T.read(pdir/'pair/failure.json')['interrupted_entire_Actor_Value_both_Adam_rollback']
    assert set(rollback)=={'liveA','liveB','shadowA','shadowB'}
    assert all(h==p['initial']['joint_hash'] for h in rollback.values())
    for obj in objects.values():P.load(obj,P.checkpoint(p['initial']['path'])['joint_state']);assert T.digest(P.state(*obj))==p['initial']['joint_hash']
    candidates={}
    for arm in ['A','B']:
        rr=[r for r in rows if r['role']=='FULL383_'+arm];success=set(T.T.safe(rr));executed={r['job_id'] for r in rr}
        lost=sorted((set(p['initial_success_ids'])&executed)-success)
        candidates[arm]=dict(valid_executed=len(rr),eight=R.M.tally(rr[:54]),six=R.M.tally(rr[54:108]),
            old_executed=R.M.tally(rr[108:]),already_lost_success_ids=lost,
            acceptable=False,complete=len(rr)==383,unexecuted_or_censored=383-len(rr))
        assert lost
    assert candidates['A']['valid_executed']==383 and candidates['B']['valid_executed']==334
    partial=[];whole_partial_steps=0
    for f in sorted((pdir/'pair/workers').glob('*/partial.npz')):
        a=E.N.S.arrays(f);n=int(a['native_counts'].sum());whole_partial_steps+=n
        assert not a['end_flags'].any() and n>0
        r=archived[str(f.relative_to(ROOT))];assert T.sha(f)==r['sha256']
        partial.append(dict(path=str(f.relative_to(ROOT)),sha256=T.sha(f),completed_decisions=len(a['native_counts']),completed_hold_native_steps=n,administrative_censor=True))
    raw_steps=sum(r['physics'] for r in rows);gap=s['physics']-raw_steps-whole_partial_steps
    assert len(partial)==2 and whole_partial_steps==10700 and gap==47
    assert s['commands']==len(rows)+4+2
    counters=sum(T.read(f)['physics'] for f in (pdir/'pair/workers').glob('*/failure.json'))
    assert counters==s['physics']
    result=dict(state='PASS_PARTIAL_ADMISSION_REVIEW_NOT_COMPLETE_T479_AUDIT',epoch=time.time(),native=0,
        historical_discarded_Adam=4*update['matched_epochs'],new_policy_updates=0,
        source_sha256=T.sha(__file__),protocol_sha256=T.sha(pdir/'protocol.json'),
        original_FAILED_summary_unchanged=True,original_full_B_missing=True,
        capture_independent_audit_and_all_source_inputs_valid=True,proposal_models_exactly_reconstructed=models,
        initial_model_unchanged_sha256=T.sha(p['initial']['path']),joint_rollback_hash=p['initial']['joint_hash'],
        candidates=candidates,partial=partial,valid_rows_physics=raw_steps,
        complete_partial_hold_physics=whole_partial_steps,unrepresented_in_progress_hold_native_steps=gap,
        physical_total_including_gap=s['physics'],command_total=s['commands'],
        B_valid=334,B_censored=2,B_not_started=47,
        fresh16=None,reload78=None,selected=p['initial_result'],
        full_raw_remote_receipt_sha256=T.sha(receipt),
        limitations=['47 native substeps executed inside unfinished holds are counted but lack complete decision arrays.',
                    'This is not a complete T479 pair audit or full B result.',
                    'Retained original model and independent capture, not rejected proposals, are eligible for a new mechanism.'],elapsed_s=time.time()-start)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['state','historical_discarded_Adam','physical_total_including_gap','B_valid','B_censored','B_not_started','elapsed_s']}))


if __name__=='__main__':main()

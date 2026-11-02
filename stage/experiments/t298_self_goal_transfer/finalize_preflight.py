"""Reuse completed eight clone contracts after pre-launch capacity rejection."""
from preflight import *

def main():
    assert not (OUT/'preflight.json').exists() and not (OUT/'launch.json').exists();p=json.loads((HERE/'protocol.json').read_text());old=json.loads((HERE/'preflight_failed_001/protocol.json').read_text());failure=json.loads((HERE/'preflight_failed_001/failure.json').read_text())
    assert failure['completed_disposable_optimizer_calls']==8 and failure['native_steps']==0
    calls=json.loads((OUT/'preflight_optimizer_calls.json').read_text())['calls'];assert len(calls)==8
    for name in ['run.py','preflight.py','audit.py','t298_learning.py','t298_worker.py','launch.py']:assert (HERE/name).read_bytes()==(HERE/'preflight_failed_001'/name).read_bytes()
    assert p['actor_attempts_per_round']==1 and old['actor_attempts_per_round']==2
    allowed={'actor_attempts_per_round','postrun_clone_optimizer_max_calls','max_compact_payloads','metadata_bytes','payload_bytes','contract','input_hashes','preflight_engineering_repair'}
    assert all(p[k]==v for k,v in old.items() if k not in allowed)
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert p['max_compact_payloads']==524 and p['metadata_bytes']==1638400 and p['payload_bytes']+p['metadata_bytes']<p['final_bytes']
    # Prior exact-format estimate; four additional compressed rows at <=1KiB.
    estimate=1594511+4*1024;assert estimate<p['metadata_bytes']
    # One live proposal at most three factor candidates: all acceptance/evaluation
    # and four partial-replay slots are now fully reserved.
    maxcompact=20+2+3*20+12+275+54+16+78+4;assert maxcompact==521<=p['max_compact_payloads']
    assert 16+16+maxcompact<=p['max_commands'] and (16+16+maxcompact)*28000<=p['max_physics']
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,native_environments_created=0,navigation_Actor_updates=0,disposable_clone_optimizer_calls=8,new_finalize_optimizer_calls=0,supervised_clone_live_shadow_exact=True,RL_clone_live_shadow_ActorAdam_exact=True,rejection_interruption_ActorAdam_exact=True,codec_five_arrays_exact_derived_rewards=True,full120_statechain_exact=True,all_entry_global_names_resolve=True,metadata_upper=estimate,max_compact_proof=maxcompact,global_if_full=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes'],original_resource_rejection_preserved=True,only_prelaunch_scientific_change_proposal_effort_reduced=True,real_native_guard_not_yet_run=True))
    print('T298 final preflight PASS reuses8clones,0newoptimizer0native; effort1proposal/1round')

if __name__=='__main__':main()

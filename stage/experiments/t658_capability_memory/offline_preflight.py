"""Standard-library-only fixtures and audited reference union, zero training."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,math,time
from continuation_contract import (ContinuationKey,OutcomeSample,IncompatibleContinuation,
    unique_reference_union,finite_validated_lower_bound,finite_horizon_reach_avoid,
    exact_difference_decomposition)
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/day_20261119/innovation_reset'

def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(name,value):(DOC/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    start=time.monotonic();DOC.mkdir(parents=True,exist_ok=True)
    # Exact two-step counterexample: identical first action and old-policy
    # continuation value1, but a changed continuation fails at the next state.
    def transition(s,a):
        if s=='start':return 'fork'
        if s=='fork':return 'goal' if a=='turn' else 'collision'
        return s
    def old(s,h):return 'forward' if s=='start' else 'turn'
    def new(s,h):return 'forward'
    goal=lambda s:s=='goal';fail=lambda s:s=='collision'
    qold=finite_horizon_reach_avoid(old,'fork',1,transition,goal,fail)
    qnew=finite_horizon_reach_avoid(new,'fork',1,transition,goal,fail)
    assert old('start',2)==new('start',2) and qold==1 and qnew==0
    delta=exact_difference_decomposition(old,new,'start',2,transition,goal,fail)
    assert delta['difference']==-1. and delta['first_action_reference_term']==0.
    assert delta['future_difference']==-1.
    # A scalar coordinate represents a change to every future policy decision.
    # Reach-avoid feasibility need not be monotone in that coordinate. This is
    # a deliberately synthetic two-step model, not a measured Stage response.
    def family(alpha):
        def policy(s,n):return alpha if s=='fork' else 0.
        return policy
    def family_transition(s,a):
        if s=='start':return 'fork'
        if s=='fork':return 'goal' if a<=.2 or .7<=a<=.8 else 'collision'
        return s
    alpha_values=[0.,.5,.75,1.]
    family_outcomes=[finite_horizon_reach_avoid(family(a),'start',2,family_transition,goal,fail) for a in alpha_values]
    assert family_outcomes==[1.,0.,1.,0.]
    # Distinct safe actions may be averaged into an unsafe action. This is a
    # mathematical fixture, not proof that Stage's actions are bimodal.
    safe=lambda a:abs(a)>=.5
    assert safe(-1.) and safe(1.) and not safe(0.)
    key=ContinuationKey('old','deterministic_clip_mu','goal','dynamics','history',100)
    label=OutcomeSample(key,True,True,False,'source')
    assert label.value_for_exact_continuation(key)==1.
    rejected=[]
    for name,value in [('policy_hash','new'),('execution_mode','stochastic_clipped_gaussian'),('goal_key','other'),('dynamics_key','changed'),('causal_history_key','other'),('remaining_native_steps',99)]:
        try:label.value_for_exact_continuation(replace(key,**{name:value}));raise AssertionError('scope leak')
        except IncompatibleContinuation:rejected.append(name)
    try:replace(label,complete=False,administrative_censor=True).value_for_exact_continuation(key);raise AssertionError('censor leak')
    except IncompatibleContinuation:pass
    try:finite_validated_lower_bound(.99,None);raise AssertionError('invented certificate')
    except ValueError:pass
    assert finite_validated_lower_bound(.9,.2)==.7
    # Real sources: recheck all immutable canonical NPZ hashes; no physics,
    # neural forward/backward, optimizer, GPU or remote data access.
    p=read(ROOT/'stage/runs/t656_qualified_recovery_pg/protocol.json')
    original=read(ROOT/'stage/runs/t646_offline_night/OFFLINE_INPUTS.json')
    oldrefs={r['id']:dict(task_id=r['id'],source_path=r['offline_path'],sha256=r['sha256'],source_policy=p['safe_retained']['policy_hash'],
        origin='T501_original_success',evidence='prior_full_history_audit_and_current_byte_hash') for r in original['members']}
    newrefs={r['job_id']:dict(task_id=r['job_id'],source_path=r['path'],sha256=r['sha256'],source_policy=r['policy_hash'],
        origin='T653_mean_success',evidence='prior_full_history_audit_and_current_byte_hash') for r in p['field_keep_rows']}
    assert len(oldrefs)==384 and len(newrefs)==8
    refs,weights=unique_reference_union(oldrefs,newrefs);assert len(refs)==387 and len(set(oldrefs)&set(newrefs))==5
    assert abs(sum(weights.values())-1)<1e-12 and len(set(weights.values()))==1
    for row in [*oldrefs.values(),*newrefs.values()]:
        assert time.monotonic()-start<120
        assert sha(row['source_path'])==row['sha256'],row['source_path']
    for key in set(oldrefs)&set(newrefs):assert refs[key]==oldrefs[key]
    assert set(newrefs)<=set(refs) and set(oldrefs)<=set(refs)
    rows=[dict(refs[key],weight=weights[key]) for key in sorted(refs)]
    manifest=dict(state='PREPARED_REFERENCE_CONTRACT_ONLY_NOT_TRAINED',required_success_ids=sorted(refs),reference_rows=rows,
        removed_duplicate_task_count=5,old384_plus_new3=True,all8_field_goal_requirements_retained=True,
        frozen_model_unchanged=True,new_native=0,new_optimizer=0)
    save('UNIQUE_387_REFERENCE_CONTRACT.json',manifest)
    report=dict(state='OFFLINE_CONTRACT_FIXTURES_PASS_NOT_A_NAVIGATION_RESULT',new_native=0,new_optimizer=0,new_Critic=0,new_Actor=0,
        exact_toy_old_continuation_success=qold,exact_toy_new_continuation_success=qnew,same_initial_action_in_toy=True,
        exact_standard_bellman_difference=delta,
        synthetic_policy_family_coordinates=alpha_values,synthetic_policy_family_outcomes=family_outcomes,
        policy_family_safety_monotonicity_not_assumed=True,
        counterexamples_are_synthetic_not_Stage=True,critic_scope_mismatches_rejected=rejected,censored_labels_rejected=True,
        no_certificate_without_external_error_bound=True,safe_action_average_counterexample=[-1,1,0],
        required_task_count=387,real_source_files_hash_checked=392,weight_per_unique_task=1/387,
        prototype_only=True,scientific_training_not_started=True,elapsed_s=time.monotonic()-start)
    save('OFFLINE_CONTRACT_PREFLIGHT.json',report)
    assert sum(f.stat().st_size for f in DOC.glob('*') if f.is_file())<4*2**20
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()

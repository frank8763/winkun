"""Offline provenance contract for future finite-horizon capability learning.

No learner, simulator, planner, controller switch or certificate is implemented.
Observed Monte Carlo outcomes are samples, not guaranteed action values.
"""
from dataclasses import dataclass
from math import isfinite

class IncompatibleContinuation(ValueError):
    pass

@dataclass(frozen=True)
class ContinuationKey:
    policy_hash: str
    execution_mode: str
    goal_key: str
    dynamics_key: str
    causal_history_key: str
    remaining_native_steps: int

    def __post_init__(self):
        if self.execution_mode not in {'deterministic_clip_mu','stochastic_clipped_gaussian'}:
            raise ValueError('Unknown execution mode')
        if self.remaining_native_steps<0:
            raise ValueError('Remaining physical horizon must be nonnegative')
        if not all([self.policy_hash,self.goal_key,self.dynamics_key,self.causal_history_key]):
            raise ValueError('Continuation identity must be explicit')

@dataclass(frozen=True)
class OutcomeSample:
    key: ContinuationKey
    safe_goal: bool
    complete: bool
    administrative_censor: bool
    source_sha256: str

    def value_for_exact_continuation(self,query: ContinuationKey):
        """This only permits reuse of an observed outcome for the SAME query.

        It does not estimate its probability or certify other states. Old
        transitions may still support an explicitly re-targeted Bellman update.
        """
        if not self.complete or self.administrative_censor:
            raise IncompatibleContinuation('Censored outcome is not failure or success')
        if query!=self.key:
            changed=[name for name in self.key.__dataclass_fields__ if getattr(query,name)!=getattr(self.key,name)]
            raise IncompatibleContinuation('Observed suffix is not this query: '+','.join(changed))
        return float(self.safe_goal)


def unique_reference_union(old_by_id,new_by_id):
    """One predeclared reference per task; old reference wins shared IDs.

    All new task IDs remain required. This is target bookkeeping, not a
    mechanism that guarantees joint fit or closed-loop task preservation.
    """
    if not old_by_id or not new_by_id:
        raise ValueError('Both sources must be explicit')
    refs={**new_by_id,**old_by_id}
    required=set(old_by_id)|set(new_by_id)
    return refs,{key:1/len(required) for key in sorted(required)}


def finite_validated_lower_bound(candidate_estimate,uniform_error_bound):
    """Only arithmetic; caller must independently establish the error bound."""
    if uniform_error_bound is None:
        raise ValueError('No validated error bound: no certificate')
    if not isfinite(candidate_estimate) or not isfinite(uniform_error_bound) or uniform_error_bound<0:
        raise ValueError('Invalid value or error bound')
    return candidate_estimate-uniform_error_bound


def finite_horizon_reach_avoid(policy,start,horizon,transition,goal,failure):
    """Exact enumeration of a supplied small deterministic model, NOT Stage."""
    state=start
    for remaining in range(horizon,-1,-1):
        if failure(state):return 0.
        if goal(state):return 1.
        if remaining==0:return 0.
        state=transition(state,policy(state,remaining))
    raise AssertionError('unreachable')


def exact_difference_decomposition(reference,candidate,start,horizon,transition,goal,failure):
    """An exact finite deterministic Bellman fixture, not a learned operator.

    For the full candidate continuation, D(h,n) = V_candidate - V_reference
    = [V_reference(next_candidate,n-1) - V_reference(h,n)]
      + D(next_candidate,n-1).

    The identity is the standard policy-performance/Bellman decomposition;
    neither it nor this implementation is claimed as a new theorem. A Stage
    history would include the complete causal prefix, with each actor's own
    carry. A toy state here is not a claim that a Stage observation is Markov.
    """
    if horizon<0:
        raise ValueError('Negative horizon')
    def value(policy,state,n):
        return finite_horizon_reach_avoid(policy,state,n,transition,goal,failure)
    def walk(state,n):
        reference_value=value(reference,state,n)
        candidate_value=value(candidate,state,n)
        if failure(state) or goal(state) or n==0:
            return dict(reference_value=reference_value,candidate_value=candidate_value,
                        difference=0.,first_action_reference_term=0.,future_difference=0.)
        next_state=transition(state,candidate(state,n))
        local=value(reference,next_state,n-1)-reference_value
        future=walk(next_state,n-1)['difference']
        difference=local+future
        if abs(difference-(candidate_value-reference_value))>1e-12:
            raise AssertionError('Bellman difference identity failed')
        return dict(reference_value=reference_value,candidate_value=candidate_value,
                    difference=difference,first_action_reference_term=local,
                    future_difference=future)
    return walk(start,horizon)

"""Five fixed lost histories: zero-native action-history sensitivity attribution.

The intervention changes only previous-command inputs on recorded histories.
It is deliberately NOT a physically consistent rollout or recovery label.
"""
from pathlib import Path
import time
import t581_learning as L
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t581_preserved_t550_rl';DOC=ROOT/'stage/docs/day_20261116/t581'
p=L.read(OUT/'protocol.json');s=L.read(OUT/'summary.json');a=L.read(OUT/'audit.json')
assert s['state']=='COMPLETE' and a['state']=='PASS'
assert not any(Path('/proc',str(i)).exists() for i in L.read(OUT/'SEALED.json')['scientific_pids'])
assert not (DOC/'POSTHOC_SUPPORT.json').exists()
L.torch.set_num_threads(1);L.torch.use_deterministic_algorithms(True)
q=L.ORIGINAL_FROZEN(p,p['base_model']['path']);candidate=L.frozen(p,s['result']['proposal']['path'])
initial=L.digest(q.state_dict());initial_candidate=L.digest(candidate.state_dict())
np,torch=L.np,L.torch

def supports(q,x):
    f,_=L.feature(q,x);result=[]
    layers=[('repair',q.repair_centers,q.repair_radii),('added',q.added_centers,q.added_radii)]+[(str(i),z.centers,z.radii) for i,z in enumerate(q.layers)]
    for name,centers,radii in layers:
        if len(radii)==0:continue
        normalized=[]
        for k in range(0,len(f),16):
            d=((f[k:k+16,None]-centers)**2).sum(-1)/radii.square()
            normalized.append(d.min(-1).values)
        ratio=torch.cat(normalized)
        result.append(dict(layer=name,centers=len(radii),radius_min=float(radii.min()),radius_median=float(radii.median()),
            radius_max=float(radii.max()),active=(ratio<1).numpy(),nearest_ratio=ratio.numpy()))
    return result

results=[]
with torch.no_grad():
    for ident in s['result']['check']['lost_initial']:
        row=next(v for v in p['keep_rows'] if v['job_id']==ident);assert L.sha(row['path'])==row['sha256']
        arrays=L.G.E.N.S.arrays(row['path']);x=torch.tensor(arrays['states']);assert np.array_equal(arrays['states'][1:,117:119],arrays['actions'][:-1])
        y=q.sequence(x);c=candidate.sequence(x)
        own=c.clamp(-1,1).float();changed=x.clone();changed[1:,117:119]=own[:-1]
        yp=q.sequence(changed);original=supports(q,x);perturbed=supports(q,changed)
        details=[]
        for before,after in zip(original,perturbed):
            active=before.pop('active');nearest=before.pop('nearest_ratio');other=after['active']
            details.append(dict(**before,original_active_decisions=int(active.sum()),
                intervention_active_decisions=int(other.sum()),lost_active_decisions=int((active&~other).sum()),
                original_nearest_ratio_median=float(np.median(nearest)),intervention_nearest_ratio_median=float(np.median(after['nearest_ratio']))))
        results.append(dict(id=ident,source_sha256=row['sha256'],decisions=len(x),
            candidate_location_change_on_original_inputs_RMS=float((c-y).square().mean().sqrt()),
            candidate_location_change_on_original_inputs_max=float((c-y).abs().max()),
            input_previous_command_max_change=float((changed-x).abs().max()),
            original_actor_response_to_previous_command_intervention_RMS=float((yp-y).square().mean().sqrt()),
            original_actor_response_to_previous_command_intervention_max=float((yp-y).abs().max()),basis_support=details))
assert L.digest(q.state_dict())==initial and L.digest(candidate.state_dict())==initial_candidate
report=dict(state='COMPLETE_ZERO_NATIVE_ZERO_OPTIMIZER',new_native=0,new_optimizer=0,
    new_model_selection=False,fixed_five_actual_lost_IDs=True,models_unchanged=True,
    synthetic_input_intervention_NOT_physical_rollout=True,does_not_prove_unique_cause_of_field_failures=True,
    result=results,training_outcomes=s['result']['training'],peer_credit=s['result']['update']['peer_batch'],
    initial_function_native_verified=L.read(OUT/'INITIAL_FUNCTION_NATIVE_PASS.json'),
    final_KL=s['result']['update']['trace'][-1]['KL'],epoch=time.time())
L.save(DOC/'POSTHOC_SUPPORT.json',report)
for x in results:print(x['id'],x['candidate_location_change_on_original_inputs_max'],x['original_actor_response_to_previous_command_intervention_max'],[(z['layer'],z['original_active_decisions'],z['lost_active_decisions'],z['radius_min']) for z in x['basis_support']])

"""No-physics safeguard/ablation wiring check; no optimizer calls."""
from support import *
from learner import verify_behavior
def main():
    assert not (OUT/'contracts.json').exists()
    plan=json.loads((HERE/'protocol.json').read_text())
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    torch.set_num_threads(1)
    p=policy(torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'])
    with np.load(plan['retention_targets']) as z:
        with torch.no_grad():err=float((p(torch.tensor(z['states']))-torch.tensor(z['actions'])).abs().max())
    assert err==0 and plan['retention_weight']==0 and plan['max_behavior_mean_KL']==.01 and plan['max_parent_anchor_KL']==1e9 and plan['max_balanced_anchor_MSE']==1e9
    identity=[]
    for row in plan['initial_identity_controls']:
        with np.load(row['trace']) as z:identity.append(verify_behavior(p,torch.tensor(z['states']),torch.tensor(z['actions']),torch.tensor(z['means']),torch.tensor(z['logp']),plan))
    result={'state':'PASS','protocol_sha256':C.sha(HERE/'protocol.json'),'physics_steps':0,'optimizer_calls':0,
            'initial_parent_action_max_error':err,'behavior_checks':identity,'retention_disabled':True,'behavior_trust_region_retained':True}
    C.save(OUT/'contracts.json',result);print(json.dumps(result))
if __name__=='__main__':main()

"""Audit completed paired outcomes and policy changes without more simulation."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *

def main():
    torch.set_num_threads(1)
    summary=json.loads((OUT/'summary.json').read_text());assert summary['state']=='COMPLETE'
    episodes=[json.loads(s) for s in (OUT/'episodes.jsonl').read_text().splitlines()]
    evaluation=[json.loads(s) for s in (OUT/'evaluation.jsonl').read_text().splitlines()]
    plan=json.loads((HERE/'protocol.json').read_text())
    for path,digest in plan['input_hashes'].items():assert sha(path)==digest,path
    initial=Policy().eval();initial.load_state_dict(torch.load(plan['initial_policy_checkpoint'],map_location='cpu',weights_only=False)['policy'])
    baseline=episodes[:8];first_new=episodes[64:72]
    for a,b in zip(baseline,first_new):
        for k in ['initial_reset_signature','success','collision','steps','final_pose']:assert a[k]==b[k],k
    groups={}
    with np.load(OUT/'frozen_head/batch_0.npz') as z:
        x=z['states'];offset=0
        for row in baseline:
            groups.setdefault(row['reset_group'],[]).append(x[offset:offset+row['decisions']]);offset+=row['decisions']
        assert offset==len(x)
    with np.load(plan['retention_demonstrations']) as z:
        states=z['states'];counts=z['native_counts'];offsets=z['episode_offsets'];clock=[]
        for j in range(6):
            lo,hi=offsets[j:j+2];clock.extend(1-np.r_[0,np.cumsum(counts[lo:hi-1])]/plan['primary_cases'][j]['step_cap'])
        groups['old6_demonstration']=[np.column_stack([states,np.asarray(clock,dtype=np.float32)])]
    changes={}
    for arm,spec in summary['training'].items():
        p=Policy().eval();p.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)['policy']);changes[arm]={}
        for name,arrays in groups.items():
            x=torch.tensor(np.concatenate(arrays),dtype=torch.float32)
            with torch.no_grad():
                delta=torch.cat([p(t)-initial(t) for t in x.split(128)])
                changes[arm][name]={'n_states':len(x),'rms_per_action':delta.square().mean(0).sqrt().tolist(),'max_absolute_per_action':delta.abs().max(0).values.tolist()}
    original=[]
    for row in evaluation:
        if row['group']!='original_5m':continue
        record={k:row[k] for k in ['model','id','success','collision','timeout','steps','nearest_m','remaining_m','native_mean_min','native_mean_max','zero_drive_decisions','low_drive_decisions','wheel_command_quantiles','initial_reset_signature']}
        record['obstacle_contacts']=[c for c in row['terminal_contacts'] if any(str(c[k]).startswith('real_') for k in ['geom1','geom2'])]
        original.append(record)
    result={'state':'PASS','initial_first_batch_matched_exactly':True,'training_physics':sum(r['steps'] for r in episodes),'training_decisions':sum(r['decisions'] for r in episodes),'training_episodes':len(episodes),'optimizer_updates':sum(v['actor_updates'] for v in summary['training'].values()),'evaluation_cases':len(evaluation),'evaluation_physics':sum(r['steps'] for r in evaluation),'original5m':original,'mean_policy_changes_on_common_states':changes,'scope':'Offline audit only. Ground contacts in raw T189 telemetry are filtered only for this obstacle summary; raw rows preserved. Nonzero learned head and capacity do not imply learned zero drive or navigation gain. Same first batch is paired evidence, not an independent success trial.'}
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()

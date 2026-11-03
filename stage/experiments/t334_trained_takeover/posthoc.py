"""Read-only descriptive action errors on consumed human correction states."""
from run import *

def main():
    p=read(HERE/'protocol.json');verify(p);assert read(DOC/'T334_AUDIT.json')['state']=='PASS'
    dest=DOC/'T334_POSTHOC_ACTION_ERRORS.json';assert not dest.exists();torch.set_num_threads(1)
    ref,_,_=M.R.load_arrays(p['reference']);x=ref['states'][150:];target=ref['actions'][150:]
    changed=target[:,1]!=x[:,-2];results={}
    for arm in ['control','recovery']:
        q=policy(p,arm)
        with torch.no_grad():pred=L.mode(q,torch.tensor(x)).numpy()
        error=(pred-target)**2
        results[arm]=dict(all_mse=error.mean(0).tolist(),first10_mse=error[:10].mean(0).tolist(),
            steer_change_mse=float(error[changed,1].mean()),steer_hold_mse=float(error[~changed,1].mean()),
            handoff_human_action=target[0].tolist(),handoff_Actor_action=pred[0].tolist(),
            first10_human_steer=target[:10,1].tolist(),first10_Actor_steer=pred[:10,1].tolist())
    save(dest,dict(posthoc=True,native_added=0,gradient_calls=0,optimizer_calls=0,
        consumed_training_correction_rows=len(x),change_rows=int(changed.sum()),
        not_independent_validation_or_physical_causal_proof=True,results=results))
    print(L.json.dumps(results))

if __name__=='__main__':main()

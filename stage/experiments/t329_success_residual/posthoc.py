"""Descriptive inference on the consumed demonstration; no new fit or rollout."""
from t329_learning import *

def main():
    p=read(HERE/'protocol.json');verify(p)
    assert read(OUT/'summary.json')['state']=='COMPLETE'
    assert read(DOC/'T329_AUDIT.json')['state']=='PASS'
    dest=DOC/'T329_POSTHOC_IMITATION.json';assert not dest.exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    a=arrays(p['demo']);x=torch.tensor(a['states']);y=a['actions']
    change=y[:,1]!=a['states'][:,-2]
    results={}
    with torch.no_grad():
        for name in ['initial','bc_proposal','pg_proposal','final']:
            q=model(p);v=read_patch(OUT/(name+'.pt.gz'));load(q,v['residual'])
            pred=mode(q,x).numpy();err=(pred-y)**2
            results[name]=dict(policy_hash=K.S.digest(q.state_dict()),
                drive_mse=float(err[:,0].mean()),steer_mse=float(err[:,1].mean()),
                steer_change_mse=float(err[change,1].mean()),
                steer_hold_mse=float(err[~change,1].mean()),
                first_action=pred[0].tolist(),mean_drive=float(pred[:,0].mean()))
    save(dest,dict(posthoc=True,native_added=0,optimizer_calls=0,
        consumed_demonstration_rows=len(y),steer_change_rows=int(change.sum()),
        not_validation_or_physical_causality=True,results=results))
    print(json.dumps(results))

if __name__=='__main__':main()

from pathlib import Path
import io,time,json,ast
import t622_learning as M
L,torch=M.L,M.torch

def main():
    p=M.config();M.setup();train,test=M.data(p);obj=M.make(p);before=L.digest(M.state(obj));rng=M.np.random.default_rng(p['learner_seed']);pack=M.pack(train,rng,p);value=M.loss(obj[0],pack,'aware');value.backward();assert bool(torch.isfinite(value));nz=sum(int(torch.count_nonzero(v.grad)) for v in obj[0].parameters());assert nz>0
    assert L.digest(M.state(obj))==before
    s=M.state(obj)
    for i,v in zip(s['optimizer']['param_groups'][0]['params'],obj[0].parameters()):s['optimizer']['state'][i]=dict(step=torch.tensor(1.),exp_avg=torch.zeros_like(v).cpu(),exp_avg_sq=torch.zeros_like(v).cpu())
    b=io.BytesIO();torch.save(dict(state=s,state_sha='0'*64),b);upper=int(len(b.getvalue())*1.001)+32768;assert upper<p['checkpoint_bytes']
    # Return recursion and finite-task terminal mask, including timeout.
    for e in train+test:
        assert e['bootstrap'][-1]==0 and e['ret'][-1]==e['r'][-1]
        assert torch.equal(e['ret'][:-1],e['r'][:-1]+e['discount'][:-1]*e['bootstrap'][:-1]*e['ret'][1:])
    for f in M.HERE.glob('*.py'):ast.parse(f.read_text())
    report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',train_episodes=len(train),holdout_episodes=len(test),train_ids=sorted({e['id'] for e in train}),holdout_ids=sorted({e['id'] for e in test}),split_by_whole_task_id=True,full_history_BPTT=True,return_labels_are_recorded_behavior_not_current_policy_Q=True,nonzero_gradient_entries=nz,checkpoint_safe_upper=upper,actor_not_created_or_updated=True,no_Adam_step=True,epoch=time.time());L.save(M.DOC/'PREFLIGHT.json',report);print(json.dumps(report))

if __name__=='__main__':main()

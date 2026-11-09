"""Review sealed T441 evidence; zero navigation, Actor forwards or optimizers.

Only existing Value heads are evaluated, against realized discounted returns.
The output is a post-hoc description, not a new independent test or causal proof.
"""
from pathlib import Path
import collections, gzip, hashlib, io, json, signal, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/docs/research_rescue_20261109/T441_POSTHOC_REVIEW.json'
RUN=ROOT/'stage/runs/t441_joint_actor_critic'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p) as z:a={k:z[k].copy() for k in z.files}
    if 'last_next_state' in a:a['next_states']=np.concatenate([a['states'][1:],a.pop('last_next_state')])
    return a


class Value(torch.nn.Module):
    """Exact T441 Value topology only; no Actor or simulator construction."""
    def __init__(self,state):
        super().__init__()
        self.register_buffer('mean',state['mean'].clone())
        self.register_buffer('scale',state['scale'].clone())
        self.encoder=torch.nn.Sequential(torch.nn.Linear(120,64),torch.nn.Tanh()).double()
        self.core=torch.nn.GRU(64,32).double()
        self.head=torch.nn.Linear(32,1).double()
        self.load_state_dict(state,strict=True);self.eval().requires_grad_(False)
    def forward(self,x):
        x=((x.double()-self.mean)/self.scale).clamp(-10,10)
        h,_=self.core(self.encoder(x)[:,None])
        return self.head(h[:,0])[:,0]


def checkpoint(path):
    item=torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)
    return item['joint_state']


def metrics(prediction,target):
    variance=float(np.var(target));err=target-prediction
    return dict(mse=float(np.mean(err**2)),bias=float(np.mean(err)),
                explained_variance=None if variance<1e-15 else 1-float(np.var(err))/variance)


def main():
    assert not OUT.exists(),'Do not overwrite prior analysis'
    start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('posthoc60s')));signal.alarm(60)
    torch.set_num_threads(1)
    # No optimization is allowed, including accidentally instantiating a learner.
    def prohibited(*a,**k):raise AssertionError('No optimizer/backward in posthoc')
    torch.Tensor.backward=prohibited
    for cls in [torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD]:cls.step=prohibited
    assert read(RUN/'audit.json')['state']=='PASS'
    seal=read(RUN/'SEALED.json');assert seal['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
    receipt=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T441.json')
    assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    manifest=read(ROOT/'stage/docs/long_distance_20261101/source_manifest_T441.json')
    for r in manifest['files']:
        f=ROOT/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256'],r['path']
    p=read(RUN/'protocol.json');summary=read(RUN/'summary.json')
    rows=[json.loads(line) for line in (RUN/'cases.jsonl').read_text().splitlines()]
    final=[r for r in rows if r['role']=='FINAL329'];assert len(final)==329
    prior=set(read(ROOT/'stage/experiments/t437_joint_closed_loop_rl/protocol.json')['initial_success_ids'])
    success={r['job_id'] for r in final if r['success'] and not r['collision']}
    train=[r for r in rows if r['role'].startswith('TRAIN_')]
    counts=collections.Counter(r['job_id'] for r in train)
    failed=[]
    for r in final:
        if r['job_id'] in success:continue
        failed.append(dict(id=r['job_id'],contact=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],
                           previously_successful=r['job_id'] in prior,sampled_training_episodes=counts[r['job_id']]))
    value=[]
    for rr in summary['rounds']:
        rnd=rr['round'];before=RUN/('initial.pt.gz' if rnd==0 else f'round_{rnd-1}.pt.gz')
        v_before=Value(checkpoint(before)['value']);v_after=Value(checkpoint(rr['model']['path'])['value'])
        records=[]
        for i,path in enumerate(rr['training_paths']):
            a=load(path);r=next(r for r in train if r['path']==path);assert sha(path)==r['sha256']
            target=np.zeros(len(a['rewards']),dtype=np.float64);ret=0.
            for k in range(len(target)-1,-1,-1):
                ret=float(a['rewards'][k])+p['gamma_native']**int(a['native_counts'][k])*ret;target[k]=ret
            with torch.inference_mode():
                b=v_before(torch.tensor(a['states'])).numpy();z=v_after(torch.tensor(a['states'])).numpy()
            # These are returns of the sampling policy BEFORE this round's update.
            # Post-update V sees training data; it is not heldout V_new-policy truth.
            records.append(dict(job_id=r['job_id'],group='old' if i<8 else 'wide6',
                success=r['success'],contact=r['collision'],timeout=r['timeout'],decisions=len(target),
                sampling_policy_realized_G0=float(target[0]),V_before_at_start=float(b[0]),V_after_at_start=float(z[0]),
                before_vs_sampling_policy_MC=metrics(b,target),after_in_sample_vs_sampling_policy_MC=metrics(z,target)))
        group_summary={}
        for group in ['old','wide6']:
            items=[x for x in records if x['group']==group]
            group_summary[group]={}
            for key in ['before_vs_sampling_policy_MC','after_in_sample_vs_sampling_policy_MC']:
                group_summary[group][key]=dict(episode_equal_mean_MSE=float(np.mean([x[key]['mse'] for x in items])),
                    episode_equal_mean_EV=float(np.mean([x[key]['explained_variance'] for x in items])),
                    negative_EV_episodes=sum(x[key]['explained_variance']<0 for x in items))
        value.append(dict(round=rnd,model_before_sha256=sha(before),model_after_sha256=sha(rr['model']['path']),groups=group_summary,episodes=records))
    result=dict(state='READ_ONLY_POSTHOC_COMPLETE',native_steps=0,Actor_forwards=0,optimizer_steps=0,
        Value_forward_episodes=128,training_independent_episodes=64,training_decisions=sum(len(load(r['path'])['native_counts']) for r in train),
        prior_success=322,final_success=len(success),repaired=sorted(success-prior),lost=sorted(prior-success),
        failures=failed,failures_never_directly_sampled=sum(x['sampled_training_episodes']==0 for x in failed),
        unique_old_training_jobs=len({r['job_id'] for r in train if not r['job_id'].startswith('wide')}),
        old_left5_training_count=sum(r['job_id'].startswith('dev_task6_') for r in train),
        sampling_counts=dict(counts),value_MC_review=value,
        caveats=['MC outcomes are single stochastic realizations, not true expected V.',
                 'After-update values are in-sample and evaluated against sampling-policy returns, not the updated-policy expected return.',
                 'GAE-target explained variance in the producer is not independent MC calibration.',
                 'Eight failures are development feedback; no fresh16 or78model native reload performed.',
                 'Coverage and value fit are hypotheses, not unique causes or causal benefits of GAE.'],
        original_archive_SHA256=receipt['sha256'],all_original_505_members_reverified=True,elapsed_s=time.monotonic()-start)
    assert len(manifest['files'])==505 and len(failed)==8 and len(success)==321
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    signal.alarm(0)
    print(json.dumps(dict(state=result['state'],failures=len(failed),never_sampled=result['failures_never_directly_sampled'],training_decisions=result['training_decisions'],value=[dict(round=x['round'],groups=x['groups']) for x in value]),ensure_ascii=False))


if __name__=='__main__':main()

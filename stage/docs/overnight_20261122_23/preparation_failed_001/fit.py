"""Finite, group-balanced replay baselines with full live/shadow audit."""
import copy, gzip, io, os, resource, sys, time
from common import *
import actor as A
torch,np,digest=A.torch,A.np,A.digest

def data():
    return torch.load(io.BytesIO(gzip.decompress((OUT/'DATA.pt.gz').read_bytes())),map_location='cpu',weights_only=False)

def events(who,trial,stage,step,kind='Adam'):
    with (OUT/'fit/fit_calls.jsonl').open('a') as f:
        f.write(json.dumps(dict(who=who,trial=trial,stage=stage,step=step,kind=kind,epoch=time.time()))+'\n')

def group_loss(head,d,ids):
    totals=[];head.eval()
    with torch.no_grad():
        for group in ids:
            take=d['groups']==group
            totals.append(float((head(d['x'][take])-d['y'][take]).square().mean()))
    return float(np.mean(totals))

def train(p,d,width,lr,seed,split,who,trial,guard):
    torch.set_num_threads(p['fit_threads']);head=A.make_head(width,seed)
    opt=torch.optim.Adam(head.parameters(),lr=lr,eps=1e-8)
    lengths=torch.tensor([r['length'] for r in d['records']],dtype=torch.int64)
    starts=torch.cat([torch.zeros(1,dtype=torch.int64),lengths.cumsum(0)[:-1]])
    old=list(range(403));new=list(range(403,410))
    old_val=[i for i in old if i%5==0];new_val=[max(new,key=lambda i:d['records'][i]['id'])]
    if split=='development':
        old=[i for i in old if i not in old_val];new=[i for i in new if i not in new_val]
    generator=torch.Generator().manual_seed(seed+991)
    old,new=torch.tensor(old),torch.tensor(new)
    def indices(ids):
        group=ids[torch.randint(len(ids),(p['replay_batch_per_group'],),generator=generator)]
        return starts[group]+(torch.rand(len(group),generator=generator,dtype=torch.float64)*lengths[group]).long()
    for step in range(p['replay_steps']):
        guard();oi,ni=indices(old),indices(new)
        prediction=head(torch.cat([d['x'][oi],d['x'][ni]]));target=torch.cat([d['y'][oi],d['y'][ni]])
        loss=(prediction-target).square().mean();assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        assert all(torch.isfinite(v.grad).all() for v in head.parameters())
        events(who,trial,'BEFORE',step);opt.step();events(who,trial,'AFTER',step)
        if step%64==0:print(json.dumps(dict(trial=trial,who=who,step=step,loss=float(loss))),flush=True)
    guard()
    metrics=dict(old_validation_mse=group_loss(head,d,old_val),new_validation_mse=group_loss(head,d,new_val))
    metrics['selection_score']=.5*(metrics['old_validation_mse']+metrics['new_validation_mse'])
    state=dict(head=copy.deepcopy(head.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),generator=generator.get_state())
    return head,state,metrics

def pair(p,d,width,lr,seed,split,trial,guard):
    head,state,metrics=train(p,d,width,lr,seed,split,'live',trial,guard)
    other,shadow,check=train(p,d,width,lr,seed,split,'shadow',trial,guard)
    assert digest(state)==digest(shadow) and metrics==check
    with (OUT/'fit'/(trial+'.pt.gz')).open('xb') as f:
        b=io.BytesIO();torch.save(state,b);f.write(gzip.compress(b.getvalue(),mtime=0))
    return head,dict(width=width,lr=lr,seed=seed,split=split,trial=trial,state_hash=digest(state),metrics=metrics,
        state_path=str(OUT/'fit'/(trial+'.pt.gz')),state_sha256=sha(OUT/'fit'/(trial+'.pt.gz')))

def guard_for(p,seconds,learning=True):
    until=min(time.time()+seconds,p['absolute_learning_stop'] if learning else p['absolute_all_stop'])-15
    last=[0.]
    def guard():
        assert time.time()<until,'Original fit/audit action deadline'
        if time.time()-last[0]>10:
            immutable(p);last[0]=time.time()
    return guard

def main():
    p=read(OUT/'protocol.json');immutable(p,True);os.sched_setaffinity(0,set(range(8,16)))
    torch.use_deterministic_algorithms(True);d=data();guard=guard_for(p,p['fit_main_seconds']);trials=[];finals={}
    for width in p['replay_widths']:
        runs=[]
        for i,lr in enumerate(p['replay_lrs']):
            _,r=pair(p,d,width,lr,p['replay_seed'],'development',f'dev_{width}_{i}',guard);trials.append(r);runs.append(r)
        chosen=min(runs,key=lambda r:(r['metrics']['selection_score'],p['replay_lrs'].index(r['lr'])))
        head,r=pair(p,d,width,chosen['lr'],p['replay_seed'],'all',f'final_{width}',guard);trials.append(r)
        q=A.attach(p,head);r['model']=A.export(q,p,OUT/'fit'/f'replay_{width}.pt.gz');finals[f'replay_{width}']=r
        save(OUT/'fit/PROGRESS.json',dict(completed_width=width,trials=trials,finals=finals))
    chosen=finals['replay_1736']
    head,r=pair(p,d,1736,chosen['lr'],p['second_seed'],'all','final_1736_seed2',guard);trials.append(r)
    q=A.attach(p,head);r['model']=A.export(q,p,OUT/'fit/replay_1736_seed2.pt.gz');finals['replay_1736_seed2']=r
    # Isolated radius ablation: preserve centers/weights/order and old arithmetic exactly.
    torch.set_num_threads(1);plain=A.Existing.frozen(p,p['existing_A']['path']);c=plain.new_centers
    other=torch.full((len(c),),float('inf'),dtype=torch.float64)
    for start in range(0,len(c),32):
        guard();v=((c[start:start+32,None]-c)**2).sum(-1)
        v[torch.arange(len(v)),torch.arange(start,start+len(v))]=float('inf');other[start:start+len(v)]=v.min(-1).values.sqrt()
    baseline=copy.deepcopy(plain.state_dict());events('live','plain','BEFORE',0,'closed_form')
    plain.new_radii.copy_(.49*other);events('live','plain','AFTER',0,'closed_form')
    shadow=A.Existing.frozen(p,p['existing_A']['path']);events('shadow','plain','BEFORE',0,'closed_form')
    shadow.new_radii.copy_(.49*other);events('shadow','plain','AFTER',0,'closed_form')
    assert digest(plain.state_dict())==digest(shadow.state_dict())
    assert all(torch.equal(v,plain.state_dict()[k]) for k,v in baseline.items() if k!='new_radii')
    model=A.export(plain,p,OUT/'fit/plain.pt.gz',A.Existing.ARCH)
    finals['plain']=dict(model=model,state_hash=digest(plain.state_dict()),radii_hash=digest(plain.new_radii))
    save(OUT/'fit/summary.json',dict(state='COMPLETE',pid=os.getpid(),trials=trials,finals=finals,
        actual_live_Adam=3840,actual_shadow_Adam=3840,actual_live_closed_form=1,actual_shadow_closed_form=1,
        new_RL=0,new_Value=0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    immutable(p,True)

def audit():
    p=read(OUT/'protocol.json');immutable(p,True);os.sched_setaffinity(0,set(range(8,16)))
    torch.use_deterministic_algorithms(True);d=data();s=read(OUT/'fit/summary.json');guard=guard_for(p,p['fit_audit_seconds'],False)
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    for name,r in s['finals'].items():
        if name=='plain':continue
        head,state,metrics=train(p,d,r['width'],r['lr'],r['seed'],'all','independent',r['trial'],guard)
        assert digest(state)==r['state_hash'] and metrics==r['metrics']
        loaded=A.frozen(p,r['model']['path']);q=A.attach(p,head)
        assert sha(r['model']['path'])==r['model']['sha256'] and digest(loaded.state_dict())==digest(q.state_dict())
    torch.set_num_threads(1);q=A.Existing.frozen(p,p['existing_A']['path']);c=q.new_centers
    radii=[]
    for start in range(0,len(c),32):
        guard();v=((c[start:start+32,None]-c)**2).sum(-1)
        v[torch.arange(len(v)),torch.arange(start,start+len(v))]=float('inf');radii.append(.49*v.min(-1).values.sqrt())
    events('independent','plain','BEFORE',0,'closed_form');q.new_radii.copy_(torch.cat(radii));events('independent','plain','AFTER',0,'closed_form')
    assert digest(q.state_dict())==s['finals']['plain']['state_hash']
    save(OUT/'fit/audit.json',dict(state='PASS',independent_Adam=1536,independent_closed_form=1,
        all_final_heads_and_optimizer_states_reconstructed_exact=True,no_native=True))
    immutable(p,True)

if __name__=='__main__':
    {'main':main,'audit':audit}[sys.argv[1]]()

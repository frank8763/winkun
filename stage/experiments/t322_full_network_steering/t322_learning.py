"""Finite whole-network imperfect steering demonstration initialization.

No successful expert labels, RL updates, advantage estimates or policy-density
claims. Canonical batch1 is used for reporting/native deployment; chunked
supervised gradients are an explicit deterministic training computation.
"""
from pathlib import Path
import sys,os,json,io,gzip,copy,time,shutil,traceback,signal
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t319_imperfect_demo_head'))
import t319_core as H
np,torch,K,U=H.np,H.torch,H.K,H.U
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t322_full_network_steering';DOC=ROOT/'stage/docs/distance_20261103'
read=H.read;HOLDER={}
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
def append(path,e):
    with path.open('a') as f:f.write(json.dumps(e,allow_nan=False,separators=(',',':'))+'\n');f.flush()
def optimizer_count():
    f=OUT/'optimizer_calls.jsonl'
    return sum(json.loads(v)['event']=='completed' for v in f.read_text().splitlines()) if f.exists() else 0
def holder_hashes():
    return {} if not HOLDER else dict(policy=K.S.digest(HOLDER['q'].state_dict()),optimizer=K.S.digest(HOLDER['opt'].state_dict()))
def mode_raw(q,x,canonical=True):
    chunks=x.split(1 if canonical else 64)
    outputs=[q.step(chunk) for chunk in chunks]
    return torch.cat([v[0] for v in outputs]),torch.cat([v[3] for v in outputs])
def dataset(p,parent):
    x,y=H.demo(p['train_demo']);vx,vy=H.demo(p['validation_demo']);ox,indices=H.anchors(p)
    changed=y[:,1]!=x[:,-2];assert changed.any() and (~changed).any()
    weights=np.where(changed,.5/changed.sum(),.5/(~changed).sum()).astype(np.float32)
    assert np.isclose(weights.sum(),1.)
    x,y,ox=map(torch.tensor,[x,y,ox])
    with torch.no_grad():dm,dr=mode_raw(parent,x);om,_=mode_raw(parent,ox)
    return dict(x=x,y=y,weights=torch.tensor(weights),old_x=ox,old_y=om,drive=dr[:,0],changed=changed,
                vx=torch.tensor(vx),vy=torch.tensor(vy),anchor_indices=indices)
def backward(p,q,data):
    # Stream the exact registered weighted full-dataset objective. Fixed chunks
    # use all rows; no sampled minibatch or hidden sequence carry.
    q.zero_grad(set_to_none=True);parts=[0.,0.,0.]
    for start in range(0,len(data['x']),64):
        end=start+64;_,_,_,raw=q.step(data['x'][start:end])
        steer=((raw[:,1]-data['y'][start:end,1]).square()*data['weights'][start:end]).sum()
        drive=(raw[:,0]-data['drive'][start:end]).square().sum()/len(data['x'])*p['demo_drive_retention']
        (steer+drive).backward();parts[0]+=float(steer.detach());parts[1]+=float(drive.detach())
    for start in range(0,len(data['old_x']),64):
        pred=q.step(data['old_x'][start:start+64])[0]
        loss=(pred-data['old_y'][start:start+64]).square().sum()/(2*len(data['old_x']))*p['retention_weight']
        loss.backward();parts[2]+=float(loss.detach())
    assert all(np.isfinite(parts));return parts
@torch.no_grad()
def metrics(q,data):
    out={}
    for name,x,y in [('train',data['x'],data['y']),('validation',data['vx'],data['vy'])]:
        pred,raw=mode_raw(q,x);changed=(y[:,1]!=x[:,-2]).numpy();m={}
        for group,mask in [('ALL',np.ones(len(x),bool)),('STEER_CHANGE',changed),('STEER_HOLD',~changed)]:
            m[group]=dict(rows=int(mask.sum()),steer_mode_MSE=float((pred[mask,1]-y[mask,1]).square().mean()),
                steer_raw_MSE=float((raw[mask,1]-y[mask,1]).square().mean()),
                copy_previous_steer_MSE=float((x[mask,-2]-y[mask,1]).square().mean()),
                human_drive_MSE_description_only=float((pred[mask,0]-y[mask,0]).square().mean()))
        out[name]=m
    pred,_=mode_raw(q,data['old_x']);out['old']=dict(rows=len(pred),mode_MSE=float((pred-data['old_y']).square().mean()),max_mode_change=float((pred-data['old_y']).abs().max()))
    return out
def fit(p,guard,event,steps=None):
    q=K.policy(p).eval();opt=K.T.L.optimizer(q,p);HOLDER.update(q=q,opt=opt)
    parent=copy.deepcopy(q).requires_grad_(False);data=dataset(p,parent)
    assert not q.memory_on and not q.base.logstd.requires_grad and K.S.digest(q.state_dict())==p['policy_hash']
    before=metrics(q,data);initial={k:K.S.digest(v) for k,v in q.state_dict().items()};records=[]
    for step in range(p['initialization_steps'] if steps is None else steps):
        guard();losses=backward(p,q,data)
        groups={name:float(v.grad.norm()) if v.grad is not None else None for name,v in q.named_parameters() if name in ['base.body.net.0.proprio.weight','encoder.0.weight','core.weight_ih_l0','core.weight_hh_l0','residual.weight']}
        norm=float(torch.nn.utils.clip_grad_norm_([v for v in q.parameters() if v.requires_grad],1.,error_if_nonfinite=True))
        event(dict(event='started',step=step));opt.step();opt.zero_grad(set_to_none=True)
        row=dict(event='completed',step=step,steering_loss=losses[0],drive_retention=losses[1],old_retention=losses[2],
            preclip_gradient_norm=norm,gradient_groups=groups,policy_hash=K.S.digest(q.state_dict()),optimizer_hash=K.S.digest(opt.state_dict()))
        records.append(row);event(row)
    guard();after=metrics(q,data);changed=[k for k,v in q.state_dict().items() if initial[k]!=K.S.digest(v)]
    return q,opt,dict(initial_policy_hash=p['policy_hash'],policy_hash=K.S.digest(q.state_dict()),optimizer_hash=K.S.digest(opt.state_dict()),
        Adam_calls=len(records),RL_updates=0,train_rows=len(data['x']),validation_rows=len(data['vx']),steer_change_rows=int(data['changed'].sum()),
        train_weights_sha256=K.S.digest(data['weights']),anchor_indices=data['anchor_indices'],changed_state_keys=changed,
        trainable_parameters=sum(v.numel() for v in q.parameters() if v.requires_grad),before=before,after=after,
        training_batch_size=64,native_reporting_batch_size=1,records=records,successful_demonstrations=0,
        validation_same_scene_not_unseen_navigation=True,all_network_parameters_enabled_except_fixed_logstd=True,
        zero_hidden_recurrent_weight_can_have_zero_gradient=True)
def checkpoint(q,opt,p):
    b=io.BytesIO();torch.save(dict(policy=q.state_dict(),optimizer=opt.state_dict(),policy_hash=K.S.digest(q.state_dict()),
        optimizer_hash=K.S.digest(opt.state_dict()),human_successes=0,RL_updates=0),b)
    z=gzip.compress(b.getvalue(),compresslevel=6,mtime=0);assert len(z)<=p['model_bytes'];return z
def worker(pipe,index,p,physical):
    sys.path.insert(0,str(ROOT/'stage/experiments/t312_online_goal_ppo'));import ppo_worker as original
    original.OUT=OUT;original.B.OUT=OUT;torch.optim.Adam.step=U.forbid_step
    return original.worker(pipe,index,p,physical)
outcome=U.outcome
def counts(rows):
    return {role:dict(executed=sum(r['role']==role for r in rows),outcomes={kind:sum(r['role']==role and outcome(r)==kind for r in rows) for kind in ['GOAL','CONTACT','TIMEOUT','INVALID','CENSORED']}) for role in ['old_guard','six_development']}

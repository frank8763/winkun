"""Frozen single-Actor output-semantic contrast; no teacher or RNG at runtime."""
from pathlib import Path
import os
import sys
import types
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t631_explored_recovery'))
import t631_native as Old
from expectation import clipped_mean
N=Old.N;M=Old.P.M
np,torch=N.np,N.torch


def output_hash(base,mode,sigma):
    return base if mode=='mode' else M.digest(dict(neural_policy_hash=base,
              deterministic_output='E_clip_Gaussian_coordinatewise_v1',sigma=list(sigma)))


def adapt(q,mode):
    assert mode in ['mode','expected']
    if mode=='mode':return q
    assert not getattr(q,'_t634_adapted',False)
    original=q.step;base=q.phash();sigma=list(q.p['raw_sigma'])
    def step(this,x,hidden=None):
        assert not this.sample
        _,loc,h,value=original(x,hidden)
        action=clipped_mean(loc.detach().cpu().numpy(),sigma)
        row=this.tape[-1]
        row['pre_transform_locations']=row['locations'].copy()
        row['raw_actions']=action[0].copy();row['locations']=action[0].copy()
        row['atom']=np.where(action[0]>=1,1,np.where(action[0]<=-1,-1,0)).astype(np.int8)
        command=x.new_tensor(action).float()
        return command,x.new_tensor(action),h,value
    q.step=types.MethodType(step,q)
    q.phash=types.MethodType(lambda this:output_hash(base,'expected',sigma),q)
    q._t634_adapted=True
    return q


class Session(Old.Session):
    def __init__(self,job,*args,**kwargs):
        super().__init__(job,*args,**kwargs)
        self.mode=job['execution_mean_mode'];assert not self.explore
        adapt(self.q,self.mode)


def payload(s,tape=None):
    a=Old.P.payload(s,tape)
    a['execution_mean_mode']=np.array(int(s.mode=='expected'),np.uint8)
    if s.mode=='expected':
        t=s.q.tape if tape is None else tape
        a['pre_transform_locations']=np.asarray([v['pre_transform_locations'] for v in t])
    return a


def inspect(p,values):
    for a,r in values:
        mode=r['request']['job'].get('execution_mean_mode','mode')
        if mode=='mode':
            assert int(a.get('execution_mean_mode',np.array(0)))==0
            clean={k:v for k,v in a.items() if k!='execution_mean_mode'}
            Old.inspect(p,[(clean,r)]);continue
        assert mode=='expected' and int(a['execution_mean_mode'])==1
        assert r['seed'] is None and not a['stochastic'].any()
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        assert not a['innovations'].any()
        q=adapt(M.frozen(p,r['request']['model']),'expected');assert q.phash()==r['policy_hash']
        q.sample=False;q.reset_noise();h=None;job=r['request']['job']
        first=int(a.get('student_takeover_first',np.array(-1)))
        if first>=0:
            source=N.E.N.S.arrays(job['source_path']);assert first==job['student_takeover_first']
            for key in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times']:
                assert np.array_equal(a[key][:first],source[key][:first]),key
            for key in ['states','poses','history_poses','history_goals','history_times']:
                assert np.array_equal(a[key][first],source[key][first]),key
            assert np.array_equal(a['initial_native_integration'],source['initial_native_integration'])
            assert a['ownership'][:first].all() and not a['ownership'][first:].any()
            assert np.array_equal(a['source_prefix_actions'][:first],source['actions'][:first])
            assert not a['source_prefix_actions'][first:].any()
        else:assert not a['ownership'].any()
        with torch.no_grad():
            for k,x in enumerate(a['states']):
                hb=np.zeros(64) if h is None else h[0,0].numpy().copy()
                command,loc,h,_=q.step(torch.tensor(x[None]),h)
                assert np.array_equal(hb,a['hidden_before'][k])
                for key,v in q.tape[-1].items():assert np.array_equal(v,a[key][k]),(k,key)
                if first<0 or k>=first:assert np.array_equal(command[0].numpy(),a['actions'][k])
                assert np.max(np.abs(N.E.N.K.H.rebuild(x,a['poses'][k],a['history_poses'][k],a['task_goal_xy'],a['history_goals'][k])-x))<=2e-6
                count=int(a['native_counts'][k]);z=p['gamma_native']**count;flag=a['end_flags'][k]
                d0=np.linalg.norm(a['poses'][k,:2]-a['task_goal_xy']);d1=np.linalg.norm(a['next_poses'][k,:2]-a['task_goal_xy'])
                reward=-.000002*(1-z)/(1-p['gamma_native'])+.03*d0-z*.03*d1*(not flag.any())
                if flag.any():reward+=p['gamma_native']**(count-1)*(float(flag[1])-float(flag[0])-.5*float(flag[2]))
                assert abs(reward-a['rewards'][k])<1e-12
        assert np.array_equal(h[0,0].numpy(),a['last_hidden_after'])
        assert np.array_equal(q.extra_hidden[0,0].numpy(),a['last_extra_hidden_after'])
        assert np.array_equal(q.anchor_hidden[0,0].numpy(),a['last_anchor_hidden_after'])


def worker(pipe,index,p,counts):
    os.environ['CUDA_VISIBLE_DEVICES']='';assert not torch.cuda.is_initialized()
    Old.P.install();real=N.G._real_affinity
    def pin(pid,mask):
        return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;N.G.T.Session=Session
    N.G.B.native_setup=lambda folder,plan:Old.Old.reset_contract.field_setup(N.F,folder,plan)
    N.E.payload=payload;N.G.T.worker(pipe,index,p,counts)

"""Bounded BC-only policy recovery from three verified Stage success trajectories.

This is an actor-only research candidate, not a resumable TD3 checkpoint.
"""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'training_plan.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    lock = (ROOT / 'training.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (ROOT / 'candidate_actor.pt').exists(), 'REFUSE_OVERWRITE_CANDIDATE'
    plan = json.loads(PLAN.read_text())
    assert sha(__file__) == plan['training_source_sha256']
    for name in ['parent_checkpoint', 'success_archive', 'core_source']:
        assert sha(plan[name]['path']) == plan[name]['sha256'], name
    torch.set_num_threads(1)
    torch.manual_seed(plan['seed'])
    rng = np.random.default_rng(plan['seed'])
    spec = importlib.util.spec_from_file_location('t151_core',plan['core_source']['path'])
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    parent = torch.load(plan['parent_checkpoint']['path'],map_location='cpu',weights_only=False)
    actor = core.Actor(112,2)
    actor.load_state_dict(parent['actor'],strict=True)
    ignored_columns = [57,58,74,75,86,87,98,99,110,111]
    with torch.no_grad():
        actor.net[0].weight[:, ignored_columns] = 0.0
    with np.load(plan['success_archive']['path'],allow_pickle=False) as z:
        states = np.asarray(z['states'],dtype=np.float32).copy()
        actions = np.asarray(z['raw_actions'],dtype=np.float32).copy()
        offsets = z['episode_offsets'].tolist()
    assert offsets == [0,10612,20319,32164]
    assert states.shape == (32164,112) and actions.shape == (32164,2)
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    state = torch.from_numpy(states)
    target = torch.from_numpy(actions)
    optimizer = torch.optim.Adam(actor.parameters(),lr=plan['learning_rate'])

    def metrics():
        actor.eval()
        errors=[]
        with torch.no_grad():
            for a in range(0,len(state),512):
                errors.append((actor(state[a:a+512])-target[a:a+512]).square())
        e=torch.cat(errors)
        return {'left_mse':float(e[:offsets[1]].mean()),
                'right_mse':float(e[offsets[1]:].mean()),
                'throttle_rmse':float(e[:,0].mean().sqrt()),
                'steer_rmse':float(e[:,1].mean().sqrt())}

    initial=metrics()
    start=time.monotonic()
    losses=[]
    actor.train()
    for update in range(1,plan['updates']+1):
        ids=np.concatenate([rng.integers(0,offsets[1],128),rng.integers(offsets[1],offsets[2],64),rng.integers(offsets[2],offsets[3],64)])
        prediction=actor(state[ids])
        loss=(prediction-target[ids]).square().mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        actor.net[0].weight.grad[:, ignored_columns] = 0.0
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0)
        optimizer.step()
        if update%100==0:
            row={'update':update,'mse':float(loss.detach()),'elapsed_s':time.monotonic()-start}
            losses.append(row)
            print('T156_BC_UPDATE',json.dumps(row),flush=True)
    final=metrics()
    candidate={'algorithm':'STAGE_BC_ONLY_RECOVERY','actor':actor.state_dict(),
        'inference_only':True,'td3_resume_compatible':False,
        'parent_sha256':plan['parent_checkpoint']['sha256'],
        'archive_sha256':plan['success_archive']['sha256'],
        'training_plan_sha256':sha(PLAN),'bc_optimizer':optimizer.state_dict(),
        'bc_updates':plan['updates'],'seed':plan['seed'],
        'ignored_action_history_columns':ignored_columns}
    tmp=ROOT/'candidate_actor.pt.partial'
    torch.save(candidate,tmp)
    tmp.replace(ROOT/'candidate_actor.pt')
    report={'experiment':plan['experiment'],'result':'TRAINING_COMPLETED_NOT_YET_EVALUATED',
        'initial_archive_fit':initial,'final_archive_fit':final,
        'training_losses':losses,'checkpoint_sha256':sha(ROOT/'candidate_actor.pt'),
        'elapsed_seconds':time.monotonic()-start,'environment_frames':0,
        'learning_method':'behavior cloning only','automatic_promotion':False,
        'generalization_claim':False,'note':'Archive-fit error is not a navigation success metric.'}
    (ROOT/'training_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('T156_BC_COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':
    main()

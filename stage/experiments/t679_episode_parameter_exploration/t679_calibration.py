"""One output-RMS calibration rule for full adapter Gaussian perturbations."""
from pathlib import Path
import copy,gzip,io,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t673_all_task_signal_gate'))
import t673_analysis as B
M,T,torch,np=B.M,B.T,B.torch,B.np
read,sha,digest=B.read,B.sha,B.digest

def parameter_layout(q):
    parts=[]
    for label,module in [('drive',q.drive_adapter),('steer',q.steer_adapter)]:
        for name,v in module.named_parameters():parts.append((label,name,v.shape,v.numel()))
    theta=torch.cat([v.detach().flatten() for v in [*q.drive_adapter.parameters(),*q.steer_adapter.parameters()]])
    assert len(theta)==11906;return theta,parts

def output(q,g,theta,layout):
    dictionaries={'drive':{},'steer':{}};k=0
    for label,name,shape,n in layout:dictionaries[label][name]=theta[k:k+n].reshape(shape);k+=n
    assert k==len(theta)
    drive=torch.func.functional_call(q.drive_adapter,dictionaries['drive'],(g['f'],))
    steer=torch.func.functional_call(q.steer_adapter,dictionaries['steer'],(g['f'],))
    return 1.25*torch.tanh(g['logits']+.3*torch.tanh(torch.cat([drive,steer],-1)))

def set_parameters(q,theta):
    k=0
    with torch.no_grad():
        for v in [*q.drive_adapter.parameters(),*q.steer_adapter.parameters()]:v.copy_(theta[k:k+v.numel()].reshape_as(v));k+=v.numel()
    assert k==len(theta)

def direction(seed):return np.random.default_rng(seed).standard_normal(11906)

def calibration(p,q,guard):
    theta,layout=parameter_layout(q);groups=[]
    for row in p['calibration_rows']:
        guard();assert sha(row['path'])==row['sha256'] and row['policy_hash']==q.phash()
        a=T.R.E.N.S.arrays(row['path']);assert not a['ownership'].any()
        g=M.features(q,a,row['job_id'],row['sha256']);assert torch.allclose(output(q,g,theta,layout),q.sequence(torch.tensor(a['states'])),atol=1e-10,rtol=0)
        groups.append(g)
    directional_energy=[];numeric=[]
    for index,seed in enumerate(p['calibration_direction_seeds']):
        guard();epsilon=theta.new_tensor(direction(seed));energies=[]
        for j,g in enumerate(groups):
            _,jvp=torch.autograd.functional.jvp(lambda th:output(q,g,th,layout),theta,epsilon,create_graph=False,strict=True)
            assert bool(torch.isfinite(jvp).all());energies.append(float(jvp.square().mean()))
            if index==0 and j==0:
                h=p['finite_parameter_epsilon'];finite=(output(q,g,theta+h*epsilon,layout)-output(q,g,theta-h*epsilon,layout))/(2*h)
                error=float((jvp-finite).abs().max());tolerance=p['numeric_atol']+p['numeric_rtol']*float(jvp.abs().max())
                numeric.append(dict(error=error,tolerance=tolerance,passed=error<=tolerance));assert numeric[-1]['passed']
        directional_energy.append(sum(energies)/len(energies))
    sensitivity=float(np.sqrt(np.mean(directional_energy)));assert sensitivity>1e-12
    scale=p['target_location_RMS']/sensitivity
    evaluation=[]
    for seed in p['evaluation_direction_seeds']:
        epsilon=theta.new_tensor(direction(seed))
        for sign in [-1,1]:
            errors=[]
            for g in groups:
                guard();delta=output(q,g,theta+sign*scale*epsilon,layout)-output(q,g,theta,layout)
                assert bool(torch.isfinite(delta).all());errors.append(float(delta.square().mean()))
            rms=float(np.sqrt(np.mean(errors)));evaluation.append(dict(seed=seed,sign=sign,actual_location_RMS=rms,admitted=p['minimum_location_RMS']<=rms<=p['maximum_location_RMS']))
    return dict(state='PASS' if all(x['admitted'] for x in evaluation) else 'CALIBRATION_GATE_FAILED',scale=scale,target=p['target_location_RMS'],
                parameter_dimension=11906,sensitivity=sensitivity,energies=directional_energy,numeric=numeric,evaluation=evaluation,
                calibration_is_network_parameter_response_not_physics=True,location_RMS_not_matched_closed_loop_or_clipped_action_variance=True,
                parameter_Gaussian_not_per_vector_normalized=True),theta

def export_variant(p,q,folder,name):
    payload=copy.deepcopy(M.P.checkpoint(p['initial']['path']));payload['joint_state']['actor']=copy.deepcopy(q.state_dict());payload['policy_hash']=q.phash()
    payload['joint_hash']=digest(payload['joint_state']);payload['frozen_parameter_exploration_only']=True
    buf=io.BytesIO();torch.save(payload,buf);blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<p['model_bytes'];path=folder/(name+'.pt.gz')
    with path.open('xb') as f:f.write(blob)
    loaded=M.L.frozen(p,path);assert loaded.phash()==q.phash() and digest(loaded.state_dict())==digest(q.state_dict())
    return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),joint_hash=payload['joint_hash'],bytes=len(blob))

def offset_stats(q,row):
    a=T.R.E.N.S.arrays(row['path'])
    with torch.no_grad():base=q.sequence(torch.tensor(a['states'])).numpy()
    raw=np.asarray(a['locations'])-base;command=np.clip(a['locations'],-1,1)-np.clip(base,-1,1)
    correlation=[]
    for j in range(2):
        left=command[:-1,j];right=command[1:,j]
        correlation.append(float(np.corrcoef(left,right)[0,1]) if len(left)>1 and left.std()>1e-12 and right.std()>1e-12 else None)
    return dict(source_sha256=row['sha256'],raw_location_offset_RMS=float(np.sqrt(np.square(raw).mean())),
                command_offset_RMS=float(np.sqrt(np.square(command).mean())),command_offset_lag1=correlation,
                same_external_history_not_new_closed_loop_center=True)

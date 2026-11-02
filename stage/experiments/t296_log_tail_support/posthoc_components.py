"""Posthoc density-channel decomposition; no new acceptance/training experiment."""
from run import ROOT,OUT,D,load,policy,np,torch,Path,json

def main():
    p=json.loads((OUT/'protocol.json').read_text());torch.set_num_threads(1);q=policy(json.loads(Path(p['source_protocol']).read_text())).eval().requires_grad_(False)
    def components(m,a):
        z,mar=D.terms(m,a);previous=torch.cat([z[:1],z[:-1]]);rho=torch.full((len(z),1),D.RHO,dtype=torch.float64);rho[0]=0;v=1-rho*rho
        return (-.5*(z-rho*previous).square()/v-.5*v.log()+.5*z.square()+mar).numpy()
    rows=[]
    for s in p['sources']:
        old=load(s['path']);new=load(s['witness_payload']);states=new['alternate_states'];actions=torch.tensor(new['alternate_actions']);n=len(states)
        with torch.no_grad():means=torch.cat([q.step(torch.tensor(x[None]))[1] for x in states])
        oldmeans=torch.tensor(old['means'][:n]);delta=components(means,actions)-components(oldmeans,actions)
        rows.append(dict(id=s['id'],prefix_log_weight_by_drive_steer=delta.sum(0).tolist(),mean_abs_native_mean_shift_by_drive_steer=np.abs(means.numpy()-oldmeans.numpy()).mean(0).tolist(),max_abs_native_mean_shift_by_drive_steer=np.abs(means.numpy()-oldmeans.numpy()).max(0).tolist()))
    expected=json.loads((ROOT/'stage/docs/distance_20261102/T296_POSTHOC_COMPONENTS.json').read_text());assert rows==expected['rows'];print('Posthoc drive/steer decomposition exact; no file modification/native/optimizer')

if __name__=='__main__':main()

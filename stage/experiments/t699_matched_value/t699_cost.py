"""Separate CPU batch1 cost and actual executed-command diagnostics."""
import gzip,io,os,resource,sys,time
from t699_common import *
import t699_actor as A
torch,np=A.torch,A.np

def run(name):
    p=read(OUT/'protocol.json');immutable(p,True);os.sched_setaffinity(0,{15})
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    from t699_native import model
    m=model(p,name);rss_before=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    q=A.frozen(p,m['path']);assert q.phash()==m['policy_hash']
    rss_loaded=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    peak_loaded=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    state=q.state_dict();parent=A.Parent.frozen(p,p['initial']['path'])
    total_bytes=sum(t.numel()*t.element_size() for t in state.values())
    parent_bytes=sum(t.numel()*t.element_size() for t in parent.state_dict().values())
    row=p['protected403_rows'][0];a=A.Existing.L.T.R.E.N.S.arrays(row['path']);x=torch.tensor(a['states'][:64],dtype=torch.float64)
    q.sample=False;q.rng=None;q.reset_noise();h=None;lat=[]
    with torch.no_grad():
        for i in range(320):
            t=time.perf_counter_ns();_,_,h,_=q.step(x[i%len(x):i%len(x)+1],h)
            if i>=64:lat.append((time.perf_counter_ns()-t)/1e6)
    result=dict(model=m,name=name,raw_state_tensor_bytes=total_bytes,common_parent_bytes=parent_bytes,
        added_tensor_bytes=total_bytes-parent_bytes,gzip_model_bytes=m['bytes'],device='CPU15',cpu_threads=1,
        batch1_median_ms=float(np.median(lat)),batch1_p95_ms=float(np.percentile(lat,95)),warmup_decisions=64,timed_decisions=256,
        process_RSS_before_model=rss_before,process_RSS_loaded=rss_loaded,process_peak_RSS_after_model=peak_loaded,
        RSS_includes_Python_Torch_and_common_dependencies=True,compressed_checkpoint_not_deployment_RAM=True)
    if name not in ['historical_parent','historical_A']:
        d=torch.load(io.BytesIO(gzip.decompress((OUT/'DATA.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
        bad_old=set();bad_new=set();count_old=count_new=0;maxerr=0.;sumerr_old=0.;sumerr_new=0.;step=0;last=time.time()
        with torch.no_grad():
            for index,(f,mu,target,group) in enumerate(zip(d['x'],d['mu'],d['commands'],d['groups'])):
                if time.time()-last>10:
                    immutable(p);last=time.time();save(OUT/'cost_progress.json',dict(name=name,completed=index,total=len(d['x'])))
                # Exact CPU batch1 head arithmetic; old parent is frozen and mu is independently audited own-carry parent location.
                if name=='plain':
                    phi=A.Parent.basis(f[None],q.new_centers,q.new_radii)
                    value=mu[None]+phi@q.new_weights if bool((phi!=0).any()) else mu[None]
                else:value=mu[None]+q.replay_head(f[None])
                # Verify this factored path against the actual complete deployed location.
                if index%4096==0:assert torch.equal(value,q.location(f[:64][None],f[64:][None]))
                cmd=value.clamp(-1,1).float()[0];expected=target.float();changed=not A.Existing.same_float32(cmd.numpy(),expected.numpy())
                err=float((cmd.double()-target).square().mean());maxerr=max(maxerr,float((cmd.double()-target).abs().max()))
                group=int(group)
                if group<403:
                    sumerr_old+=err
                    if changed:bad_old.add(d['records'][group]['id']);count_old+=1
                else:
                    sumerr_new+=err
                    if changed:bad_new.add(d['records'][group]['id']);count_new+=1
        result['historical_command_drift']=dict(old_ids_changed=sorted(bad_old),new_ids_changed=sorted(bad_new),
            old_decisions_changed=count_old,new_decisions_changed=count_new,old_total=81956,new_total=1737,
            old_mean_squared_command_error=sumerr_old/81956,new_mean_squared_command_error=sumerr_new/1737,max_absolute_command_error=maxerr,
            support_overlap_is_not_assumed_to_change_commands=True,own_frozen_carry_verified_at_data_preflight=True,
            CPU_batch1_all_decisions=True,complete_deployed_location_crosscheck_stride=4096)
    save(OUT/'cost'/f'{name}.json',result);immutable(p,True)
    print(json.dumps({k:v for k,v in result.items() if k!='historical_command_drift'}),flush=True)

if __name__=='__main__':run(sys.argv[1])

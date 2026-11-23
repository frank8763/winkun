"""One original numerical repro and one fixed equivalent representation probe."""
from pathlib import Path
import importlib.util,gzip,io,json,sys,time,os
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/day_20261123/t707';OUT=ROOT/'stage/runs/t707_numerical_replay'
spec=importlib.util.spec_from_file_location('t707_original_numerical_seam',ROOT/'stage/experiments/t706_taskwise_replay/fit.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
np,torch=F.np,F.torch;A=F.A;C=F.C;save=C.save

def main(mode):
    bound=C.read(DOC/'DIAGNOSIS_BOUND.json');C.stopped();torch.set_num_threads(1);os.sched_setaffinity(0,{15})
    def guard():assert time.time()<bound['absolute_stop']-5;C.C.resources()
    if mode=='red':
        p=C.read(ROOT/'stage/runs/t706_taskwise_replay/protocol.json');data=torch.load(io.BytesIO(gzip.decompress((ROOT/'stage/runs/t699_matched_value/DATA.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
        q=A.frozen(p,p['repair_initial']['path']);head=q.replay_head
        with torch.no_grad():
            h=head[1](head[0](data['x']));Z=torch.cat([h,torch.ones(len(h),1,dtype=torch.float64)],1).numpy();W0=torch.cat([head[2].weight.T,head[2].bias[None]],0).numpy().copy()
        stat=F.stats(Z,data['y'].numpy(),data['groups'].numpy(),data['records'],guard)
        np.savez_compressed(OUT/'diagnosis_gram.npz',G=np.stack([s['G'] for s in stat]),B=np.stack([s['B'] for s in stat]),constant=np.array([s['constant'] for s in stat]),W0=W0)
        (OUT/'diagnosis_ids.json').write_text(json.dumps([s['id'] for s in stat]))
    else:
        cache=dict(np.load(OUT/'diagnosis_gram.npz'));W0=cache['W0'];ids=json.loads((OUT/'diagnosis_ids.json').read_text());stat=[dict(id=name,G=cache['G'][i],B=cache['B'][i],constant=float(cache['constant'][i])) for i,name in enumerate(ids)]
    original=F.clarabel.DefaultSolver;events=[];curves=[];modeldata={};solutions=[]
    class Wrapped:
        def __init__(self,P,q,mat,b,cones,settings):
            self.original_size=len(q);self.cut=False
            if mode=='green':
                assert q[-1]==0 and P[:,-1].nnz==0 and mat[:,-1].nnz==0
                P=P[:-1,:-1].tocsc();q=q[:-1];mat=mat[:,:-1].tocsc();self.cut=True
            settings.max_iter=64;settings.time_limit=60
            modeldata.update(variables=len(q),rows=len(b),nonzeros=mat.nnz,free_last_column=(not self.cut))
            self.solver=original(P,q,mat,b,cones,settings)
        def set_termination_callback(self,callback):
            def observe(info):
                guard();curves.append(dict(iteration=int(info.iterations),gap=float(info.gap_abs),primal=float(info.res_primal),dual=float(info.res_dual)))
                return callback(info)
            self.solver.set_termination_callback(observe)
        def solve(self):
            sol=self.solver.solve();solutions.append(sol)
            if self.cut:
                # Restore a harmless zero epigraph only for the legacy certificate
                # checker; all optimized variables came from the reduced problem.
                class Result:pass
                r=Result()
                for k in ['status','iterations','z','s']:setattr(r,k,getattr(sol,k))
                r.x=list(sol.x)+[0.];return r
            return sol
    F.clarabel.DefaultSolver=Wrapped
    error=None;returned=None
    try:W,returned=F.conic(stat,W0,'mean_taskwise',guard,lambda st,k,kind:events.append(dict(stage=st,iteration=k,kind=kind)),time_limit=60.)
    except BaseException:
        import traceback;error=traceback.format_exc()
    finally:F.clarabel.DefaultSolver=original
    result=dict(mode=mode,error=error,curves=curves,events=events,matrix=modeldata,raw_status=str(solutions[-1].status) if solutions else None,iterations=int(solutions[-1].iterations) if solutions else 0,
        no_actor_assignments=True,no_native=True,new_authorized_diagnosis_not_original_retry=True,epoch=time.time(),certificate=returned)
    if solutions:
        s=solutions[-1];np.savez_compressed(OUT/('diagnosis_'+mode+'_solution.npz'),x=np.asarray(s.x),z=np.asarray(s.z),s=np.asarray(s.s))
    save(DOC/('DIAGNOSIS_'+mode.upper()+'.json'),result)
    if mode=='red':assert error and "'NumericalError', 17" in error,('Original exact failure did not reproduce',result['raw_status'],result['iterations'])
    else:assert not error,('Equivalent reduced representation did not pass',error)
    print(json.dumps(dict(mode=mode,verdict='RED_REPRODUCED' if mode=='red' else 'GREEN_CERTIFIED',status=result['raw_status'],iterations=result['iterations'],matrix=modeldata)),flush=True)

if __name__=='__main__':main(sys.argv[1])

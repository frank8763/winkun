"""Read-only finite-basis interference diagnosis. No solve, update, or simulator."""
from pathlib import Path
import sys,json,time,os
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t575_runtime as R

def main():
 R.setup();p=R.configuration();R.verify(p,True);assert R.read(R.OUT/'audit.json')['state']=='PASS';sealed=R.read(R.OUT/'SEALED.json');assert all(not Path('/proc',str(pid)).exists() for pid in sealed['scientific_pids']);torch=R.torch;torch.set_num_threads(1);q=R.L.frozen(p,p['initial']['path']);s=R.read(R.OUT/'summary.json');new=R.L.frozen(p,s['result']['proposal']['path']);before=q.state_dict();after=new.state_dict();changed=[k for k in before if not torch.equal(before[k],after[k])];assert changed==['continual.2.weight','continual.2.bias'];Hs=[];measure=[];lost=set(s['result']['check']['lost_current21']);memory=R.G.rows(R.OUT,'memory_rows.jsonl');started=time.time()
 with torch.no_grad():
  for row in memory:
   assert R.sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);H,b=R.L.basis(q,x);z=new.sequence(x);delta=z-b;command_delta=z.clamp(-1,1).float().double()-b.clamp(-1,1).float().double();assert float((b-torch.tensor(a['locations'])).abs().max())<1e-9;Hs.append(H)
   measure.append(dict(id=row['job_id'],lost_in_autonomous_test=row['job_id'] in lost,n=len(H),raw_location_RMS=float(delta.square().mean().sqrt()),raw_location_absmax=float(delta.abs().max()),command_absmax=float(command_delta.abs().max()),command_RMS=float(command_delta.square().mean().sqrt())))
  H=torch.cat(Hs);sing=torch.linalg.svdvals(H);rank=int(torch.linalg.matrix_rank(H));nullity=H.shape[1]-rank
 out=dict(state='READ_ONLY_ANALYSIS_COMPLETE',original_summary_sha256=R.sha(R.OUT/'summary.json'),source_initial=p['initial'],source_proposal=s['result']['proposal'],changed_state_keys=changed,old_history_design_shape=list(H.shape),rank_default_torch_tolerance=rank,nullity_per_action=nullity,smallest_singular_value=float(sing[-1]),largest_singular_value=float(sing[0]),condition_number=float(sing[0]/sing[-1]),finite_basis_implication='If full column rank, exactly zero raw-location changes at all retained old histories force zero final-layer coefficient change. This does not prove old success and new success cannot coexist, does not prove backbone capacity insufficient, and does not certify trajectory stability.',per_old_task=measure,new_native=0,new_Adam=0,new_RL=0,new_linear_solves=0,elapsed_s=time.time()-started)
 dst=R.DOC/'POSTHOC_INTERFERENCE.json';assert not dst.exists();R.save(dst,out);print(json.dumps({k:v for k,v in out.items() if k not in ['per_old_task','source_initial','source_proposal']},ensure_ascii=False));print(json.dumps([x for x in measure if x['lost_in_autonomous_test']],ensure_ascii=False))
if __name__=='__main__':main()

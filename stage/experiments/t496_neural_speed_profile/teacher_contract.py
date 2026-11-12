"""Zero-native/zero-Adam check of the exact registered temporal teacher."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import speed_runtime as R
G,S,torch=R.G,R.S,R.torch
p=R.read(R.OUT/'protocol.json');R.verify(p,True)
prior=G.ROOT/'stage/runs/t494_timeout_cause'
row=next(r for r in G.rows(prior) if r['job_id']=='wide117_8m_dev6_RIGHT_17')
a=G.E.N.S.arrays(row['path']);teacher=R.pulse_frozen(p,p['initial']['path']);base=S.PLAIN_FROZEN(p,p['initial']['path']);h=hb=None
with torch.no_grad():
    for i,x in enumerate(a['states'][:100]):
        command,loc,h,_=teacher.step(torch.tensor(x[None]),h)
        expected,oldloc,hb,_=base.step(torch.tensor(x[None]),hb);expected=expected.clone()
        if i<p['pulse_decisions']:expected[:,0]=(expected[:,0]+expected.new_tensor(p['raw_sigma'][0])).clamp(-1,1)
        assert torch.equal(command,expected) and torch.equal(h,hb) and torch.equal(loc,oldloc)
        assert G.digest(teacher.tape[-1])==G.digest(base.tape[-1])
R.save(R.OUT/'teacher_forward_contract.json',dict(state='PASS',native=0,Adam=0,first50_exact_existing_sigma_drive_increment=True,after50_exact_original_actor=True,unchanged_raw_proposal_and_hidden=True,source_actual_history=row['sha256'],not_navigation_success=True))
print('T496_TEACHER_FORWARD_CONTRACT_PASS')

"""Same T641 experiment; one risk aggregation change and full critical trace."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t642_tail_retention';DOC=ROOT/'stage/docs/day_20261118/t642'
sys.path.insert(0,str(ROOT/'stage/experiments/t641_verified_recovery_learning'))
import t641_runtime as Old
import t642_learning as Learning
R=Old.R;Old.HERE=HERE;Old.OUT=OUT;Old.DOC=DOC
R.HERE=HERE;R.OUT=OUT;R.DOC=DOC;Old.Mem.HERE=HERE;Old.Mem.OUT=OUT;Old.Mem.DOC=DOC
R.M=Learning
ORIGINAL_REQUEST=R.Runtime.request


class Runtime(R.Runtime):
    def request(self,jobs,model,role,full=False,references=None):
        if role!='pre_old6':return super().request(jobs,model,role,full,references)
        result=[]
        for job in jobs:
            # Capture actual failing-path states, rather than inferring them
            # from same-state action comparisons or compact command records.
            result+=super().request([job],model,role,full=job['id']==self.p['critical_trace_id'])
        return result


if __name__=='__main__':
    R.Runtime=Runtime
    def no_network(*args,**kwargs):raise RuntimeError('Verified RAM only during science')
    R.D.remote_bytes=no_network;R.E.N.offline()
    {'main':Old.main,'audit':Old.audit,'verify_teacher':Old.verify_teacher,'preflight_teacher':Old.preflight_teacher}[sys.argv[1]]()

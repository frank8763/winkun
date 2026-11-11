"""Actual runtime-audit symbol seam, zero native/optimizer/model forward."""
import ast,json
from pathlib import Path
import t484_learning as L

def main():
    here=Path(__file__).resolve().parent
    for f in ['t484_runtime.py',str(here.parent/'t482_evening_coverage/t482_runtime.py')]:
        p=Path(f) if Path(f).is_absolute() else here/f
        attrs={n.attr for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='L'}
        assert all(hasattr(L,a) for a in attrs),attrs-set(vars(L))
    assert L.vtrace is L.F.B.vtrace and L.fit is L.F.fit and L.batch is L.F.batch
    r=L.torch.tensor([.2,1.],dtype=L.torch.float64);d=r.new_tensor([.9,0]);v=r.new_tensor([.1,.3,0])
    target,adv,weight=L.vtrace(r,d,v,r.new_zeros(2))
    L.torch.testing.assert_close(target,r.new_tensor([1.1,1.]),rtol=0,atol=1e-14)
    assert L.torch.equal(weight,L.torch.ones_like(weight)) and not target.requires_grad
    print(json.dumps(dict(state='PASS_REAL_AUDIT_SYMBOL_AND_MC_IDENTITY',native=0,optimizer=0,model_forward=0,training_functions_exact_same_objects=True)))
if __name__=='__main__':main()

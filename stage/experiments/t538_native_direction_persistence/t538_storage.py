"""Shape-bounded uncompressed NPZ: save each completed branch immediately."""
from pathlib import Path
import io,os
import numpy as np
def encode(branches):
    assert branches and all(set(b)==set(branches[0]) for b in branches)
    fields={k:np.stack([b[k] for b in branches]) for k in sorted(branches[0])}
    f=io.BytesIO();np.savez(f,**fields);return f.getvalue()
def write(path,branches,limit):
    blob=encode(branches);assert len(blob)<=limit,(len(blob),limit)
    path=Path(path);tmp=Path(str(path)+'.partial')
    with tmp.open('wb') as h:h.write(blob);h.flush();os.fsync(h.fileno())
    tmp.replace(path)
def branch(z,index):return {k:np.asarray(z[k][index]) for k in z.files}

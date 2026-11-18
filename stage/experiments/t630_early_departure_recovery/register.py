"""Register only after this experiment's explicit window/quota approval."""
from pathlib import Path
import hashlib
import json
import time
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/day_20261118/t630'
OUT=ROOT/'stage/runs/t630_early_departure_recovery'


def main():
    p=json.loads((HERE/'protocol.json').read_text())
    a=json.loads((DOC/'AUTHORIZATION.json').read_text())
    assert a['state']=='APPROVED' and a['max_physics']==p['max_physics']
    assert a['max_commands']==18 and a['new_optimizer']==0
    assert a['relative_all_stop_seconds']==3600
    assert not OUT.exists()
    assert json.loads((DOC/'PREFLIGHT.json').read_text())['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    assert json.loads((DOC/'RESET_PREFLIGHT.json').read_text())['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    now=time.time();p.update(authorization_approved=True,
        absolute_learning_stop=now,absolute_native_stop=now+1200,absolute_all_stop=now+3600,
        registration_epoch=now,authorization_sha256=hashlib.sha256((DOC/'AUTHORIZATION.json').read_bytes()).hexdigest())
    p['inputs'][str(DOC/'AUTHORIZATION.json')]=p['authorization_sha256']
    for f,h in p['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    b=(json.dumps(p,indent=2)+'\n').encode()
    (HERE/'protocol.json').write_bytes(b);OUT.mkdir();(OUT/'protocol.json').write_bytes(b)
    print('T630_REGISTERED_NOT_LAUNCHED',int(now+3600))


if __name__=='__main__':main()

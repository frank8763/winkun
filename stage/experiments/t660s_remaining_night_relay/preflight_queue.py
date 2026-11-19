"""Boundary fixtures of actual admission; zero physics/optimizers/network."""
from pathlib import Path
import json,time
import admission
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/overnight_20261119/resume'
def main():
    p=json.loads((HERE/'protocol.json').read_text());c=p['rl_template'];now=time.time()
    ledger=dict(physics=0,commands=0,main_seconds=0.,Actor_live=0,Value_live=0)
    assert admission.allowed(p,c,ledger,now,0)[0]
    rejected=[]
    for key,used in [('physics','night_max_physics'),('commands','night_max_commands'),('main_seconds','night_main_seconds'),('Actor_live','Actor_each_cap'),('Value_live','Value_each_cap')]:
        bad=dict(ledger);bad[key]=p[used]
        assert not admission.allowed(p,c,bad,now,0)[0];rejected.append(key)
    assert not admission.allowed(p,c,ledger,now,p['max_bytes'])[0]
    assert not admission.allowed(p,c,ledger,p['absolute_learning_stop'],0)[0]
    assert not admission.allowed(p,c,ledger,p['absolute_native_stop']-c['main_seconds'],0)[0]
    assert admission.stable_screen(['old1','old2'],['old1','new3'])==(False,['old2'])
    assert admission.stable_screen(['old1','old2'],['old1','old2','new3'])==(True,[])
    value=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',all_compute_caps_rejected=rejected,
               storage_and_learning_native_deadline_rejected=True,repair_one_lose_one_screen_rejected=True,
               four_fixed_mechanisms=4,coverage_finite_batches=p['coverage_batches'],new_native=0,new_optimizer=0)
    (DOC/'QUEUE_PREFLIGHT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()

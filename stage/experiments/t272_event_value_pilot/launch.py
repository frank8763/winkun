"""One registered offline fit; never restart an ended directory."""
from pathlib import Path
import json
import subprocess
import time
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t272_event_value_pilot'


def main():
    assert not any((OUT/n).exists() for n in ['launch.json','status.json','summary.json'])
    assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    plan=json.loads((HERE/'protocol.json').read_text())
    log=(OUT/'run.log').open('xb')
    proc=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'train.py')],cwd=ROOT,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    with (OUT/'launch.json').open('x') as f:json.dump(dict(pid=proc.pid,launch_epoch=time.time(),single_launch=True,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        remote_source_commit=receipt['remote_commit'],wall_seconds=plan['wall_seconds'],
        physics_budget=0,Actor_update_budget=0,event_model_optimizer_budget=500),f,indent=2)
    log.close();print(json.dumps(dict(pid=proc.pid,offline_event_fit=True)),flush=True)


if __name__=='__main__':main()

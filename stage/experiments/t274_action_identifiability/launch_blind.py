"""Unique source-registered offline action-availability ablation."""
from pathlib import Path
import json,subprocess,time
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t274_action_identifiability'


def main():
    assert not any((OUT/n).exists() for n in ['launch.json','status.json','summary.json'])
    assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    log=(OUT/'run.log').open('xb');proc=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'run_blind.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    with (OUT/'launch.json').open('x') as f:json.dump(dict(pid=proc.pid,launch_epoch=time.time(),single_launch=True,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        remote_source_commit=receipt['remote_commit'],wall_seconds=300,physics_budget=0,Actor_update_budget=0,event_optimizer_budget=500),f,indent=2)
    log.close();print(json.dumps(dict(pid=proc.pid)),flush=True)


if __name__=='__main__':main()

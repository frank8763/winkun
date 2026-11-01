from pathlib import Path
import json,subprocess,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t276_future_time_pilot'


def main():
    assert not any((OUT/n).exists() for n in ['status.json','summary.json','launch.json'])
    assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    log=(OUT/'run.log').open('xb');p=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'train_time.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    with (OUT/'launch.json').open('x') as f:json.dump(dict(pid=p.pid,launch_epoch=time.time(),single_launch=True,wall_seconds=300,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_source_commit=receipt['remote_commit'],
        optimizer_budget=500,physics_budget=0,Actor_update_budget=0),f,indent=2)
    log.close();print(json.dumps(dict(pid=p.pid)),flush=True)


if __name__=='__main__':main()

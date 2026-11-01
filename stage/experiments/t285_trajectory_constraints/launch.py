from pathlib import Path
import json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t285_trajectory_constraints'


def main():
    assert not any((OUT/n).exists() for n in ['launch.json','status.json','summary.json'])
    assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    env=os.environ.copy();env['STAGE_PROCESS_CPU']='0'
    with (OUT/'run.log').open('xb') as f:
        process=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage T285 two fixed trajectory constraints',str(ROOT/'run_stage.sh'),str(HERE/'run.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    with (OUT/'launch.json').open('x') as f:json.dump(dict(pid=process.pid,launch_epoch=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_commit=receipt['remote_commit'],wall_seconds=900,single_launch=True),f,indent=2)
    print(json.dumps(dict(pid=process.pid)))


if __name__=='__main__':main()

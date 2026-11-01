from pathlib import Path
import json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t278_fixed_goal_starts'


def main():
    assert not any((OUT/n).exists() for n in ['launch.json','status.json','summary.json'])
    assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    env=os.environ.copy();env['STAGE_PROCESS_CPU']='2'
    log=(OUT/'run.log').open('xb');p=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage T278 fixed final goal frozen starts',str(ROOT/'run_stage.sh'),str(HERE/'run_probe.py')],
        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    with (OUT/'launch.json').open('x') as f:json.dump(dict(pid=p.pid,launch_epoch=time.time(),single_launch=True,wall_seconds=600,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_source_commit=receipt['remote_commit'],
        CPU_affinity_requested=2,commands_budget=8,physics_budget=224000,optimizer_budget=0),f,indent=2)
    log.close();print(json.dumps(dict(pid=p.pid)),flush=True)


if __name__=='__main__':main()

"""User-requested schedule revision: stop only this Stage at durable round boundary."""
from pathlib import Path
import json,os,signal,time,subprocess
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'stage/runs/t663_daytime_factorial_relay'
DOC=ROOT/'stage/docs/day_20261120/t664'
def read(p):return json.loads(Path(p).read_text())
def owned(pid):
    assert '/stage-t663.service' in Path('/proc',str(pid),'cgroup').read_text()
def main():
    os.sched_setaffinity(0,{14});os.nice(19)
    driver=read(OLD/'DRIVER_STARTED.json')['pid'];owned(driver)
    state=read(OLD/'status.json');assert state['child']=='ppo_binary_s005' and state['action']=='main'
    child=state['child_pid'];owned(child);assert os.getpgid(child)==child
    helper=read(ROOT/'stage/docs/day_20261120/t663/AFFINITY_GUARD.json')['pid'];owned(helper)
    os.kill(driver,signal.SIGSTOP)
    deadline=min(time.time()+1800,read(OLD/'protocol.json')['absolute_learning_stop'])
    while time.time()<deadline:
        f=OLD/'ppo_binary_s005/ROUNDS.json'
        if f.exists():
            try:rounds=read(f)
            except json.JSONDecodeError:time.sleep(.05);continue
            if rounds:
                # No source or run mutations: freeze helper before freezing the
                # owned computation group so it cannot auto-CONT this pause.
                os.kill(helper,signal.SIGSTOP);os.killpg(child,signal.SIGSTOP)
                calls=OLD/'ppo_binary_s005/optimizer_calls.jsonl'
                entries=[json.loads(x) for x in calls.read_text().splitlines()] if calls.exists() else []
                receipt=dict(state='USER_REQUESTED_COMPACT_SCHEDULE_BOUNDARY_STOP',epoch=time.time(),
                    driver=driver,child=child,helper=helper,complete_rounds=len(rounds),
                    optimizer_entries=entries,last_status=read(OLD/'status.json'),
                    full_original_child_reservation_charged=True,raw_preserved=True,
                    reason='User requested shorter experiment segments; do not edit running protocol or repeat completed blocks')
                (DOC/'BOUNDARY_STOP.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
                os.killpg(child,signal.SIGKILL)
                subprocess.run(['systemctl','--user','kill','--signal=SIGKILL','--kill-whom=all','stage-t663.service'],check=True)
                print('DURABLE_BOUNDARY_STOPPED',len(rounds),flush=True);return
        if not Path('/proc',str(child)).exists():raise RuntimeError('Scientific child ended before requested boundary')
        time.sleep(.1)
    # Do not leave the healthy original supervisor parked on monitor failure.
    os.kill(driver,signal.SIGCONT)
    raise TimeoutError('Boundary not reached; original supervisor resumed')
if __name__=='__main__':main()

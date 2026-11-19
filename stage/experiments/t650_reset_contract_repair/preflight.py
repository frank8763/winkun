"""Capture the post-setup factory; regression invokes the actual repair seam."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,json,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/day_20261119/t650'
sys.path.insert(0,str(ROOT/'stage/experiments/t630_early_departure_recovery'))
import reset_contract as Broken
import consistent_reset as Fixed


def fixture():
    calls=[];R=SimpleNamespace(task_env=None);F=SimpleNamespace(E=SimpleNamespace(N=SimpleNamespace(K=SimpleNamespace(R=R),science=lambda p:p)))
    def bare(create,job):calls.append('bare');return ('env','narrow','actual')
    def base_setup(folder,p):
        def wide(create,job):calls.append('wide_radial');return ('env','wide_radial','actual')
        R.task_env=wide;return ('module','native','create','witness','reward')
    F.BASE_SETUP=base_setup;F.BASE_TASK=bare
    return F,base_setup,R,calls


def main():
    F,setup,R,calls=fixture();Broken.field_setup(F,None,{})
    old=R.task_env(None,{'wide_course':True,'radial_distance_m':8})
    assert old[1]=='narrow' and calls==['bare']
    F,setup,R,calls=fixture();Fixed.setup(F,None,{},setup)
    new=R.task_env(None,{'wide_course':True,'radial_distance_m':8})
    assert new[1]=='wide_radial' and calls==['wide_radial']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    (DOC/'FACTORY_REGRESSION.json').write_text(json.dumps(dict(state='PASS_RED_OLD_GREEN_REPAIRED_FACTORY',old_observation=old[1],repaired_observation=new[1],actual_reset_constructor_seam=True,new_native=0,new_optimizer=0),indent=2)+'\n')
    print('OLD_RED_NEW_GREEN_SETUP_FACTORY_CAPTURE_NO_NATIVE')

if __name__=='__main__':main()

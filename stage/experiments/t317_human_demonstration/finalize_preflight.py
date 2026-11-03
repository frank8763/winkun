"""Complete guard/HTTP checks, reuse the passed native geometry/render probe."""
from recorder import *
import ast,types
def main():
    assert not any((OUT/n).exists() for n in ['preflight.json','launch.json','status.json']);old=HERE/'preflight_superseded_001';proof=read(old/'preflight.json');assert proof['state']=='PASS' and proof['physics_steps']==0
    def classes(path):
        text=path.read_text();return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.ClassDef)}
    before,after=classes(old/'recorder.py'),classes(HERE/'recorder.py')
    for name in ['Session','View']:assert before[name]==after[name],name
    assert K.C.sha(OUT/'preview.jpg')==read(old/'notes.json')['preview_sha256']
    p=read(HERE/'protocol.json');inputs=dict(p['input_hashes'])
    for f in HERE.glob('*.py'):inputs[str(f)]=K.C.sha(f)
    p['input_hashes']=inputs;p['prelaunch_engineering_contract']='Actual Session/View classes are byte-source-identical to passed initial geometry/render preflight. HTTP input errors distinct from executionfailure; true failures closecollector and preservecheckpoint. Reserved180s usedforlastactionreplay until15sbeforeabsoluteend. No extra factory/native/optimizer/gradient/action command; originalpreflight source/protocol/proof retained.'
    for f in [HERE/'protocol.json',OUT/'protocol.json']:K.C.save(f,p)
    app=App.__new__(App);app.p=p;app.meter=types.SimpleNamespace(hold=lambda:None,physics=0);app.deadline=time.time()+60;app.in_replay=False
    try:app.guard()
    except AssertionError:pass
    else:raise AssertionError('Collection accepted inside reserved replay window')
    app.in_replay=True;app.guard()
    app.deadline=time.time()-1
    try:app.guard()
    except AssertionError:pass
    else:raise AssertionError('Replay accepted after absolute deadline')
    verify(p)
    K.C.save(OUT/'preflight.json',dict(proof,protocol_sha256=K.C.sha(HERE/'protocol.json'),original_preflight_geometry_not_repeated=True,original_actual_Session_View_source_unchanged=True,reserved_replay_window_and_absolute_stop_contract_PASS=True,HTTP_input_errors_vs_fatal_execution_distinct=True,extra_physics_steps=0,extra_geometry_commands=0,extra_Adam_or_gradient_calls=0))
    print('T317 final preflight PASS; reused originalgeometry/render,0extra native/commands/optimizer')
if __name__=='__main__':main()

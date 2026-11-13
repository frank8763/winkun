"""Audit the actual SGD and retained partial evaluation after output-route stop."""
from pathlib import Path
import copy,json,os,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t539_runtime as R
def main():
    root=R.read(R.OUT/'protocol.json');p=R.configuration(root);R.verify(p,True);s=R.read(R.OUT/'summary.json')
    assert s['state']=='FAILED' and not Path('/proc',str(s['pid'])).exists() and not (R.OUT/'audit.json').exists()
    relocation=R.read(R.DOC/'T539_OUTPUT_RELOCATION.json');assert relocation['all_original_T531_archived_SHA_unchanged']
    mapping={r['original_misdirected_path']:r['canonical_path'] for r in relocation['new_T539_files']}
    assert all(R.sha(f['canonical_path'])==f['sha256'] for f in relocation['new_T539_files'])
    obj=R.L.make(p);items=R.L.arrays(p);before=R.L.state(obj)
    for row in p['training_rows']:
        rr=copy.deepcopy(row);rr['request']['model']=p['initial']['path'];assert rr['policy_hash']==p['initial']['policy_hash']
        R.F.M.audit_full_history(p,rr,R.E.N.S.arrays(row['path']))
    u=R.L.fit(obj,items,p,lambda:None,lambda *_:None);assert u==s['result']['update'] and R.digest(R.L.state(obj))==s['result']['model']['joint_hash']
    assert R.sha(s['result']['model']['path'])==s['result']['model']['sha256']
    controls=[R.E.N.S.arrays(r['path']) for r in p['old_success_rows']];assert R.L.trust(p,obj[0],items+controls)==s['result']['trust']
    rows=R.G.rows(R.OUT/'misdirected');assert len(rows)==10
    for r in rows:
        f=Path(mapping[r['path']]);assert R.sha(f)==r['sha256'];a=R.E.N.S.arrays(f)
        assert set(a)=={'actions','native_counts','end_flags'} and all(R.digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert int(a['native_counts'].sum())==r['physics'] and a['end_flags'][-1].tolist()==[int(r['collision']),int(r['success']),int(r['timeout'])]
    generations=R.read(R.OUT/'misdirected/worker_generations.json');ids=[s['pid'],R.read(R.OUT/'launch.json')['pid']]+[i for g in generations for i in [g['parent'],*g['workers']]]
    assert all(not Path('/proc',str(i)).exists() for i in ids)
    failures=[R.read(f) for f in (R.OUT/'misdirected/workers').glob('*/failure.json')]
    assert len(failures)==2 and sum(f['physics'] for f in failures)==s['physics']==235017
    complete=sum(r['physics'] for r in rows);assert complete==191394
    assert R.G.rows(R.OUT,'optimizer_calls.jsonl')==[[w,t,0] for w in ['live','shadow'] for t in ['BEFORE','AFTER']]
    result=dict(state='PASS',scope='ONE_REAL_SGD_AND_TEN_COMPLETED_CASES_PARTIAL_ONLY',pid=os.getpid(),new_native=0,discarded_historical_SGD=1,original_source_Actor_Adam_and_Value_unchanged=True,new_physics=235017,completed_case_physics=complete,interrupted_two_case_physics=235017-complete,commands=s['commands'],old8=R.G.R.M.tally(rows[:8]),field24=None,first_two_field_cases_only=R.G.R.M.tally(rows[8:]),remaining22_not_completed=True,all32exact_not_executed=True,all_original_T531_archived_hashes_preserved=True,canonical_relocation_ledger_sha256=R.sha(R.DOC/'T539_OUTPUT_RELOCATION.json'),candidate_not_promoted=True)
    R.save(R.OUT/'audit.json',result);R.save(R.OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=sorted(set(ids)),finalizer_writes_only_docs_after_this_point=True,engineering_output_relocation=True,epoch=time.time()))
    R.save(R.DOC/'T539_PARTIAL_AUDIT_REGISTRATION.json',dict(new_native=0,discarded_historical_SGD=1,script_sha256=R.sha(__file__),original_summary_sha256=R.sha(R.OUT/'summary.json'),scope=result['scope'],original_protocol_unchanged=True))
    print('T539_PARTIAL_AUDIT_PASS_OLD8_ALL8_FULLFIELD_INCOMPLETE',flush=True)
if __name__=='__main__':main()

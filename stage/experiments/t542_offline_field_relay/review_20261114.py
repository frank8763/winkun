"""Read-only raw closeout after the expired night; zero physics/optimizer.

Do not alter failed summary or rerun ended work. Completed-child existing
independent audits remain their original evidence; this review checks persisted
files, counts and the partial final evaluation, not a full learning re-audit.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t542_runtime as R
OUT=R.OUT;DOC=R.DOC
read,save,sha,digest=R.read,R.save,R.sha,R.digest

def main():
 dest=DOC/'MORNING_RAW_REVIEW.json';assert not dest.exists();p=read(OUT/'protocol.json');sealed=read(OUT/'SEALED.json');launch=read(OUT/'launch.json')
 assert all(not Path('/proc',str(i)).exists() for i in sealed['scientific_pids']+[launch['pid']]);assert sha(p['source_initial']['path'])==p['source_initial']['sha256']
 assert all(sha(f)==h for f,h in p['inputs'].items());assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
 preserved=[];runs=[];allphys=allcmd=0;live=0;traincounts=dict(valid=0,success=0,contact=0,timeout=0);arms={};initial=p['initial_check'];peaks=[]
 for folder in sorted(x for x in OUT.iterdir() if x.is_dir() and not x.name.startswith('prelaunch')):
  s=read(folder/'summary.json');allphys+=s['physics'];allcmd+=s['commands'];rows=R.G.rows(folder);reps=R.G.rows(folder,'replays.jsonl');result=s.get('result') or {}
  if s['state']=='COMPLETE':
   assert read(folder/'audit.json')['state']=='PASS';m=read(folder/'local_manifest.json') if (folder/'local_manifest.json').exists() else None
   if m:
    for x in m['files']:assert (ROOT/x['path']).stat().st_size==x['bytes'] and sha(ROOT/x['path'])==x['sha256'];preserved.append(x['path'])
   assert s['physics']==sum(r['physics'] for r in rows+reps)
   by={r['path']:r for r in rows}
   for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
  if 'training' in result:
   arm=folder.name.rsplit('_',1)[0];a=arms.setdefault(arm,dict(rounds=0,live_SGD=0,shadow_SGD=0,independent_rebuild_discarded_SGD=0,retained_updates=0,train=dict(valid=0,success=0,contact=0,timeout=0),max_field_success=0,zero_signal_rounds=0));a['rounds']+=1;a['live_SGD']+=result['steps'];a['shadow_SGD']+=result['steps'];a['independent_rebuild_discarded_SGD']+=read(folder/'audit.json')['discarded_historical_SGD'];a['retained_updates']+=int(result['accepted']);a['zero_signal_rounds']+=int(result['steps']==0)
   live+=result['steps']
   for key in traincounts:a['train'][key]+=result['training'][key];traincounts[key]+=result['training'][key]
   if result['check']:
    c=result['check'];assert R.assess(R.plan(folder),rows[4:])==c;a['max_field_success']=max(a['max_field_success'],c['field24']['success'])
    if c['field24']['success']>11:peaks.append(dict(child=folder.name,old8=c['old8'],field24=c['field24'],gain=sorted(set(c['field_success_ids'])-set(initial['field_success_ids'])),lost=sorted(set(initial['field_success_ids'])-set(c['field_success_ids'])),accepted=result['accepted']))
  runs.append(dict(child=folder.name,state=s['state'],physics=s['physics'],commands=s['commands'],seconds=s['elapsed_s'],existing_independent_audit=(folder/'audit.json').exists()))
 final=OUT/'final_selected407';fs=read(final/'summary.json');assert fs['state']=='FAILED';rows=R.G.rows(final);assert len(rows)==350;before=0
 for r in rows:
  assert sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert int(a['native_counts'].sum())==r['physics'];assert a['end_flags'][-1].tolist()==[int(r['collision']),int(r['success']),int(r['timeout'])];before+=r['physics']
 partial=[]
 for f in final.glob('workers/*/partial.npz'):
  a=R.E.N.S.arrays(f);phys=int(a['native_counts'].sum());assert not a['end_flags'].any() and a['states'].shape[1]==120
  partial.append(dict(path=str(f),sha256=sha(f),physics=phys,decisions=len(a['states']),outcome='ADMINISTRATIVE_CENSOR_NOT_NAVIGATION_TIMEOUT'))
 partial_recorded=sum(x['physics'] for x in partial);unrecorded_hold_steps=fs['physics']-before-partial_recorded;assert unrecorded_hold_steps==59 and len(partial)==2;assert sum(read(f)['physics'] for f in final.glob('workers/*/failure.json'))==fs['physics']
 jobs=read(final/'protocol.json')['final_jobs'];by={r['job_id']:r for r in rows};groups={}
 for name,a,b in [('old275',0,275),('wide6_54',275,329),('wide8_54',329,383),('field24',383,407)]:
  rr=[by[j['id']] for j in jobs[a:b] if j['id'] in by];t=R.G.R.M.tally(rr) if rr else dict(valid=0,success=0,contact=0,timeout=0,invalid=0,censored=0)
  groups[name]=dict(expected=b-a,completed=len(rr),tally=t,not_completed=b-a-len(rr))
 for f in final.glob('workers/*/failure.json'):assert "p['native_stop_epoch']-5" in read(f)['error']
 root=read(OUT/'summary.json');assert allphys==root['physics']==53337590 and allcmd==root['commands']==2224
 assert allphys<p['max_physics'] and allcmd<p['max_commands'];assert live==88 and len(runs)==35
 stamp=lambda t:datetime.fromtimestamp(t,ZoneInfo('Asia/Shanghai')).isoformat()
 result=dict(state='PASS_PERSISTED_EVIDENCE_AND_PARTIAL_ACCOUNTING_ONLY',source_original_protocol_and_scientific_inputs_unchanged=True,all_own_processes_stopped=True,new_native=0,new_optimizer=0,not_a_full_failed_child_audit=True,previous_completed_child_independent_audits_verified=34,completed_child_manifest_members_checked=len(preserved),night_start=stamp(launch['epoch']),night_science_end=stamp(sealed['epoch']),science_elapsed_seconds=sealed['epoch']-launch['epoch'],driver_report_end=stamp(read(DOC/'FINAL_REPORT.json')['ended_epoch']),all_before_deadline=True,actual_physics=allphys,actual_commands=allcmd,rounds=32,arms=arms,total_training=traincounts,training_native_exact_repeats=128,live_Actor_SGD=88,shadow_SGD=88,independent_audit_discarded_SGD=88,preflight_synthetic_discarded_SGD=24,retained_new_updates=0,research_peaks=peaks,retained_model=read(DOC/'FINAL_REPORT.json')['best']['model'],retained_field24=initial['field24'],retained_old8=initial['old8'],final_evaluation=dict(state='PARTIAL_ADMINISTRATIVE_TIME_LIMIT_STOP',relative_limit_seconds=3300,actual_seconds=fs['elapsed_s'],completed=350,expected=407,groups=groups,partial=partial,complete_recorded_physics=before,partial_saved_complete_hold_physics=partial_recorded,charged_interrupted_hold_physics_without_observation_row=unrecorded_hold_steps,partial_physics=fs['physics']-before,not_started=55,censored_cases=2,whole407_success_rate=None),admission_stop=read(DOC/'ADMISSION_STOP.json'),scope_notes=['field11/24 is retained historical T507 result,not overnight new gain','T507 full old275245/275 is current research parent; safeT255 historical275/275 separate and unchanged','rtg12/24 gained a field task but old8 fell7/8 and contact10>9; cannot promote','two partial final trajectories are administrative censor,not native navigation timeouts','fresh16/reload78 not executed','no new training or deadline extension during morning inspection'],children=runs,epoch=time.time())
 save(dest,result);print('MORNING_RAW_REVIEW_PASS',len(preserved),'members',allphys,allcmd,flush=True)
if __name__=='__main__':main()

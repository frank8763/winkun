"""Freeze both source factors and test real own-carry coordinate identities."""
from pathlib import Path
import ast,copy,importlib.util,io,json,sys,time
import t682_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
T,M,torch,np=S.T,S.M,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def main():
    started=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    preceding=read(HERE/'protocol.json')['preflight_seconds'] if (HERE/'protocol.json').exists() else 0
    priorroot=ROOT/'stage/runs/t681_frozen_variant_transfer';prior=read(priorroot/'protocol.json');report=read(priorroot/'FINAL_REPORT.json')
    assert report['state']=='COMPLETE' and report['audit']['state']=='PASS' and report['helper_stopped']
    assert report['gate']=='FROZEN_VARIANT_TRANSFER_OBSERVED'
    seal=read(priorroot/'SEALED.json');ids=seal['scientific_pids']+[report['helper_pid'],read(priorroot/'DRIVER_STARTED.json')['pid']]
    assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in ids)
    p=copy.deepcopy(prior);p.pop('registration_epoch',None)
    # Remove unused inherited job copies and calibration payload. Actual
    # canonical evaluation manifest and dependencies remain unchanged.
    for key in ['calibration_rows','validation_rows','screen_jobs','key_jobs','training_jobs','field_jobs','coverage_jobs','paired_jobs',
                'pending_completion','scope_reference','collection_jobs','sentinel_jobs','reconstruction_jobs','precheck_jobs','replacement_jobs','transfer_jobs']:
        p.pop(key,None)
    old=['dev_task6_00','dev_task6_02','dev_task6_13','dev_task6_20']
    fields=['t502_field_sector'+k for k in ['0_offset1','0_offset2','1_offset1','2_offset1','5_offset1','7_offset1']]
    keys=old+fields;by={j['id']:j for j in p['evaluation_jobs']};jobs=[by[k] for k in keys]
    parents=prior['transfer_models'];assert [x['name'] for x in parents]==['direction00_minus','direction03_plus']
    auth=read(DOC.parent/'AUTHORIZATION.json');now=time.time()
    p.update(id='T682_STATIC_HEAD_FACTOR_ABLATION',authorization_approved=False,out=str(OUT),arm='head_factor',
        factor_parents=parents,panel_ids=keys,old_panel_ids=old,field_panel_ids=fields,factor_jobs=jobs,
        source_summaries=[str(ROOT/'stage/runs/t679_episode_parameter_exploration/summary.json'),str(priorroot/'summary.json')],
        baseline_old_cases=str(ROOT/'stage/runs/t501_joint_command_repair/joint/cases.jsonl.t595.xz'),
        max_physics=8*sum(j['case']['step_cap'] for j in jobs),max_commands=86,max_bytes=256*2**20,child_bytes=256*2**20,
        main_seconds=1800,audit_seconds=300,absolute_learning_stop=min(now+1800,auth['absolute_learning_stop']),
        absolute_native_stop=min(now+1800,auth['absolute_native_stop']),absolute_all_stop=min(now+2400,auth['absolute_all_stop']),
        finalization_seconds=60,new_Actor=0,new_Value=0,new_optimizer=0,model_bytes=2*2**20)
    assert p['max_physics']==2647464 and 4*len(jobs)*2+6==86
    assert 40*p['full_bytes']+40*p['compact_bytes']+4*p['model_bytes']+8*2**20<p['max_bytes']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);T.setup(p)
    assert read(ROOT/'stage/runs/t501_joint_command_repair/joint/audit.json')['state']=='PASS'
    baseline=S.baseline_rows(p);source=S.parent_rows(p)
    original=M.L.frozen(dict(p,raw_sigma=[.2,.2]),p['safe_retained']['path']);base=M.L.frozen(p,p['initial']['path'])
    assert digest(original.state_dict())==digest(base.state_dict())
    errors=[];hashes=[];source_files=set()
    for parent in parents:
        qsource=M.L.frozen(p,parent['model']['path']);assert sha(parent['model']['path'])==parent['model']['sha256']
        for head in ['drive','steer']:
            q=S.hybrid(p,parent,head);before=digest(q.state_dict());hashes.append(q.phash())
            for row in source[parent['name']]:
                assert sha(row['path'])==row['sha256'] and row['seed'] is None and row['loaded_model_sha256']==parent['model']['sha256']
                actual_job=copy.deepcopy(row['request']['job']);actual_job.pop('initial_contract',None)
                expected_job=copy.deepcopy(by[row['job_id']]);expected_job.pop('initial_contract',None);assert actual_job==expected_job
                source_files.add(Path(row['path']));aa=T.R.E.N.S.arrays(row['path']);T.N.inspect(p,aa,row)
                with torch.no_grad():
                    x=torch.tensor(aa['states']);initial=base.sequence(x);both=qsource.sequence(x);loc=q.sequence(x)
                assert float((both-torch.tensor(aa['locations'])).abs().max())<1e-10
                index=0 if head=='drive' else 1;err=max(float((loc[:,index]-both[:,index]).abs().max()),float((loc[:,1-index]-initial[:,1-index]).abs().max()))
                assert err<1e-12;errors.append(err)
            # In-memory checkpoint payload exercises exact state serialization;
            # real path loader already loaded both source and base above.
            packet=copy.deepcopy(M.P.checkpoint(p['initial']['path']));packet['joint_state']['actor']=copy.deepcopy(q.state_dict());packet['policy_hash']=q.phash()
            buf=io.BytesIO();torch.save(packet,buf);buf.seek(0);again=torch.load(buf,weights_only=False)
            loaded=M.L.Actor(p).eval().requires_grad_(False);loaded.load_state_dict(again['joint_state']['actor'])
            assert loaded.phash()==q.phash() and digest(q.state_dict())==before and all(not v.requires_grad for v in q.parameters())
    assert len(set(hashes))==4 and len(errors)==40
    # Gate logic fixture: no model can qualify by mixing parents' successes.
    def admissible(oldflags,sets):return all(oldflags) and len(set.intersection(*sets))>=2
    assert admissible([True,True],[{'a','b'},{'a','b','c'}])
    assert not admissible([True,False],[{'a','b'},{'a','b'}]) and not admissible([True,True],[{'a','b'},{'c','d'}])
    T.N.P.frozen=T.N.L.frozen;assert T.N.P.frozen is T.N.L.frozen
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    spec=importlib.util.spec_from_file_location('_t682_helper_preflight',HERE/'affinity_guard.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert helper.OUT==OUT and helper.CG.name=='stage-t682.service'
    resources=S.Wait.Raw.resources();assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',two_sources_four_static_heads=True,each_changed_head_parameters=5953,unchanged_backbone_and_other_head_exact=True,
        all40_external_history_coordinate_identity=True,max_coordinate_error=max(errors),all20source_full_checked=True,
        real_original_and_source_loaders=True,serialized_state_in_memory_exact=True,no_runtime_selector=True,no_gradients_or_optimizer=True,
        baseline_old4_original_audited_metadata_only_remote_raw_not_reread=True,gate_common_ID_fixture=True,canonical_worker_loader=True,
        resource_helper_configuration=True,new_native=0,new_optimizer=0,resources=resources)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in prior['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T682_PRIMARY_REVIEW.md'}|source_files
    deps|={priorroot/f for f in ['protocol.json','FINAL_REPORT.json','SEALED.json','DRIVER_STARTED.json','audit.json','summary.json','FIELD_TRANSFER.json']}
    deps|={Path(p['baseline_old_cases']),ROOT/'stage/runs/t501_joint_command_repair/joint/audit.json'}
    for key,row in baseline.items():
        if key in old:
            reference=Path(row['path']+'.t629-remote.json');meta=read(reference)
            deps|={reference,Path(meta['receipt']),Path(meta['manifest'])}
        else:deps.add(Path(row['path']))
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-started+preceding
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','absolute_native_stop','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre))

if __name__=='__main__':main()

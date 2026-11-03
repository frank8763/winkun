from t332_learning import *
import contextlib,importlib.util
sys.path.insert(0,str(ROOT/'stage/experiments/t280_local_cost'))
import t280_worker as Native
import t332_iid_native as IID

def worker(pipe,index,p,physical,preflight_only=False):
    q=model(p).requires_grad_(False);initial=K.S.digest(q.state_dict())
    if preflight_only:
        command=pipe.recv();q.load_state_dict({k:torch.tensor(v) for k,v in command['state'].items()},strict=True)
        x=torch.tensor(command['x']);output=q.step(x)
        pipe.send(dict(ready=True,policy_hash=K.S.digest(q.state_dict()),mode=output[0].numpy(),mean=output[1].numpy(),initial_hash=initial))
        pipe.close();return
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True);log=Native.LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;witness=None;total=0
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No worker optimizer'))
        spec=importlib.util.spec_from_file_location('stage332_course_native',ROOT/'stage/experiments/t328_wide_gap_80s/recorder.py')
        course=importlib.util.module_from_spec(spec);spec.loader.exec_module(course);course.ACTIVE=read(p['course_protocol'])
        native,create,witness,reward=course.setup(folder/'native');base_factory=course.original_task_env
        def factory(create,job):
            env,obs,actual=base_factory(create,job)
            if job.get('wide_course'):
                assert K.R.arrays_hash(K.R.integration(env))==course.ACTIVE['original_geometry']['integration_sha256']
                obs=course.widen(env,witness)
            return env,obs,actual
        K.R.task_env=factory;B.OUT=OUT
        assert B.R is K.R and Native.R is K.R
        pipe.send(dict(ready=True,pid=os.getpid(),CPU_affinity=sorted(os.sched_getaffinity(0)),initial_hash=initial))
        while True:
            job=pipe.recv()
            if job is None:break
            q.load_state_dict({k:torch.tensor(v) for k,v in job['policy_np'].items()},strict=True)
            assert K.S.digest(q.state_dict())==job['policy_hash'] and K.S.digest(q.frozen.state_dict())==p['frozen_policy_hash']
            if job['role']=='training':item=IID.generate(job,q,native,create,witness,job['deadline_epoch'],physical,index,total)
            else:item=Native.generate(job,q,native,create,job['deadline_epoch'],physical,index,total)
            total+=item['row']['physics_steps'];row=item['row'];row.pop('original_job',None)
            row.update(role=job['role'],policy_hash=job['policy_hash'],wide_course=job.get('wide_course',False),source_reset_id=job['source_reset_id'],cap=job['case']['step_cap'])
            # Ended env witnesses are no longer live; no new environment created.
            witness.events.clear();pipe.send(item)
    except EOFError:pass
    except BaseException:
        error=dict(error=traceback.format_exc(),completed_worker_physics=total);save(folder/'failure.json',error)
        try:pipe.send(error)
        except OSError:pass
    finally:
        if witness is not None:witness.close()
        pipe.close();log.close()

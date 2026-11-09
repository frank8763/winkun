# Joint recurrent Actor–Critic engineering draft

Stage only. No experiment is registered or launched in this directory. There is
no Actor, critic, simulator adapter, driver, optimizer or trained model here.
The proposal is in `stage/docs/research_rescue_20261109/DECISION_AND_RESCUE_PLAN.md`.

`physical_gae.py` is a NumPy reference for variable-duration, physical-time GAE.
It independently represents bootstrap permission and continuation of an advantage
trace. A finite task deadline is terminal; a rollout cut can bootstrap from the
actual final observation without leaking into an automatically reset episode.
The caller remains responsible for the underlying reward and censoring contract.
Per-native-step lambda is an explicit physical-time design choice, not an
original GAE formula or a locked scientific hyperparameter.

The deterministic contract checks use synthetic arrays only:

```bash
taskset -c 0 ./run_stage.sh stage/experiments/joint_recurrent_actor_critic/test_physical_gae.py
```

Passing these checks is not critic calibration, an Actor update, a navigation
experiment or evidence that GAE will solve the Stage task. Before any training,
register a finite protocol, information/action/distribution/sequence contracts,
actual capacity and throughput, source hashes and an absolute stop time. Keep
safeT255 frozen and all old failures preserved. Do not resume ended directories.

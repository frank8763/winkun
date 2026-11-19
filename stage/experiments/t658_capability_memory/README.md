# Capability-memory research prototype — not a registered experiment

This directory contains **offline contracts and exact synthetic fixtures only**.
There is no navigation learner, trained critic, simulator invocation, optimizer,
GPU use, launcher or new accepted model. The number in the directory name does
not register or authorize an experiment. The 2026-11-19 science window ended at
17:00 Beijing.

`continuation_contract.py` makes continuation identity and censoring explicit.
Its exact finite-state Bellman difference fixture illustrates why checking a
first action with the old continuation does not evaluate a changed complete
controller. The identity is standard policy-performance algebra, not a new
algorithm. The synthetic policy-family example also rejects assuming that
safe-reaching outcomes vary monotonically with update size.

`offline_preflight.py` verifies the bytes of the existing 384+8 source histories
and prepares one reference for each of their **387 unique task IDs**. T501 is
the reference for the five shared IDs; all eight field success requirements
remain. This repairs imitation-target bookkeeping; it does not show a single
policy achieves all 387 goals.

The saved reports live in `stage/docs/day_20261119/innovation_reset/`.
The design and novelty limits are in `RESEARCH_RESET_AND_PROPOSAL.md` there.
`policy_hash` is provenance, not a differentiable policy representation. No
certificate follows from a hash, low KL, an ensemble or these passing fixtures.

Offline reproduction (standard library, no network or science execution):

```bash
python3 -B stage/experiments/t658_capability_memory/offline_preflight.py
```

Do not connect this prototype to Actor updates. A later implementation needs
explicit policy-family inputs, valid candidate-continuation data, calibrated
held-out evaluation, a bounded protocol and a valid authorized window.

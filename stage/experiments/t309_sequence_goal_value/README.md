# T309 fixed H3 multi-goal value and DQC feasibility

Read distance_20261103/T309_PLAN.md and the frozen protocol. 64 real-goal
centers, 64 first-action controls, 128 exact replays, 16 zero-step initial
geometry checks. Three matched contrastive heads and a frozen-teacher DQC
student, 6000 total head Adam steps. The Actor is never updated in this run.

prepare.py → run.py preflight → clean local commit/exact remote full-root
source synchronization → launch.py once → stopped → audit.py → complete raw
remote binary/member readback → source synchronization. Existing files and
failed experiment directories are never restarted. Even a passing value gate
requires a separate bounded Actor protocol with complete capacity checks.

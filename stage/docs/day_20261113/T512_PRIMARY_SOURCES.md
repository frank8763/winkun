# T512 primary-source check, 2026-11-13

Checked publisher method summaries, not claimed full-paper reproduction:

- Ross, Gordon, Bagnell, [DAgger, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html): actions change later observation distributions, so ordinary iid supervised assumptions fail. Motivation for collecting at learner states. This experiment queries no expert and does not implement DAgger or inherit its guarantees.
- Florensa et al., [Reverse Curriculum Generation, CoRL 2017](https://proceedings.mlr.press/v78/florensa17a.html): adapting initial states around achievable goals can support sparse-reward learning. Here the state is reached through a complete actual student prefix, never arbitrary teleportation or resetting the remaining episode clock. Four bounded recovery probes are not a new reverse curriculum algorithm or training gain.

Project evidence: existing accepted parent has two near-goal timeouts. Earlier original-start stochastic successes did not survive supervised consolidation gates. Three ranked hypotheses: (1) recovery coverage missing on actual failure states, (2) terminal steering/speed coordination, (3) global command interference. This diagnostic can establish existence of sampled successful continuations from one state; it cannot establish unique cause, safe global policy improvement or publication novelty.

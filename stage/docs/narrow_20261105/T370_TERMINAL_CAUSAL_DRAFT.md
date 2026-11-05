# Next experiment draft: localize final goal-approach control before more training

This is a concrete reviewable draft followingT369; NOT registered, implemented or launched. User asked to check effects and think about the next experiment. Stage/Phase fully isolated. No additionalT369steps, noise/horizon/seed grids, successful-goal threshold changes or model promotion.

## Question

T369futurecentre traverses the scene, reaches a recorded0.314m near miss, then overshoots. At first distance<=1m its pose is close to a verified human approach state but its turn command differs. Can a short steering intervention at this *actual* approach state change terminal capture under the same native dynamics? If not, is the policy already outside its local recoverable region by this point? This isolates a local hypothesis, not all five narrow starts or global navigation.

## Fixed protocol to register later

FreezeT369futuremodel/rawdata/source hashes. Choose the first recorded decision with actual centre distance<=1m: index248/native24,800/49.6s, selected by one explicit rule, not a time/angle sweep. From the unchanged original6mstart/new0.70m80s scene, reproduce the full actual248-action prefix and compare every observable/native/history array before intervention. Preserve goal,elapsedclock,steering joints/actuators,speed,previous executed action and current-parameter GRUhistory. No pose/velocity/clock reset at takeover or arbitrary state teleport.

Four fixed conditions: original deterministic Actor continuation; five decisions (1s) steering−1; five decisionssteering0; five decisionssteering+1. During interventions preserve Actor-proposed drive and update GRU on every actual observation with the real previous executed action in packet. After those five decisions resume the exact same Actor, not a scripted path or rolling search. The unchanged control must exactly match the completeT369failure arrays/integration. Each condition gets one whole native regeneration and full-array comparison. All prefix/intervention/continuation/replay physical steps are counted.

Forced actions are external causal probes during research, never a deployed navigation policy, expert label or on-policy rollout. Do not pool branch success with original-start autonomous success. No planner,map,waypoints,runtime teacher/router in the tested Actor; do not train from these records as if they were human demonstrations. A single improved probe shows sensitivity/recoverability for that state, not a safe controller or generalized solution.

Record full120/pose/history/hidden/actual and proposed actions/override mask/nativecounts/GOAL-contact-timeout and native integration, nearest recorded/actual native distance where available, heading-to-goal and actual steering response. Retain failed/invalid/censored branches. The unmodifiedT369result remains immutable.

Suggested hard caps for a later formal registration:1zero-native geometry+4primary+4exact=9all commands;320,000all native steps (eight original80smax40k trajectories),0Actor/gradient/Adam;main900s,audit300s,backup600s,all1800s;own4MiB (4full<=512KiB,1partial<=512KiB,metadata/runtime<=1MiB),global1GiB/free40GiB with fresh actual worst-case admission. Source/model/protocol/hash/clock contracts and exact prefix control before unique launch; independent zero-native audit then fullRAM remote member-SHA backup. No automatic extra probes or arbitrary1m→2m→3m grid if this fails.

## Decision after the causal check

If a branch converts the same factual approach into safeGOAL, prioritize a separate training intervention for terminal goal alignment and its earlier recovery states, keeping whole-route/native retention checks. The causal probe itself is not learned-policy success or valid expert supervision; any new target generator needs an explicit separate contract, or actual expert labels. No goal-bearing penalty is assumed safe around obstacles.

If all interventions fail or only change failure mode, retain this negative result and examine earlier dynamics/recovery coverage using existing raw data before selecting another distinct mechanism. Do not merely add epochs, widen the0.20mgoalcircle, extend80s or promote a rejected candidate. Neither action sequencing nor fullACT/vision is ruled out by this local diagnostic. Full original6mrelease gates remain unchanged;7/8m waits.

# T357: original-start transfer to the feasible 0.70 m opening

Stage only; no navigation training, optimizer, planner, route, waypoint, runtime teacher, or policy selection. Freeze the selected T353 iid recurrent Actor, including its full causal hidden-state carry. Keep the T255 safety model untouched.

T356 established three local native passages through the 0.70 m scene with the complete tyre collision footprint. T355 never advanced beyond its 1.17 m opening. Therefore its failure cannot answer whether this policy can navigate an actually narrower feasible opening.

Run five consumed LEFT development resets (centre and ±3 cm translations) from the real six-metre snapshot in the 0.70 m scene, with the explicitly separate 80 s course cap. Run one original-start wide centre at 1.17 m / 80 s as a historical reproduction control, and the two unmodified original 0.57 m / 56 s centres as benchmark controls. Preserve all 120-dimensional states, actions, next states, histories, hidden states, density tape, and native integration. Regenerate every trajectory exactly. Repetitions and matching historical trajectories are not independent new samples.

Preflight: three zero-physics geometry environments, source/model SHA, initial history reconstruction, unchanged non-obstacle physics, active tyre masks, and 0.70 / 1.17 / 0.57 m gap checks. Main: eight primary native trajectories plus eight exact native regenerations. No retry, width/noise/seed grid, modification of a running protocol, or restart of an ended directory.

Budget: 19 commands including three preflight geometry commands; at most 640,000 native steps; main 1,200 s, independent zero-native audit 300 s, all work 1,800 s; own 8 MiB, global logical newruns 1 GiB, free disk at least 40 GiB. Eight full payloads at most 768 KiB each, one partial at most 768 KiB, metadata 768 KiB, logs 128 KiB. Full reservation precedes launch. External timeout and sleep inhibitor required. Audit and complete raw backup precede closeout.

Report success/collision/timeout/censor separately for new 0.70 m course and the original benchmark. Five known development successes, if obtained, are not the complete old275 / original six54 / fresh16 / reload78 gate. No seven/eight-metre training follows this diagnostic. If transfer fails, use the recorded complete causal states to locate the approach failure before registering a distinct bounded narrow-scene learning mechanism. Do not convert a local passage into a navigation expert or relabel original benchmark success.

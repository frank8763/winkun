# T369: matched future-action supervision failed the narrow-scene gate

Navigation/training COMPLETE in1246.411632s,163all commands,2,963,630all physical steps. Both matched arms completed600live+600exactshadow supervised Adam:1200live/1200shadow total. Independent audit reconstructed1200historical discardedAdam and passed all input/model/trace/mask/causality/hidden/history/reward checks;16preflight discardedAdam separate.96primary checks include32shared initial+32final/arm;64independent saved-model native readbacks all exact and are repeats, not new independent development samples. No new RL updates/native training collection.

| Fixed checks | Initial shared parent | Current-only arm | Future-supervision arm |
|---|---:|---:|---:|
| New0.70m/80s five |2/5|0/5 (5contact)|0/5 (4contact,1timeout)|
| Wide1.17m/80s five |0/5|3/5 (1contact,1timeout)|3/5 (2contact)|
| Old20 |9/20|9/20 (11contact)|10/20 (7contact,3timeout)|
| Original0.57m/56s centres |0/2|0/2 (2timeout)|1/2 (1contact)|

Both fitted candidates lost initial new-narrow success IDs and failed retention. They are preserved research models with the initial fallback, not promoted replacements. The one original-right-centre success is not the full54-case six-metre gate or an unseen new task. Originalold275/originalsix54/fresh16/release78 were not executed/null, not0%failures. Seven/eight metres remain unentered. SafeT255/T353/T365 and all failures remain intact.

Future-head rawsteerRMSE on human data improves versus the control's untrained future heads (~0.28/0.26→0.16/0.14), but current rawsteerRMSE worsens (~0.095/0.082→0.104/0.091). This registered auxiliary prediction change did not improve narrow closed-loop outcomes. It does not reject full ACT/CVAE/temporal ensembling, which were not implemented or deployed, nor prove all future-action supervision useless. Matched controls isolate this fixed future-loss intervention withinT369, not its difference fromT365 (the factual rehearsal recipe also changed).

## More specific centre failure

The future arm's centre survives without collision for80s and travels about8.5579m summed over recorded decision-endpoint segments. It reaches a recorded endpoint distance0.313863395m at57.2s, outside the unchanged0.20mGOAL radius, then moves away and times out at2.236375456m. This is not simply an inability to travel six metres or a too-short simulated duration. It does not prove every failed start shares this mechanism, nor provide the continuous-native minimum between stored endpoints.

At first recorded distance<=1m (49.6s), actualfuturecentrepose[1.842124,-3.601028,-1.280009],bodygoalbearing−0.087661rad,drive0.872016/steer+0.413213. At56–58s the target is increasingly on the negative-bearing side while observed yaw increments are positive and drive remains1; it passes the goal laterally. HumanT361firstwithin1m (51.2s) is at[1.864403,-3.580647,-1.268026] andsteer−0.999; T364at51.4s also usessteer−0.999. These are different full histories/states, not interchangeable expert action labels or controlled comparisons. Actual steering joint angles/rates are already observed. See T369_RECORDED_GOAL_APPROACH.json and T369_TERMINAL_COMPARISON.json.

Read-only uniform averaging of overlapping forecasts on human histories increases teacher-forced steerRMSE for both sources (futurearm~0.101/0.086→0.135/0.108 using clipped actions). This is descriptive, not a native result or proof against every ensemble. It does not justify automatically appending an ensemble/horizon grid toT369.

## Evidence closure

Original driver failed during second-segment GitHub metadata readback withTLS EOF; original DRIVER_ERROR and backup log remain unchanged. A transport-only retry reused verified attachments, downloaded all3segments and verified all121raw members and archive5342f570d4b81b5d6773ef9544e10e807f77650866b44d893d396d74f7a55eb0. No training/navigation/audit was restarted. All native/worker/inhibitor/driver PIDs are gone. Local121member and all3modelSHA rechecks PASS;global1,063,339,083B<1GiB,own3,498,916B<12MiB,free188,958,814,208B>=40GiB.

Fittedcurrentgzip5f9db7d171757a4dee661c13799da0674137e52e03c278a7d377ca51ee3de0d2;futuregzip19f428f8221967bf8e5823cd309652be3a4b7dfd522208cccd987cd91c4b79fb;initialfallbacke5c4aececc8f88db5f059c4598d612a0bb8bd7600c2405843f958a1e3c754ef3. Full rawsource/model/ActorAdam information is retained in original summaries and remote manifests.

Next priority is a fixed zero-optimizer causal check of goal-approach steering before1m, preserving actual prefix/clock/actuators/history, rather than increasing generic BC/RL epochs or modifyingGOALradius/time. Its draft is not a registered or launched run. A positive forced-action branch would be diagnostic recoverability evidence, not autonomous stability or on-policy training/expert data.

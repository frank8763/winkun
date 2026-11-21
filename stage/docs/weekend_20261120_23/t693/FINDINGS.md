# T693：启动前时间准入失败，科学部分未执行

T693 的真实零native零optimizer入口预检PASS、预算已独立RESERVED，源码精确GitHub同步后，启动前检查发现原absolute deadline剩余时间不足已预留完整main+audit路径。实际源码同步shell exit0，remote `224c34e9b84fc36d6ab0510acb6f8a9d864d3712`、树 `fac7adada325cddac1f9c1b635442bb54016976c` 与当时HEAD完全相同。未尝试launch，unit inactive/MainPID0，无launch/DRIVER_STARTED/science_started、无science PID。

原登记deadline为2026-11-21北京时间22:13:50.483903（epoch1795270430.483903），不修改、不重新锚定、不换编号重启相同pilot。PRELAUNCH_REJECTION记录实际检查时间及余量；完整main14400+audit1500需15900秒，仅物理按原2500/s admission亦需13057.1464秒。后来元数据复核余量9502.8809秒，连该物理预算也不足。不删除验收、不借已消费准备重新登记延时。

**实际0physics、0command、0Actor/Value/optimizer、0main/audit。没有72fresh72exact，没有真实actioncredit或新candidate，更没有科学audit/407性能结论。** 原PREFLIGHT是实现合同，不能称MVD已失败、方差大或Actor保持失败。本次是登记至launch之间墙钟流失导致的工程准入失败；不能将它包装成一次新科学负结果。没有证据将整个等待间隔当作网络同步计算时间，传输仍按既有300秒上界逐commit去重结算。

预检原失败F.K属性不存在时0native0optimizer，原错误/源码快照保留；修为既有E.N.K.R入口后mainbody20.388s+进口10s+失败20s+中断补充10s+finalize10s（正式合并preflight账70.9138秒，含测量尾开销），另早期PREPARATION10秒已单独结算，均无科学更新。最坏保留的32,642,866physics/1086cmd/14400main1500audit和Actor各2没有实际消费，ledger解除未执行预留且保留原登记行和preflight耗时。

全部实现/协议/准备/预检/拒绝证据保留。T501继续保留；本轮无新学习。后续不得把同一MVD/score采样设计换号重新给deadline，不继续同slot/seed/sample/weight诊断。若有不同原理的新机制，仍需一手/历史反证审查和完整入口预算；若无有意义事项，应如实收尾，不为填时钟制造实验。

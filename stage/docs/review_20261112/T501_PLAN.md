# T501正式协议：双动作神经修复与340项真实成功保持

用户2026-11-12明确“好，下一轮开始”。这是新的有界训练和383完整验收，不复用T498/T500预算；原八米通过模型与全部失败证据保持。启动仅以preflight PASS、源码远端精确树、launch与实际进程为准。

固定假设/检验：1)仅速度撤销可修已见宽六米LEFT18，但两个旧任务代表仍接触，故需允许转向共同修复；2)43专家成功历史动作范围可表达不等于神经函数可拟合或闭环状态覆盖，需43修复损失和自主结果共同判断；3)共享参数可伤其他任务，须完整340原成功ID保持，不用平均loss或总分掩盖损失。本pilot联合改变转向残差/修复数据与平衡损失，不能单独归因某模块或称新算法原创。

架构：T498同一120因果输入、原encoder128、真实64GRUcarry、原2动作head；保留5953参数神经速度残差，再加184->32->1神经转向残差，末层零初始化。两个残差各在原logit加.3*tanh(网络输出)，单Actor直接两动作，无任务ID/侧别/手工时间阈值分支，无地图/规划/路点/运行teacher/router。原input/goal/history/几何/速度执行器/成功半径/时限不改。

只训两残差11906参数，encoder/core/head/Value冻结。一个扩展Adam：原drive四参数640步moment逐位保留，新steer四参数无moment(step0)。原Value与ValueAdam精确不变。新架构初始joint hash不同是新增零分支和optimizer槽位，动作/真实hidden必须与原模型一致；不得误称字节完全同旧拓扑。原T498checkpoint不覆盖。

数据：43修复实际GOAL源（T50042原生成功+42exact与原1实际成功混合恢复全history审计）；340既有该学生自主成功（old233/six53/eight54），共383不同任务ID。监督用真实执行动作，非onpolicy密度。keep用原未裁剪locations且初始接近零损失，避免clipped目标漂移。

固定每步loss=.5repair+.5keep；repair按task4(7)、task6(35)、wide6(1)三家族等权，组内等回合，回合loss=.5mean+.5最差20%决策风险，动作两维等权且±1边界用既有可行集hinge。keep按old233/six53/eight54三组等权、组内等回合，location MSE。固定256live+256shadow、独立审计256历史丢弃Adam；预检两clone各2步共4丢弃Adam另计。学习率继承.001，无seed/epoch/权重/噪声网格，无新PG/Value更新。

零物理预检：640/0Adam槽位映射、真实当前模型动作/隐藏序列初始不变、340keep初始零、两clone真实监督更新精确、序列化读回、模拟中断/拒绝全ActorAdam回退、实际CPUworker取消6命令0native；用4步测算完整512步最坏学习耗时，须估计<1000秒（main学习1200秒）。未通过不启动，错误原样保留；不能以预检学习改善称导航收益。

一次固定候选完整自主383（old275+wide6 54+wide8 54）全部full history保存，缺测/无效/删失单列。研究接受：全部有效、初始340成功ID全部保持、成功>340，允许科学部分进步但不能称过门；否则整Actor和映射的成熟Adam回退。最终联合开发必须383/383全安全。fresh16/reload78未预留未执行，不能用新留出选模型或自动多训。

预算：最多9846771物理、393命令（387native主管线+6预检）；main7200秒（学习最多1200）、audit1200秒、all10800秒且备份/source留1500。学习绝对2026-11-12T15:35:59.651377+08:00，原生绝对2026-11-12T17:20:59.651377+08:00，全部绝对2026-11-12T18:05:59.651377+08:00，相对更早从严，不保证耗满。own180MiB，全383ZIP shape bound160084120B+20MiB模型/metadata/partial预留，完整global1028994547B<1GiB/free40GiB。不得删未备份唯一资料或扩大预算。

CPU14/15总quota100%=1核idle/nice19/CPUWeight1/IOWeight1，CUDA禁用、CPU学习、实际Mesa llvmpipe初始丢弃渲染断言，全子孙cgroup约束。Phase内容/进程/模型/结果研究完全隔离，不检查其资源状态。

driver唯一writer/finalizer；active时禁止并发source/audit/backup或其他native。原生父worker退后唯一独立audit验证所有数组/hidden/history/密度/奖励/源模型SHA/256历史ActorAdam重建，localmanifest后RAM完整远端二进制逐成员SHA（分包），Git只代码/协议/报告/清单/收据。TLS失败保原件PENDING，不称完成、不重新索权。全部结束后报告真实old/six/eight/成功增减与模型状态，不将原成功集合聚合成其他政策门槛。

# T304：原六米固定策略下的三步动作价值pilot

这是独立有界的新机制pilot，非T294/T303续预算；注册、零步预检、源码精确同步后才唯一launch。Stage/Phase隔离。safeT255、T299研究模型、所有失败及两HD视频不变。当前最佳完整6m仍10/54，旧274/275，不能晋升7/8/全图。

## 固定控制与干预

冻结T299（原6mRIGHT已有10成功、LEFT仍27接触）；只取已消费开发LEFT/RIGHT各03/09/18/24八个原始reset/原6m目标。选RIGHT这四例已有成功控制，LEFT四例是匹配编号的失败控制；该条件化选择不当全域成功率或新泛化样本。

每个病例分支点固定b=floor((原父决策数-3)/2)，三步parent action chunk预先来自原始紧凑动作记录。三个分支：同drive，steer分别偏移-.03/0/+.03后按native边界裁剪，全部0.6秒短片段是预先固定的open-loop动作。之后同冻结Actor在各分支真实状态继续反馈；整个回合原目标/时钟/previous-action/舵机等因果120输入保持真实，不瞬移/切目标。

原始起点完整重走，每个分支一次unique+独立精确重生，共24+24；八个center全部12数组/终态积分须精确匹配T299历史，所有分支前b个输入/物理prefix相同，三步后逐状态重算必须是同Actor输出。额外强制动作仅模拟器诊断，不是部署教师或自主导航新成功；deterministic logp0禁止作AR on-policy数据。

## 价值学习与控制

定义Q3为这个三步动作宏加固定T299确定性续行，在原有限时域/终止下的MC值。用已有eta.1成本目标，保留环境原reward，标签为分支点Gbranch-Gcenter，不将终点距离当成功。整个回合没有新query/重标记目标，不冒用未来目标字段或混合行为Q。

每侧03/09/18六个reset×3分支18行训练；每侧24两个reset×3分支6行测试，均已消费开发，非新的导航fresh留出。重生不增加训练行数。比较完全同初始/4609参数的126→32→16→1 Tanh回归头，full action chunk vs将其归一后6动作列置0的blind头。训练-only输入mean/std（std下限1e-3）与target RMS（下限1e-6）；500fullbatch Adam各一头，lr.001/clip5/固定seed，1000新head更新、0Actor更新，不选最佳checkpoint/调网格。

预注册screen全要满足：测试总体MSE≤blind一半；每个测试reset MSE≤blind的102%；每个测试reset至少两个真实advantage差>1e-8的可排序对；所有有效对符号正确；两个最高预测选择都在真实最高值1e-8之内。保留每项MSE/真实差别/GOAL接触超时，不用测试排序当全局Q校准。即使screen全过，本项也不自动接Actor或训练7/8。

## 预算与原始证据

48全部命令、1344000全部2ms物理步、4workers；总2400秒，最后300秒留拟合/验收/归档准备，外部timeout更早截止从严，不延长。own7.25MiB，24full120 state-chain≤192KiB、两head+Adam各≤256KiB、evidence128KiB、最多四个partial replay各192KiB，payload6016KiB+metadata1MiB，buffer384KiB；global≤1GiB/free≥40GiB。实际full worstcase预留启动前核算，不因free很多扩1GiB。

零步预检不构造native、不作Adam：源/模型/输入SHA、动作分支同drive且不同steer、模型盲不变/梯度、全局名及预算。源码提交/远端精确树后唯一launch；active期间不写其他runs或sync_source。所有真实物理/命令/inflight/partial计账；失败/中断头权重和Adam也保留，不重启旧目录。

结束先确认父/4worker/timeout/inhibitor停止；独立audit0native，1000丢弃headAdam重构、所有full120/12数组/积分重生/前后动作/价值/划分/归一和源SHA精确。全部原始数据在RAM归档，远端完整读回逐成员SHA后才称备份完成；再精确源同步。完整old275/6m54/fresh16/78本项未安排，必须写未执行，不能当失败率；6m尚未全门槛，不进7/8。

# T381 原生采样训练教师：一次恢复可行性试验

2026-11-05，用户授权继续。由T381_NATIVE_MPPI_DRAFT落地；当前文件是注册合同，不表示已启动。Phase完全隔离。T379/T380均结束，不重启、不续其优化器或模型网格。safeT255及所有人工/失败证据保持。

目的：检验原生MuJoCo教师能否从T379真实8秒偏离状态提供可执行纠正。T380简化模型筛选失败，所以本轮不使用它预测。教师拥有完整仿真状态与碰撞查询的特权，预测也是仿真物理工作；本轮0Actor、0优化器、0新RL。部署仍为单Actor直接输出动作，无地图、规划器、路点或运行时教师。

## 科学合同

固定T379最终模型、新0.70米/80秒中心，原起点、轮胎/碰撞mask、wheel5/force120、不倒车、.002秒原生步、每hold100步、.20米目标半径均不改。这是新070课程，不代替原0.57米/56秒六米验收。

两次完整pass。各先从原起点重现T379失败，完整数组/hidden/history/积分必须精确历史原件；再原起点重建40hold=8秒前缀。该开发偏离点是依据已知失败事后选择，不是fresh或保证可恢复。保存mjSTATE_INTEGRATION、Python状态、实际previous动作/步数、四帧history与真实Actor隐藏状态；恢复后重绑Witness frame身份，逐位检查native积分/当前120/历史。另执行历史后5hold精确参考再恢复，真实500步/1命令计费。

从同一快照一次性预测33条：恒定当前父mode nominal，以及seed20261105381的32条标准正态创新；drive/steer标准差[.20,.40]，每5hold保持同创新，60hold=12秒，实际命令裁剪[-1,1]。固定一次采样/一次加权，不增加seed、幅度、时域、迭代或改选best sample。原生native.step_checked推进，禁止stock bicycle pose覆盖。

每条到接触/目标/原任务真cap即停，否则12秒horizon结束。cost=1000*CONTACT+10*终端真实目标距离+.01*已执行相邻命令变化平方和；温度1，softmax负cost加权所有完整60hold候选命令，得到一条新序列。早终止候选的未执行计划控制仍是原固定序列；代价只计实际执行前缀，相同规则预先锁定。加权均值无安全保证，必须独立执行而不是以样本安全代替。

原8秒快照后真实执行加权序列至12秒或终止；其后同一冻结Actor闭环到原80秒终止。教师期间仍用实际新观测推进Actor隐藏状态，下一packet含实际教师动作。ownership=1只表示教师执行；Actor proposal/logp单独命名，不能将教师动作伪作人控或on-policy动作。实际无接触GOAL才是混合轨迹的训练候选，接触/超时/未完成分列；即使成功也不是Actor自主六米通过。

第二pass从原起点重复全部预测和混合执行；预测逐hold完整120/history/reward/native积分的SHA与首pass对应，紧凑实际pose/kinematics/count/flags同时逐位检查。混合轨迹逐hold比较完整recordSHA、最终全部数组与native积分。第一差异立即保存单步状态证据并停止，不生成无法保存的后续唯一数据。重复不算独立成功。预测hash是完整状态的一致性证据，不声称已保存预测每一步全部120原数值；其余紧凑原数值完整保存。真实混合轨迹保留完整120、next、history、hidden、proposal、ownership和初末积分。

指定wall–circle窄口的穿越若没有独立几何验证，明确未验，不将到达或绕行宣称穿口；历史GOAL定义不改。原old275/six54/fresh16/reload78均不在本pilot中执行，不进入7/8米。先验证教师，若真恢复再另立数据纠偏学习协议。

## 资源与顺序

每pass最多38命令：父历史1、前缀1、恢复参考1、候选33、加权执行1、Actor续行1（即使前段终止，续行命令记零步）。物理最坏40000+4000+500+33*6000+6000+30000=278500，两pass557000，另1零物理geometry预检。全部上限80命令/600000物理，不因snapshot恢复豁免预测成本。

main1200秒，独立audit300秒/0native，归档300秒，全部1800秒，外层timeout更早从严，不延长。own1MiB/global1GiB/free40GiB。full或partial共512KiB、预测256KiB、prefix4KiB、metadata96KiB、log32KiB、首差异32KiB，合932KiB留92KiB。注册时实算完整预留；预检用不可压缩随机内容模拟最大400hold full/33*60 compact编码，保存尺寸而不保留合成大文件，0物理0Actor训练。所有code/input SHA、source commit/远端精确树及预检PASS后唯一launch。

结束所有原生父链退出后，独立零native审计重算固定噪声、clip、cost、weights、ownership、Actor实际history递推、奖励、全部物理/命令、源/父SHA和两pass证据；不追加控制查询。再RAM归档、远端二进制全成员SHA读回及local复核，Git仅源码协议汇总清单收据，不写大二进制。失败保留raw/partial/error，不重启旧目录，不临时扩搜索预算。

## 一手依据及限度

[MPPI作者论文](https://arxiv.org/pdf/1509.01149)提供采样控制背景；本轮包含命令裁剪、块噪声和自定义cost，称MPPI式有限采样pilot，不声称完全复现其路径积分推导、全局最优或新算法。[GPS](https://proceedings.mlr.press/v28/levine13.html)支持训练期轨迹优化指导策略的路线，但本轮尚未训练学生。[MuJoCo状态API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-setstate)用于原生状态保存恢复，额外环境history/Actor状态仍必须自行同步。本轮未使用PRIEST的投影器或全局路线。

# T302：困难区间自身经验均衡学习与逐块旧任务接受

用户2026-11-02再次授权推进六米，通过后进入七米、八米及全图。Stage/Phase完全隔离，部署仍单Actor120因果输入hidden每决策清零，无地图/规划器/路点/运行时教师/路由。

本项是新数据覆盖+闭环接受机制的有界工程pilot，不是给T299追加500步或换seed网格。从安全T255重新开始，T299不安全候选仅提供已发生RIGHT6m自身成功数据，不能当部署基线。六米最新完整10/54、旧274/275；未达门槛不晋升/不训练7或8。

## 固定数据和学习

五条T263/T267 LEFT5.5m真实安全完整轨迹，十条T299 RIGHT6m真实自主成功（T301全部12数组/积分精确恢复）组成已消费的15轨迹开发数据。LEFT是含相关探索的自身历史行为，RIGHT是确定性自身行为，混合行为仅用于目标条件动作监督，绝非新on-policy RL或HPG/IS估计。完整120及真实原goal历史，不做future goal重标记。5.5m只作为完整训练任务最终goal，绝不在六米部署回合内切换目标。

侧权重各.5，同侧轨迹等权；每轨迹每步采4行（共60），旧8任务每任务4个anchor（32行）。沿用T299连续截断AR条件NLL/logtail/current+previous batch1梯度，原rho=.98/sigma(.035,.03)、oldmodeMSE权100、baseLR3e-6/residualLR1e-4/clip1、Actor容量保持。新数据对应的parent全批side-equal modeMSE/NLL梯度范数比在0物理0优化器时固定，训练不调尺度或权重。

最多10块×50监督live+50shadow=500live500shadow。每块前保存整个Actor+Adam；块末live/shadow逐位相同后真实old20检查，只有全部安全GOAL接受，否则Actor/Adam和shadow全部回退并立即停止后续块。不放宽old20，不尝试同块其他步长/seed/权重。失败proposal及Adam哈希、接受结果、实际调用保存，独立审计完整重构失败块与回退，而非抹去拒绝成本。只检查完全相同policySHA的native cache可供复用，不算第二次独立物理。

父模型先独立执行三个LEFT5.5m初态控制，候选末尾同三例，不能把原本会的能力算新增。MSE/NLL仅诊断，实际old20和bridge成功分列。最终候选无论局部门槛如何，完整old275+原六米54都测；全部安全才16fresh+78独立native重载。缺测/删失/无效单列，不用中心/探索成功替代稳定。

**本项不注册新的导航RL采集/策略梯度更新**；它检验更贴近失败区间的经验能否在保护旧能力时保留学习。监督与本线真实DRL谱系明确分开，后续原六米RL需要独立新的当前policy交互与完整预算。不得将离线成功动作当PPO on-policy。数据与接受机制共同变化，不能单独归因NLL或声称GCSL/PCGrad/保护原创。

## 有限预算与证据

650全部命令、18.2M全部物理步、5400秒main，前3600学习/保护/诊断，末1800完整验收；外部同5400秒TERM/1秒KILL更早从严。own11MiB/global1GiB/free40GiB；640×10KiB紧凑5数组（reward=env-eta*cost精确导出）、2.5MiB最终Actor+empty freshAdam、1.75MiBmetadata，源15完整120只读复用。最坏200块保护（首块拒绝后safe基线重查最多40，亦包含在200上限）+3父bridge+3候选bridge+275旧+54六米+16fresh+78reload+4partial=633≤640。没有新RL完整训练payload。

监督Adam不重复写大断点，保存各块前后/候选/回退全部SHA、原源/15完整数据/固定采样规则，结束独立最多1000丢弃Adam精确重构所有尝试和回退；最终Actor+emptyAdam导出读回精确。预检最多4丢弃Adam，合成2步live2步shadow、真实函数失败块整ActorAdam回退、预优化器中断回退和所有函数全局名/资源/源hash核验，0native。预检通过/代码本地提交/GitHub完整root-tree同步后才唯一launch。

T300只44冷文件有历史全远端读回证明并已无损编码，原229新上传未完成；不把失败starter算备份，源/错误/恢复账本保留。T301数据恢复和备份完成后按实际global容量预留T302，不先跑后补空间。运行中native-exclusive，不写其他runs/源同步。结束全部父/4workers/timeout/inhibitor退出，独立审计全部源/数据SHA、命令物理、块ActorAdam、保护cache和原生验收，再RAM全原始远端二进制/成员SHA读回和源同步。safeT255/用户HD视频/原失败均不动。

只有六米old275+dev54+fresh16+reload78真实全过，才依据用户已给授权准备独立有界七米，再同等门槛后八米，逐步全图；必须先容量/协议/零步合同/精确源码再启动。不得拿新编号延长本项失败预算，不能承诺本轮一定通过或把10/54当阶段完成。

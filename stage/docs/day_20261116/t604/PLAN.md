# T604 成功恢复策略初始化匹配试验

固定T60342条历史/原rawlocation目标/mask/同32GRU/128steps/lr3e-4/norm1/tail损失，仅Actor初始化改为T598原成功教师；freshAdam。不是新RL/critic/原创算法。保留T550，不晋升T598为全场地安全策略。

三处固定学生接管+3exact均必须safeGOAL，再15哨兵（原八+已知失守去重）；哨兵中原T598成功项不得丢，之后补34及4keyexact成完整49。完整原T59831成功保持、修回至少1个相对T550失守ID、三恢复全过才保留研究中间候选，否则整ActorAdam回退。最终原39+三恢复/完整407/未见/独立模型原生读回门不降低。当前恢复只是同一已消费field场景三个前缀，cap128.224s，不称原80s六米通过。

canonical保存完整120history/双32carry，另外两接管与49只紧凑保存，liveRAM全history检查/逐数组积分exact；后两者不宣称独立完整离线history重建。全部partial/错误保留，不加seed/网格/轮数，不重启旧作业。单Actor120因果输入真实carry，无运行教师/地图/规划器/路点/路由。Phase完全隔离。

预算拟独立2879616物理63命令，128live128shadow128历史审计+4丢弃预检；own5.125MiB，full2MiB/model384KiB/58compact12KiB/metadata2MiB最坏5300224B<5373952B。global1GiB/free40GiB/CPU14,15总200低优先GPUoff不变。main1200learn600audit300backup180source120，准入1830秒；21:10停学21:20原生21:30全停，更早相对截止从严。此前21:00授权不覆盖延长，正式新预算/截止确认前0学习0仿真，仅准备。

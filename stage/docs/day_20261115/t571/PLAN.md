# T571：自身成功状态复习＋已验证恢复的持续学习pilot

用户2026-11-15明确允许CPU多核/可选GPU、现在开始加速；新确认今晚20:30全部停止。学习19:00停止、原生20:00停止，main3600秒/前1800学习/audit600/backup420/source180，从相对/绝对更早截止。日19M全部物理/600全部命令/12300主管线秒不扩。T570入口零计算错误保留；T570R 24次完整实际history前反向并发hash相同，但小网络并发吞吐仅串行.917倍，因此学习保持单进程，原生使用两独立worker/CPU14,15总200%/Nice19；GPU关闭，不占共享GPU。

## 可否定假设与方法边界

T569同T562旧8保持、field13/24，相对T550丢sector0_offset2和sector1_offset1。旧平均误差、旧中心和有限教师轨迹上的函数保持均不代表整个闭环保持。假设：在当前学生真实会到达的成功状态上保持其原始连续策略位置函数，并加入经实际验证成功的恢复纠正，能在不丢原21项的情况下恢复两个失守任务。

CLEAR支持replay与行为保持，但本轮只有监督rehearsal/恢复BC，不含V-trace/Q/PPO，不称完整CLEAR或新RL算法。参考CONTINUAL_LEARNING_PRIMARY_REVIEW中的五篇原始论文。冻结T562完整encoder/GRU/原平滑头，新增zero-output184→32tanh→2平滑残差5986参数；部署无任务ID/教师/地图/路由。冻结旧部件本身不保证组合函数不变。真实GRUcarry全history重算，不把逐帧zero-init或旧teacher hidden当学生隐状态。

## 固定数据与唯一提案

1. 同原T562原起点/goal/clock/动作/geometry，原21成功任务各原生精确再生一次（两worker）。全部实际数组hash/积分必须与T569同模型旧结果一致，这21是物理重复，不充新独立样本。主线实际fullhistory/奖励/密度检查；落盘只保存完整序列120维states、原locations与actions及其SHA，不保存全部native/history/hidden大数组。原始compact与所有全数组hash单独保留。独立audit只重核持久化子集和重算学习，不伪称全原生历史重建。
2. 恢复标签固定两条：T568 sector0真实39秒前缀后教师成功的接管后动作；T550已有sector1_offset1整条成功轨迹。后者是旧成功示范，尚非当前学生偏离状态的恢复验证，不能冒称两条都新采集成功恢复。不增加teacher查询/种子/噪声/几何/时限网格。
3. 用冻结body对完整history生成因果特征；因body不更新，可缓存RAM特征，不伪造新BPTT。旧21每任务等权locMSE，总权重1；两条修复每任务等权，均值与20%尾部动作损失各半，总权重1。饱和teacher动作用原已有feasible-set hinge，不把教师当高斯onpolicy。固定lr1e-4/512Adam，独立shadow512逐步结果/最终ActorAdam一致；独立audit再次512历史丢弃重构。0新PPO/Critic/RL。
4. 唯一候选完整32纯学生自主，再32全数组/积分原生精确重生。参考集合必须包含T550原23项（当前21全部＋两个失守任务），逐ID保留，不用总分抵销。未达则整Actor/Adam回退至本轮初始研究副本，失败候选原始保存；T550仍是已保留模型，不因本轮32通过而替换正式部署资格。

## 预算与工程合同

两代worker启动/停止共8命令，21采集＋32候选＋32exact共85请求，总93命令。物理上界为21stepcaps＋2×32stepcaps=3,968,757；所有worker原生计数计入。512live/512shadow/512audit/4预检丢弃另列，不用总更新数声称都保留。完整407/六八54/fresh16/reload78未安排=null。

own12.5MiB：21序列按真实stepcap合计原始子集504字节/决策＋每文件64KiB头上界；两个partial各1280KiB含n+1history正确schema；memory原子384KiB、ActorAdam1.5MiB、54compact各12KiB、metadata/log/manifest1MiB。精确外推以PREFLIGHT为准。已完整备份的结束缓存在COLD_CACHE逐成员fresh remote读回后可恢复移出；保留原模型字节到GitHub及恢复账本，不删除唯一证据、当前T562/T550/T568和训练输入不动。global≤1GiB/free≥40GiB，RAM特征临时≤约1GiB，服务MemoryMax5GiB。

新Actor上线前零原生预检：初始函数精确等T562、真实序列/逐步一致、固定body、梯度/密度/饱和动作、4合成丢弃Adam的shadow与成熟Adam/拒绝中断回退、源与模型SHA/容量；预检非导航成功。源码提交、GitHub精确根树、注册唯一launch后方运行，独立driver单writer负责结束audit/全raw RAM远端SHA/source。无网不影响科学计算，但远端失败标pending，不虚称备份完成。

此项为一个有界工程pilot，变化包含新增残差、当前学生状态复习及恢复数据；不能单独归因CLEAR、记忆或新层。若失败，保留逐例损失/到达/接触/超时与失败模型，用下一轮新假设推进，不追加512步或换seed续预算。

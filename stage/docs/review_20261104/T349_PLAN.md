# T349：统一真实成功示范目标，检验六米自主闭环

2026-11-04 Stage only。用户明确“现在开始跑六米的，跑出来给我”，授权本轮有界实验；Phase聊天/模型/研究完全隔离。T348全部结束、独立audit PASS、85原件远端逐成员SHA已过，不重启旧目录。safeT255原件保留。

## 新假设与固定改动

T348两臂wide5均0/5，memory旧8/8但old20仅10/20。T347真实同起态314决策成功，与T342成功策略在相同初始120/零hidden给出不同动作，T348预测折中；尚非唯一因果证明。

本轮固定**只用T347的真实完整成功序列作为新技能监督**，移除T328/T331/T340/T342四组新技能来源。旧211回合、旧八任务均衡采样、权重old/new各.5、每臂600次Adam、lr.001、初始化seed20261104348、6306可训练参数、GRU16/120输入、数值路径都与T348相同。两臂仍reset/memory，分别逐决策清零与真实跨0.2秒决策携带；从安全基础重新开始，不续失败模型。每次完整当前参数因果BPTT，从真实回合起点零hidden，无未来信息或陈旧缓存。

来源减少也使T347曝光从120增到600次，必须如实报告这一耦合；不能把结果单独归因“去掉冲突”。单一来源可能过拟合、缩小覆盖，也不能保证保持旧能力。新增RL/Q/原生训练采集0；本阶段为示范预训练，若获取成功再另注册真正序列RL与门槛，不把监督Adam称新RL。

部署仍单Actor局部观测→真实动作，无地图/规划器/路点/运行时教师或路由。原wheel5/force120不改，宽1.17m/80秒课程与原窄56秒分列；速度/倒车是后续新plant与动作域，不能混改。

## 文献检查

2026-11-04再核对[IBC官方论文页](https://proceedings.mlr.press/v164/florence22a.html)摘要、[作者项目页](https://implicitbc.github.io/)多值函数示例，及[Ni等ICML2022](https://proceedings.mlr.press/v162/ni22a.html)摘要/此前已读作者序列源码。这里先检验一致目标数据，未实现IBC能量动作优化、Diffusion或完整recurrent SAC/R2D2，不复制作者代码、不宣称原创。已有T329单人控BC失败等反证仍有效；T349具有新实际120序列与匹配循环/reset条件，只能检验当前资料/600步条件，不保证成功。

## 有界执行与加速合同

初始共同安全策略wide5一次；每臂final old20/wide5/原窄2，再27独立模型读回原生全数组/hidden/history/积分精确。7预检geometry+5initial+2*(27+27)=120全部命令、最坏4.52M所有物理，含所有重复。仅每臂matched wide_dev1完整384KiB，其余57primary紧凑4KiB；54读回RAM全数组对比保留hash，不增加独立样本。完整275/原six54/fresh16/reload78本pilot不执行/null，不能晋升原六米稳定、7/8。

1200live监督+1200shadow，独立audit另1200历史丢弃，预检8丢弃另列。整块更新前不可变输入hash，Actor/成熟Adam加载deepcopy、拒绝和中断整块回退，精确shadow后才保留。序列/step<1e-12、实际spawn输出精确、有效hidden/因果前缀与7真实geometry0native合同复用同实现但本轮实际执行预检，不拿旧PASS当本轮执行。

主任务1800秒，末600留验收；audit600，全工作3600，以launch绝对截止和外部timeout1800秒TERM/1秒KILL更早从严。两worker CPU1/2、学习CPU0，不增加样本/步数来填时间。own3.25MiB：2full×384KiB+2partial×384KiB+2model×256KiB+57compact×4KiB+metadata768KiB+logs128KiB=3248128B<3407872B。全局newruns1GiB/free40GiB；启动前约1070204717B，完整最坏约1073612589B，仅约129KiB全局余量，不扩大或删除原件。

加速仅预算检查工程路径：开始精确遍历并冻结其他Stage runs的路径/字节/mtime清单，完整预留own上界；flock持有单writer锁，每≥.5秒精确计量当前目录、检查free，最多10秒完整复核其他目录清单，任何外部改写失败关闭；结束再精确全局核账。协议要求运行中无其他Stage写入，不能据周期检查声称可防任意外部进程瞬时写盘。预检只在新目录生成可丢弃小字节，测试完整预留、超额拒绝、外部变更拒绝和全局最坏拒绝，0native/Adam；不对唯一证据做测试写入。

代码/输入SHA与协议锁定、干净本地提交/远端精确根树后唯一launch。active单writer不改协议源、不外写其他runs或source同步。结束所有父worker/timeout/inhibitor退后唯一独立audit（按紧凑优化器事件解码核账、两臂600历史Adam/最终模型、2full轨迹及54读回hash、全部命令物理）。全部原件RAM归档、远端完整二进制逐成员SHA，最后精确source同步。失败不追加epoch/seed/权重或换编号续该预算，保留失败与模型。

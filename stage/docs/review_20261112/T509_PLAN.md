# T509：从场地11/24继续，两项近目标失败的联合成功学习

2026-11-12 用户要求下一步开始。T508完整冻结验收结束并独立audit PASS，24全数组/积分exact通过、36文件全远端逐成员SHA、父worker/driver退出。同一接受模型old8=8/8、field11/24（9contact、4timeout、0无效删失），原field7保持、新增4；T507断电证据及T505/T501最佳单独保留。Stage/Phase全部隔离。

原生记录分型：4超时中sector1_offset1最近决策余0.23122254m、sector0_offset2余0.48335168m后驶离；其余sector2/4_offset3还余7.19/6.85m且慢。另9项接触在其他方向，不与近目标错过混称统一速度不足。按已见失败最近决策余距升序固定选择这两项，不拿端点最近当连续native最小，不放宽原0.2m成功门/时限/速度/场地。文献及反证见research_20261112_t509/PRIMARY_SOURCES.md：成功自身经验整合已有SIL/PolicyDistillation等近邻；本轮无Value/优势正部，不是原SIL/AWR/DAgger或原创算法，固定距离排序不是ALP-GMM/GoalGAN。此次改变训练任务/数据/联合损失等是工程pilot，非单模块因果创新。

初始为T508接受的T507block0 Actor完整成熟Adam drive1025/steer385，监督lr1e-4均保留；11906现有双神经残差参数可训、骨干/head/Value与ValueAdam冻结。120因果输入/64GRU真carry、单Actor直接两动作，无运行时教师/规划器/地图/路点/任务router，无额外网络容量或传感器。原成功/碰撞/时间/reward/gamma/action语义保持。

先固定原起终点每任务4条真实iid rawsigma[.2,.2]、固定SeedSequence8采集+8原生exact，完整120/history/hidden/raw/atom/logp/innovation/奖励与积分保存。两任务各至少一条真正无接触原生GOAL且未越±7.5m资格边界才允许一次联合监督学习；否则0Actor/Value更新停止，不换seed/噪声/追加回合。所有失败原始保留。八探索不是八独立训练seed，不把噪声成功当确定性验收。

联合数据：所有8条采集中的qualifiedGOAL，按两任务等权/任务内成功回合等权；这是成功筛选行为监督，采集时Actor固定、更新后明确旧行为数据，不当新PG。执行float32动作沿裁剪边界可行集hinge（不是伪造唯一raw或当前logp），repair每回合半完整物理时间mean/半离最终目标<=1m物理时间mean；1m只是离线权重，不是waypoint/部署输入/reward/成功门。keep来自当前接受模型实际old8+field11共19条成功history，原unclippedlocations，old8/field11两家族等权、家族内等回合，半物理mean/半upper20状态风险。总半repair半keep、sigma归一两动作等权。冻结骨干可缓存完整真实GRU序列特征，但不能每行hidden清零，初始keep数值零须预检核验。

最多2块，每块16live+16独立shadow监督Adam，0新PG/ValueAdam。每块实际21任务（old8+field11+2新）原起点自主检查并21全数组/积分原生exact；初始19项逐ID保护，新成功加入保护集。所有保护ID不丢、repair目标确实降低、无无效/删失才临时保留，首次拒绝整块ActorAdam回退并立即止学，不从同状态/数据重复优化。21全部过则提前止学。有至少一新确定性成功且19原成功保持的候选才测剩余11field，再完整24field exact；guard13field同policy结果复用，不假称独立重新采13。最终old8=8、原field11所有ID保持、qualifiedfield>11、contact<=9、0无效删失才接受；否则整实验ActorAdam回退，接受11/24最佳始终独立不受影响。若守门失败或无新确定性成功，fullfield24=null，不当0%失败率。

最多135物理回合=8train8exact+2x21guard+2x21exact+11剩余24exact；主139命令含4启动退出，预检6零物理共145。最多61full文件（8+42+11），所有exact物理计账、数组不重复存第二份。最多32live32shadow、独立audit最多32丢弃历史重构，预检4合成clone另计。own60MiB/global1GiB/free40GiB、最坏shape+14MiB模型metadatapartial预留，实际完整余量先算，不删唯一证据/不扩额度。

主3600秒、学习相对2400、audit600、归档source1200预留，all6600。绝对学习注册now+3600/native+4500/all+6600，与相对更早从严，不能延长。CPU14/15总quota1核idle/nice19/CPU-IOweight1，GPU禁用且实际Mesa llvmpipe；不查看/修改Phase进程或设置。

预检真实19keep完整历史及2合成噪声fixture组，2clone各2步合计4丢弃合成Adam、0native/Value：验证真实动作density、归一/边界loss、keep初始零、成熟Adam/固定骨干/克隆一致/全回退/序列化、两真实worker取消。合成fixture不算导航/成功数据。正式代码提交/完整远端根树精确同步后唯一systemd启动，driver唯一writer不外写其他runs、不并发audit/备份/source或改协议。科学全部进程消失后唯一零native独立历史Adam/成功筛选/原始history密度奖励/逐ID接受/回退/计账核验，RAM全原始远端逐成员SHA，再报告源码同步。失败保rawpartial、不restart/扫grid/换seed。

旧275/six54/eight54/fresh16/78本轮未安排null，不能将19保持当旧379全部保持；field24仍已见开发集非全场地泛化。本轮失败只能否定这个有界配置，不代表任务不可达，也不保证两个新任务或全部24均本轮成功。

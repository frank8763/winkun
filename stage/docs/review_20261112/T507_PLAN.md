# T507：十二条新成功经验与十五项既有能力的联合整合

2026-11-12 用户要求下一步开始。T506已结束：12新自身探索均GOAL且12exact通过，16live16shadow真实PG后守门13/15（dev_task6_13接触，sector2_offset1超时），候选已按协议完整回退；全field24未执行null而非0%，原T505接受模型未变。独立audit PASS，52原文件全远端逐成员SHA，所有进程退出。本轮不是追加T506训练/噪声/轮数。

从接受T505(old8=8/8，field7/24)重新初始化，不从T506失败候选继续。完整原Actor与成熟Adam drive993/steer353保留，监督学习率1e-4与接受T505一致，11906现有双残差参数可训。没有本轮新branch/网络容量变化；骨干/原head/Value及ValueAdam冻结，Value0。真实120因果输入与64GRUcarry，部署单Actor直接drive/steer，无地图/规划器/路点/教师/任务router；原物理/几何/速度上限/奖励/时间/0.2m成功门不变。Stage/Phase全隔离。

机制证据：新三任务sector7off1/sector7off2/sector0off1在同T505初始Actor的真实iid探索全部4/4成功；已存在完整可执行成功覆盖，不需再先搜更多噪声。T506PG改变模型后旧/新已有能力损伤，不能只以新探索12/12冒充确定性成功。下一步检验联合成功监督是否能将这批已存真实成功转入同一确定性Actor，并且同时保持已有15项闭环成功。12成功的Actorpolicyhash必须等于本轮接受初始Actor，Adamlr等元状态差异明确，不把旧行为动作当本轮onpolicy。

文献近邻、读取范围及反证见research_20261112_t507/PRIMARY_SOURCES.md；Policy Distillation、SIL、DAgger等不提供少量随机成功必转确定性成功的保证，标签多模态/去噪均值/学生诱导分布均可能失败。本轮是成功经验联合监督整合，0新策略梯度/0新训练物理采集，不是原SIL/AWR/DAgger复现，不称创新或单模块因果收益。与T506改变目标/保持比例/更新预算/学习率，工程pilot而非受控算法优越性证明。

固定两部分：repair为T506原12条完整原生无接触GOAL/未越场地边界的执行float32动作（不平滑/不拼接/不截断history，源NPZ/全部数组/创新/密度/积分保持）；三任务等权、任务内4回合等权。keep为T505同接受模型实际原生old8+field7的15条成功history，目标原未裁剪locations；old8与field7两家族等权、家族内回合等权。合计半repair/半keep，两动作sigma归一等权，repair每回合半完整真实native持续时间加权mean/半最终目标距离<=1m片段时间加权mean，keep每回合半物理时间mean/半最坏20%状态风险。1m仅为训练离线权重，不是路点、部署输入、奖励或成功半径变化。repair沿既有动作裁剪边界可行集hinge，不能凭空把执行±1映射成某个唯一raw action或伪造当前策略logp；keep精确匹配原unclippedlocation。真实冻结GRU完整sequence特征允许缓存，逐数组密度/history验真，不能每行hidden清零。

最多3固定块各32live+32独立shadow监督Adam；每块导出唯一候选，实际old8+field7+3新任务18项原起点自主无教师执行，再18全数组/积分原生exact。初始保护15成功IDs，新通过任务加入保护；保护成功不丢、无无效删失、repair代理下降才临时保留该块。首次拒绝整块ActorAdam回退并立即止学，拒绝后不从同状态重复相同确定性优化。若18全safe成功即停止进一步学习，避免无意义继续改动。

只在已有15保持且至少1个新任务确定性安全成功的已保留候选上执行剩余14field任务，再全24field exact。同一候选guard已含old8+field10，因此final24复用10个guard结果，不能假称独立又测试10次或用不同policy拼成功集合。最终保留要求old8=8、原field7全部ID保持、qualifiedfield>7、contact<=10且0无效删失；未过则整实验ActorAdam回退到接受T505，所有拒绝模型/raw/失败保留。若无新任务通过，field24不执行null，不当0%失败率。

最大物理回合3x18评估+3x18exact+14剩余+24exact=146；主150命令（含4启动退出）+预检6零物理共156。最多68full数组文件（54+14），exact原生计费但不存冗余第二份数组，物理不冒充独立新样本。最多96live/96shadow/96独立丢弃历史重构、预检4另计；三个块为同一次训练非3独立seed。详细physics/shape/capacity以T507_BUDGET注册实算。own64MiB/global1GiB/free40GiB，不删除唯一证据/扩大额度；本轮活动时不给其他runs写入。

主4200秒、学习相对2700、audit900、归档source预留1500、all7200；绝对学习注册now+4200/native+5100/all+7200，与相对更早从严，不延长。CPU14/15合计quota100%=1核、idle/nice19/CPU-IO weight1、GPU禁用且实际Mesa llvmpipe检查，不查看或调整Phase进程。

正式预检两clone各2步共4丢弃历史监督Adam/0native/Value，验证12成功sourceActor/完整真history/密度/三任务分组与执行边界loss、15keep初始数值零、成熟Adam/序列化/克隆一致/全回退、18guard ID负例及两真实worker取消。预检不等于新导航成功。代码本地提交及GitHub完整根树精确同步后唯一systemd启动，driver唯一writer。科学全部父worker退出后独立零native全历史Adam逐块重构、数据原始密度/history/奖励/接受/回退/物理命令计账；RAM全部原始远端二进制逐成员SHA，再报告/源码树同步。失败保留partial，不restart旧目录/加seed/扫参数。

本轮不重新验完整old275/six54/eight54/fresh16/78，旧八中心不等于全部旧任务；场地24是受选择开发集，不是未见泛化或全场地稳定。通过后才继续补其余碰撞/慢速/其他方向案例，不能保证一轮全部跑通。

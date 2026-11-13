# T515：真实恢复后缀→单Actor最终动作层

Stage only，Phase完全隔离。用户11月13日授权下一轮开始。T514已全原生GOAL44.838s、无接触/越界、1exact、独立audit及22原件完整远端SHA。原T507接受模型old8/field11保留。当前只做一次有界示范学习，不冒称新增深度强化学习或教师即学生。

注册前只读范围审计：教师接管59决策的drive与steer目标全部落在固定原head、±.3logit残差可输出范围之外。例如首标签[.414447,1]，原残差可行drive[.813795,1.100374]、steer[-.325280,.402247]。这是同一固定完整history的逐点输出限制，不证明其他轨迹/策略不可导航。T515因此只解冻现有最终Linear64→2共130参数，结构不变，编码器/GRU及原11906残差冻结；原残差成熟Adam1025/385的8个状态全保留且不更新，另加新head Adam组lr.001、初始0步。Value/ValueAdam冻结。全部参数函数初始等于接受模型，新增优化器group使joint hash不同，语义不可混同。

固定一条T514完整真实history，缓存全序列真实GRUcarry特征；只ownership=1的59个恢复决策用实际teacher_actions监督，原前166决策使用学生原unclipped locations保持，绝不把学生前缀标教师。19个接受自身成功完整history做原locations保持(old8/field11两家族等权)。全局loss=.5恢复+.1同任务前缀+.4旧19保持，各误差sigma归一、半物理时间mean半最差20%决策mean，teacher边界用原feasible action hinge，非伪造teacher logp。原学生proposal密度仅核验，不将教师动作作on-policy PPO。

只一固定块128live+128独立shadow headAdam，audit128丢弃历史精确重构；预检2+2=4丢弃步。无新Value/PG/数据探索/超参网格，不重启失败目录。仅监督目标可表达范围解除不保证一个线性层能全history兼容拟合，loss下降也不代表闭环成功。

同一候选从原起点无教师执行旧8+原field11+目标sector1off1共20项，再20全数组/积分exact；保护原19逐ID安全GOAL/0无效删失且恢复loss下降才临时接受。首拒绝整Actor+所有Adam回退并停止。目标未自主GOAL不跑完整field；目标成功且原19全保，才剩余12field与完整24exact（guard已有12field结果为同policy复用，非独立新重复）。最终old8=8/8、原field11IDs全保、fieldqualified>11、contact≤9、0无效删失才接受研究候选，否则整轮回退，原最佳独立不变。全old275/six54/eight54/fresh16/78未执行null。

预算全部3,735,228物理/80main+6pref=86命令（40guard/exact+12剩余+24exact+4启动退出），main1800秒/学600/audit600/backup1200，全截止prepare+5400秒，原生prepare+2700学prepare+1200。own30MiB/global1GiB/free40，完整32条原始fullshape23896848B+6MiB模型/metadata/partial须小于own；不足不启动、不删除唯一记录/放宽额度。CPU14/15总1核idle/GPU禁用，不读改Phase进程。

预检0native，真实完整teacher标签/前缀mask、初始19位置精确、decoder梯度、原模型参数/残差Adam冻结、live-shadow/拒绝与中断整状态回退、保存重载/实际worker取消全部过；源码提交/远端根树精确后唯一launch。结束native父worker全退后独立核验全部源/模型/数组/真实history/动作密度/奖励、128历史Adam及接受回退。全部原始RAM远端二进制逐成员SHA，报告与源码精确同步。运行中一个writer，不改协议源/并发audit或备份。

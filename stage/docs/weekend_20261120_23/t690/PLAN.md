# T690：只改轨迹score范数加权的留一法基线

准备中，无科学run/登记/launch。T689固定几何求解的预登记联合条件失败已停止，不改CG/damping/容差救。下一实际学习只改终态PG的peer baseline估计器，回到原T678 executed原11906adapter范围，骨干/GRU全冻结，单120/GRU64不变。

同T501sigma.05 initial、T672全407/1724完整原生历史、任务和轨迹等权、终态Y=safeGOAL−collision、ratio clip.2、score_scale512、原387executed hinge动作参考/.2²/weight10、freshAdam lr3e-5/eps1e-8/globalclip1，仅一次live+一次shadow+一次independent Actor，0Value。T678固定executed原候选作为历史控制，不重跑其已消费更新/原生，不恢复旧候选或追加同配置步骤。

对同task完整轨迹i，s_i=∇11906[Σ_t log clippedGaussian(raw/atom/history;theta0)/512]。新b_i=Σ_{j≠i}Y_j||s_j||²/Σ_{j≠i}||s_j||²，A_i=Y_i−b_i。精确零peer denominator时才用普通peermean；不加可调epsilon/指数。b_i和s_j全在theta0计算后detach/freeze，不通过baseline/score权重反传，不含自身Y_i或score_i。原始withinID创新seed独立性须核实，跨任务相关不当新增独立样本。所有同Y组A=0与原定义完全一致，允许跳过无关score测量，但不把无信号当保护。

人口最优常数baseline的score范数加权原则只支持方差控制动机；这里是独立peer上的小样本plug-in LOO，不保证本batch/总体降方差，不修复随机目标与确定性验收的区别，也不构成旧能力保持机制已证。有限PPO surrogate随baseline改变；只在行为点且满足score恒等/独立peer等假设下期望第一梯度目标相同，不宣称整个clipped PPO函数恒等。score使用原mixed clipGaussian概率模型，不混T684rawlatent/AR。

数学/实现合同：等score权重退化原LOO、ownY/score排除、permutation equivariance、常数reward平移不变、zero denominator fallback、constantY零adv。真实全407梯度须与逐trajectory frozen A_i·s_i及原参考gradient独立求和误差<1e-10。固定dev6_00与全387参考的11906方向中心差分eps1e-6 abs1e-8+rel1e-4，±epsilon时baseline保持theta0不重算；全state恢复。全部baseline/norm/adv/labels/fallback与逐任务正负/终态、gradient hash/cos/相对差照报。贡献样本离散度因peer相关不是无偏方差估计，只描述不筛选学习或改权重。

真实原Adam首步解析位移及实际位移误差<1e-12、同68任务均值latentKL max≤.01的原数值门保留，不匹配新步长或norm救。实际更新与shadow必须逐位相同，模型导出/hash/loader真实验证后固定原55任务full+exact、原32旧全保且至少1新才补余352形成同candidate407；最终原384逐ID全保+新增+完整407+独立audit才保留，联合387另列。失败不增seed/sample/steps、baseline指数/权重或改reward，不用高ESS/低KL/经验方差/审计PASS叫研究成功。

最坏13,625,321物理/468cmd（407+55+6），Actor live/shadow/independent各1、Value0、额外optimizer预检0；5700main+600audit/root7200秒≤2h、学习1200s、完整407留3500s。384MiB child/416MiB root/global10GiB/free40GiB、400%Nice19、CPU8/10/12 native14学习审计、4GiB无swap/CUDA关闭。实际全数据入口0native0optimizer预检/一手审查/最坏ledger RESERVED/GitHub精确树后唯一launch，健康只读监督，无并行audit/backup/source或外写runs。原T292状态baseline虽平均改善却局部噪声恶化、T519timeLOO与T678/T686代理误判是必须保留的反证；不称原创或无偏性实证已证。

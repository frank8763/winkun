# T691：在全部真实更新后旧历史上保持旧策略局部函数

准备中，无科学登记或launch。T690已COMPLETE/auditPASS但丢old00/03/14，停止peerbaseline及同T672梯度补丁。T691从保留T501重新初始化，不恢复T690失败候选；T690只是预先固定的训练历史来源。两臂只改变旧任务保持监督所见历史分布，均不做PG/Q/Value/新reward/teacher成功标签，无runtime teacher/router/planner/地图/waypoint。

原T501单120/GRU64、冻结原body/GRU/head、11906adapters。两臂均64次fullbatch freshAdam lr1e-4 eps1e-8 clipnorm1/old总权重10/new总权重1/.2² actionhinge，64live64shadow64independent各臂，不因中间loss或验收变动追加/早选候选。原384 nominal own-function目标+同6条T681 source00_minus原起点deterministic成功的实际应用动作。6项field0_1/0_2/1_1/2_1/5_1/7_1，全数纳入，不从6挑更好；0_2/2_1/5_1替换原T653目标且两臂同替换，按ID去重，384old+6acquisition=390unique。6均相对T501新，只有3项相对387新，不宣称6条全是首次数据；原联合387另外验收，不把训练源成功拼成部署模型。

nominal臂：384旧ID各1/384时间均值的原nominal历史损失。induced臂：对T690固定55screen中属于原384的全部32旧ID，每ID替换为.5 nominal历史均值+.5该ID完整T690实际历史均值；其余352 unchanged。各ID总体权重仍1/384，oldgroup总权重10，增加实际历史条数是主要变量，不增加32任务权重。全部29成功+3碰撞诱导旧历史逐decision纳入，不选T668两容易恢复、无新时窗/接管/seed查询。

旧标签定义为T501在每条实际外部states/past executedactions历史上的自身GRU从reset重建后clip(mu)，float64、量化前局部函数。它不是T690执行动作、不是GOAL或可靠恢复专家标签，不假装T501可从每个失败状态恢复。采用float64目标保证theta0旧loss/gradient精确0；此与真实float32执行的量化差明确分开。T681新学习目标才是已审计成功实际float32动作（hinge处理饱和边界），必须逐步由固定源模型自己的sequence重构locations/commands完全通过。所有body固定时可缓存完整history核验后的features；不能把其他Actor hidden传给学生。

64步过程中新6监督使参数偏离后，induced局部旧函数保持才提供不同恢复梯度；初始两臂gradient相同属预期，不能因此取消对照，也不能宣称已找到容量或遗忘唯一根因。真实全390+32入口初始函数/梯度/方向差分与实际首Adam公式临时表示后的不同恢复梯度都须验证，0native0optimizer。预检不是64step训练重演，不完整提前学习。全部标签SHA/hash、weights、source、任务与carry/array均锁定。

两个固定候选训练/流程在任何新native之前共同预注册源码锁定；按固定顺序nominal再induced独立child，每child fit→native→audit，无前臂结果驱动后臂任何改变。本助手不在两臂完成前解读中间效果或调参。每臂64步最终模型导出，live-shadow exact和完整Actor/Adam回退；same68 old任务面板max meanlatentKL<=.01原数值门不变，失败不缩lr/步数救。通过才原55full55exact/32旧保护+23outside，原32全保且新>=1才余352补407；原384逐ID全保+新增+完整407+independentaudit才留，联合387另列。任一臂loss/随机/低KL/审计PASS都非研究成功。无论结果停止这次固定设计，不增queries/step/权重。

最坏每臂13,625,321物理468cmd，合计27,250,642物理936cmd；Actor live/shadow/independent各128，Value0，额外optimizer预检0。每child5700main+900audit/两臂main11400+audit1800/root14400s4h有限接力，每child≤2h，学习1200s、条件完整407留3500s；root800MiB/child384MiB/global10GiB/free40GiB/4GiB无swap/CPU8/10/12 native14学习audit/400%Nice19/CUDAoff。原T690source已全退结算，T665第五8MiB批已完整远端26成员SHA恢复索引/deleted0。先主审/真实0native0optimizer预检/独立RESERVED/GitHub精确树后唯一launch，健康时不改锁定源/外写runs/并行source/audit/backup，不延原deadline。

DAgger/经验聚合不是创新，T603/604/T609-618/T631/632/T660实际BC失败以及T6683-4成功恢复门FAIL全部保持；本项不同的是全部32外部历史上的旧局部函数保持，而非选择成功后缀、T690初始化或证明专家可靠。固定拒绝候选占用不是迭代DAgger分布保证，6条源由成功发现筛选，不当独立泛化。详T691_OCCUPANCY_PRIMARY_REVIEW。

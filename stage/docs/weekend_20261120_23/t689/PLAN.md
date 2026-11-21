# T689：旧成功历史函数度量中的一次自然方向提议

准备中，尚无run、登记或launch。本项不恢复T687固定全参数Adam或T688参数norm匹配SGD。回到原T678 executed的11906 adapters/冻结120输入GRU64骨干、同T501sigma.05初始、同T672407任务1724历史/同终态LOO PG ratio.2/scale512/同387executed hinge动作参考/.2²/weight10。只改变更新映射为旧函数度量的自然方向；旧T678已消费的Adam作为原控制引用，不重做其更新或原生。

F=384原T501成功ID等权、每条历史时间等权的Jmuᵀdiag(.05^-2)Jmu。原body/GRU完全冻结，因此真实完整q.sequence历史核验后的特征可分块复用，只对两个adapter求导。F是此经验参考分布的latentGaussian KL Hessian，不是clip命令边缘Fisher、onpolicy占用度量、环境导数或闭环保持证书；与三个T653获取目标分开，后者仍在原联合loss中不在保护metric中。

固定damping .1I、最多10次CG、相对residual≤.1、提前停≤1e-8，每次曲率须正/finite、g·d<0、dFd>0。不得因门失败追加CG或调damping。全384真实矩阵向量积的固定方向数值差分eps1e-6须过abs1e-6+rel1e-4及对称误差<1e-9/PSD，全部state恢复。报告残差、damping能量份额、曲率和所有数值。

方向只按原T678executed检查点相对initial在同384参考上的真实mean latentKL做归一化：控制事先固定，不选更好臂。先二次近似scale0=sqrt(2target/(dFd))，唯一固定根区间[0,2scale0]；端点不能包target即拒绝、不增区间。先检查固定8个等间隔knot的KL递增/正方向导数，取首个观测跨越target的相邻knot区间；此后每中点必须在端点KL范围内且方向导数正，否则拒绝。只声称有限采点branch检验，不证明连续区间单调或全局唯一根。区间二分最多32次，abs1e-12或relative.001以内才允许提议，明示固定branch/全部临时表示probe数。只匹配mean function变化，不匹配perID/最大KL或闭环表现；不是按新成功/旧验收选择step，也不改.01原数值门救。所有临时probe0optimizer、restore后仅一次真实NaturalStep更新及一影子/一独立历史重建，实际Actor各1，不调用Adam，Value0。全部探针/拒绝/影子/独立墙钟计入science/audit，保持完整状态回退。

真实loss gradient需与T673/T677原adapter11906数组误差<1e-10；原68ID随机历史面板max taskmean latentKL≤.01仍是数值门，不作为研究成功门。通过后固定原T67855任务全部full+exact（全部已知失败、原32旧保护、全部23原未成功）；当前T687未产生新失败病例。此screen原32旧全部safe且有新增才余352按同候选补407，不复跑screen块。原384逐ID全保+至少1新原始起点+完整407+独立audit才保留，联合387单列。失败不换metric/seed/根区间/尺度/damping，不多步救；负结果照报，不把正proxy、meanKL匹配或审计PASS当保留。

最坏13,625,321物理/468cmd（407+55+6），1live+1shadow+1independent Actor/0Value/0额外optimizer预检；5700main+600audit/root7200s≤2h，学习前1200s，完整407留3500s。384MiB child/416MiB root/global10GiB/free40GiB，CPU8/10/12 native14学习审计、400%Nice19、4GiB无swap/CUDA关闭，资源重叠只暂停本Stage且原deadline不延。真实入口零native零optimizer预检/独立RESERVED/GitHub精确树后唯一launch；运行中健康只读、不并行source/audit/backup、不外写runs。所有失败/raw/控制身份/具名模型保留。

本方法是经验参考分布自然方向和现有PPO/回放的有限结合，不宣称NPG/TRPO完整复现或原创。T313既有经验score-Fisher/.1damping/10CG及有限line search仅离线代理局部门、未导航，非首次Fisher/CG；本项改为全384动作积分Gaussian度量和非线性实际meanKL配对。T674结果半空间投影非线性失败、T648有限响应约束、T678小KL仍丢旧是必要反证；只有实际完整闭环能够检验新方向是否有用。

# T572R：原始可行 active-set QP 数值恢复来源

2026-11-15。仅更换同一凸问题的有界数值求解路径；沿用 T572 的数据、目标、冻结特征、正定 `Q`、`c` 和全部旧历史约束。每个输出为 33 维，两个输出独立求解；不增加科学因素、Adam、RL、特征或超参数网格，不延长原 T572 ADMM5000。预算、绝对截止和启动资格以本轮 PLAN 为准。

## 实际核读的一手来源

1. **David Bindel，Cornell CS4220 官方讲义，2026-04-29**：[An active set approach](https://www.cs.cornell.edu/courses/cs4220/2026sp/lec/2026-04-29.html#an-active-set-approach)。已核读该节算法文字与 `qp_as` 实现：工作集约束作为等式求解；非零方向沿保持全体约束可行的最大步长前进，遇阻挡加入约束；零方向时检验乘子，存在负乘子则删除对应约束。此为标准 primal active-set 方法，不能称本项目原创。该讲义不是本项目数值稳健性或 256 次内收敛的保证。

2. **Nocedal / Wright 作者官方书页**：[Numerical Optimization, Second Edition](https://users.iems.northwestern.edu/~nocedal/book/index.html)。本次核查到作者、书名与第二版信息；未取得并逐页核读 Algorithm 16.3 原页，不能写成“已读完整教材”或“逐行复现 Algorithm 16.3”。实际算法细节依据上面的 Cornell 原始课程讲义和公开实现。

3. **CVXOPT 官方用户手册**：[Quadratic Cone Programs](https://cvxopt.org/userguide/coneprog.html#quadratic-cone-programs)。已核读二次问题及 `optimal` / `unknown` 段：KKT 同时包含原始可行、对偶非负、stationarity、互补性；数值困难或达到迭代上限会产生未证最优的迭代点。这里只用其公式交叉核查证书含义，不安装或调用 CVXOPT，也不将其内部求解器称作本项目算法。

## 本项目代数与固定验收

对每一输出，写 `min ½wᵀQw+cᵀw, Cw≤d`，其中 `C=[H;−H]`、`d=.002·1`；若实现对行归一化，须同步缩放右端并另核原始未缩放约束。`w=0` 严格可行。当前梯度 `g=Qw+c`，工作集为 `I` 时解等式子问题 `min_p ½pᵀQp+gᵀp, C_I p=0`，其 KKT 为 `Qp+g+C_Iᵀμ=0`。因此本符号约定下终止要求 `μ≥0`；负乘子应移除，不能照搬相反不等号约定的符号。

非零方向的阻挡步长为 `α=min(1,min_{i∉I,C_i p>0}(d_i−C_iw)/(C_i p))`。这是本项目对标准算法的记号转换。等式线性求解仅指求解该子问题，不意味着浮点计算绝对精确；退化、相关约束或证书失败都须保留失败并停止，不能放宽界限后宣称成功。

每输出最多 **256 次工作集迭代**，到限未收敛即失败，不追加迭代。实际完整保存的头须同时满足原始可行违约、对偶负乘子违约、stationarity、complementarity 各 `≤1e−7`，且全部原始旧特征 `max|H W|≤.002+1e−9`；保存并由独立审计重算。初始可行或单独 primal 过关均不能代替完整证书；证书通过也只覆盖有限历史上的策略 location，不保证闭环安全、恢复成功或泛化。

T572 B 的 ADMM5000 数值未达验收与零原生执行事实应原样保留；T572R 是另行登记的同问题求解恢复，不覆盖旧失败、不追加旧 ADMM、不重跑 A 充当新独立对照。本文不报告 T572R 求解或原生运行已完成。

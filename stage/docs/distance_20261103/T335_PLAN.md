# T335：本次用户原因分析的只读证据复核

2026-11-03用户要求认真查论文、分析为何5米过而6米失败。无新训练授权预算推断。本项只读取T331成功与T334两失败原件，复核最近接触物体、轨迹偏离及动作反馈分解；不运行新导航、参数优化或算法搜索。所有Phase研究/聊天/模型完全隔离。

两主冻结geometry：原宽场景工厂、circle9移位/.80s协议不变，创建初始env后把已保存203维final native integration完整恢复，mj_forward仅重新计算当前接触，再重设所有integration字段并逐位比较。查原_collision同一规则：机器人-非机器人、排除floor/ground/plane、contact.dist≤.002m。记录真实geom/body/距离/位置/scene与final SHA。全局禁止mj_step，0任何物理步；不是重新驾驶或规划，也不改变原模型/NPZ。只读2创建，在独立audit再2创建，合计4geometry/0physical，主180秒+审计180秒。

同时在已存T334决策起点与T331相同时刻状态上冻结Actor推理。分解实际动作差=q(x_actual)-a_human=[q(x_human)-a_human]+[q(x_actual)-q(x_human)]，两项分别记录静态模仿误差和状态反馈差，逐位核验自己的动作；这是代数描述，不是两者的物理因果份额。记录保存决策起点首次位置差超过1cm/5cm（.2秒采样，不声称逐物理首次）。所有比较明确posthoc/已消费训练状态，不是独立验证。

依据本轮原始论文阅读：TaSIL arXiv2205.14812v2 §3的轨迹附近鲁棒性解释了为什么单点模仿误差不足；其专家稳定/导数假设未在我方验证，不应用理论保证。DAgger/HG-DAgger与CCIL ICLR2024 §4均强调偏离状态恢复；CCIL是学习动力学生成有验证依据的纠正标签，不能直接给观测加噪后沿用旧动作。RFCL arXiv2405.03379v1 §4、作者curriculum.py blob69095f4cd90eb8967dc19fea503c4ca4c7e03994及RLPD v4 §4/作者train_finetuning.py blob80b1b9ca9a5073d98c3064452af709222a7704e1已读，用于本轮方案评估，未复制新算法源码或称已实现。

own1MiB（metadata896KiB+buffer128KiB，无二进制拷贝），global≤1GiB/free≥40GiB，preflight只数组hash/尺寸/前缀检查0geometry0physics0Actor/Adam。主/独立审计全部输入SHA/冻结推理及接触结果一致。干净源完整远端后唯一launch；结束父/timeout/inhibitor退、完整RAM远端逐成员SHA及source收尾。主2geometry命令/审计2另计，不把4当独立导航数据。原T334结果/源/失败/模型/视频只读，不重启旧目录，不进七八米、不新增RL。若恢复未复现contact，保留反例，不凭旧circle9结论替代新事实。

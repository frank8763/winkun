# T302后论文筛选与可检验的方法决策

检索日期2026-11-02；阅读范围逐项注明。未穷尽全网，不下载外部模型/整库/依赖，不使用Phase研究，也没有复现外部论文成绩。本文是研究决策，不是算法收益或顶刊创新证明。

## 项目证据约束

safeT255在学习六米之前已六米失败，故灾难性遗忘不是充分解释。T299完整六米10/54（右10/27、左0/27）但旧274/275，说明右侧可改善又有旧能力干扰；不证明左侧不可达。T302新困难成功数据首50步使old24/hold06碰撞，整块回退；拒绝proposal未测六米，不能说六米已学会只被guard挡住。其新数据parent modeMSE仅.000928也没有解决自主控制。T284已有同状态动作小变化、闭环轨迹大偏离的描述证据；尚未证明Jacobian是唯一根因。

## 直接适配的四项一手工作

1. [TaSIL: Taylor Series Imitation Learning](https://arxiv.org/html/2205.14812)，NeurIPS2022、v2 2023。查正文§3–4/式14–15、稳定性假设及[作者仓库](https://github.com/unstable-zeros/TaSIL)README和imitation_learning/bc.py。方法匹配专家策略的高阶局部导数，不只匹配动作值。借鉴：先核验新旧Actor的输入导数/裁剪边界；若有对应证据再考虑训练期局部反馈保持。限制：增量input-to-state稳定性及光滑性不是Stage已验证事实，硬mode裁剪也不能直接套定理；原safe可作训练期旧函数参考，但不作为运行时教师。导数蒸馏早已有，不能称原创。未复制作者实现。

2. [A Single Goal Is All You Need / SGCRL](https://arxiv.org/html/2408.05804)，ICLR2025正式版另核对[会议PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf)。查§3、式1–4/算法说明及[作者learning.py](https://raw.githubusercontent.com/graliuce/sgcrl/main/contrastive/learning.py) critic_loss/actor_loss；没有运行外部代码。重点是固定困难最终目标收集、future状态对比价值、训练Actor仍用多目标，不是反复拟合成功动作。future采样用折扣几何分布，InfoNCE加LogSumExp正则，不归一化representation；Actor需真正使用价值选择动作。Stage需完整重建120目标历史和AR控制上下文；占用密度不是首次安全GOAL概率。T293只做过离线动作/盲预测头且筛查失败，并未实现完整在线SGCRL，也不能据此否定此家族。下一价值pilot须有真实动作分支排序，未过不能接Actor。

3. [Horizon Generalization in Reinforcement Learning](https://horizon-generalization.github.io/static/pdf/horizon_generalization.pdf)，ICLR2025。查§3/5.2–5.5，定理2、限制及[作者项目页](https://horizon-generalization.github.io/)。长距离推广与规划不变性/准度量结构有关，定理要求所涉状态空间处处的近距离能力，并非几个固定起点五米通过。Stage的末段新状态并未覆盖，故不能从五米100%推十米。可借鉴训练侧目标条件价值/准度量归纳偏置和覆盖检查，不运行外部图规划/路点。原仓库README直链返回404，未称审计其实现。准度量和值距离已有方法，不以简单增加距离头冒称创新。

4. [Demystifying the Mechanisms Behind Emergent Exploration in Goal-conditioned RL](https://arxiv.org/html/2510.14129)，查v1正文§3–4及限制；会议搜索记录显示ICLR2026版本，但OpenReview正文被浏览器挑战拦截，本文理论细节只依据可访问arXiv版本。文中在表示对齐等假设下解释SGCRL的隐式奖励与探索—利用变化，提示价值表示不应只有分类准确率，还需检验是否实际改变探索/动作。Stage尚无这些假设/在线结果；不把其等价关系套到我们有碰撞、有限时间、AR状态和保持约束的全部目标上。

## 对照和排除项

- [DART, CoRL2017](https://proceedings.mlr.press/v78/laskey17a.html)：核对正式摘要和正文噪声注入定义/作者README。纠偏示范覆盖有价值，但依赖能在扰动状态恢复的supervisor。Stage尚无左六米成功纠偏者，不能把给旧策略加噪声就叫已实现DART，或凭空造安全标签。
- [SR²L, ICML2020](https://proceedings.mlr.press/v119/shen20b.html)与[CAPS, ICRA2021](https://ai.bu.edu/caps/)：本轮只复核正式摘要/作者方法概览。一般空间/时间平滑是近邻，不等同匹配旧反馈，也不保证绕障时保留必要急转。Stage120输入存在历史和物理相关性，不给各列独立任意噪声充当真实观测。
- [NavOL, ICML2026](https://proceedings.mlr.press/v306/wei26f.html)：只查正式摘要，依赖全局planner标签且输出future waypoints，不直接采用到本线。
- [Safe Multi-Agent Navigation guided by Goal-Conditioned Safe RL, ICRA2025](https://safe-visual-mapf-mers.csail.mit.edu/)：查作者页方法，使用回放图和waypoint计划，不直接采用。
- [TD-MPC2](https://arxiv.org/abs/2310.16828)：本轮只核对摘要，部署局部轨迹优化不符合当前单Actor直接动作协议，不用其成绩当Stage能达到的保证。

## 实际执行顺序

先T303固定历史候选/64真实状态离线诊断，0native/0新Actor训练，精确恢复50历史Adam并独立审计；只描述导数/裁剪变化，不因结果自动加入正则。下一有界学习必须把**六米真实最终目标回报**作为主验证；若采用目标条件Critic，先验证行为/目标历史合同、占用或first-hit语义与真实动作价值排序，再接Actor。保留机制用匹配普通replay/值保持为对照，单因素判断其是否有效；不把TaSIL、SGCRL、准度量、PCGrad一次全混入无法归因。

论文可研究的是：在执行器状态、目标历史和有限时域安全终止下，怎样获得能改善真实动作选择的目标条件价值，并约束闭环能力退化。它目前只是研究问题。需要真正左六米解决、旧能力保持、逐级7/8/10米、匹配消融、多训练seed和未见地图，不能拿论文拼接或数据审计宣称顶刊贡献已成立。

# T646一手材料复核（2026-11-18）

本轮查看范围为下面论文官方摘要页、garage官方BC源文件页面；不是全文穷尽或作者结果的本项目复现。

- Ross et al., DAgger, AISTATS2011，https://proceedings.mlr.press/v15/ross11a.html ：序列决策中自身动作改变后续状态分布。本项目采用真实学生偏离状态的数据聚合思路；成功采样后缀不是任意状态上可用的理想专家，不继承其理论保证。
- Rolnick et al., CLEAR，NeurIPS2019/arXiv1811.11682v2，https://arxiv.org/abs/1811.11682 ：回放与行为保持可缓解遗忘，但论文使用on/off-policy混合，不能把本轮仅监督回放称完整CLEAR。T641/T642/T645R已经给出本项目平均/尾部动作保持仍闭环失败的反证。
- garage官方BC，https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/bc.py ：作为监督策略学习术语及实现对照，不复制其算法源码；本轮为连续裁剪动作、真实循环历史、完整任务等权，非直接框架替换。未锁定可复现commit，因此不声称逐版本复现。

执行裁剪损失在饱和方向允许保持同执行命令的不同raw幅值；只是可行集监督，不是稳定性证明。全Actor开放可能增加新能力，也可能扩大旧任务破坏；本轮用匹配对照和384逐ID闭环门检验。无论成功与否，本轮单种子/开发任务不支持顶刊创新或泛化结论；后续需多种子、独立几何/传感动力学扰动、对照消融和真机。

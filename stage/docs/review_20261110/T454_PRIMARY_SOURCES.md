# T454：固定历史函数保持与闭环退化的一手边界

2026-11-10，Stage限定文献复核。依据[T453检查](T453_FINDINGS.md)和[T453一手依据](T453_PRIMARY_SOURCES.md)，服务于父模型、128BC预热、PPO0、PPO1四个既有`plain120encoder128GRU64head2`检查点，在同三条实际父模型完整history（oldtask2center、wideRIGHTcenter、failed00）上的零native、零优化器诊断。本笔记没有生成导航证据，没有提出新训练预算。

## 已核验的三项一手材料

| 来源及本次阅读范围 | 支持、反证和适用边界 |
| --- | --- |
| Rusu等，*Policy Distillation*，ICLR 2016，[作者全文](https://arxiv.org/pdf/1511.06295)，§3.2–3.3、§4.1 | 教师产生输入与输出目标，学生用NLL/MSE/KL监督拟合；多任务交错采样且使用任务专用输出层。支持将旧策略函数另列为监督保持目标。离散Atari/Q输出和多头设置不同于Stage连续递归单头；文中经验成功不构成有限旧history误差小即可保住全部闭环成功集合的定理。 |
| Kapturowski等，*Recurrent Experience Replay in Distributed Reinforcement Learning*，ICLR 2019，[作者出版页](https://willdabney.com/publication/r2d2/)，作者摘要；[论文链接](https://openreview.net/pdf?id=r1lyTjAqYX)本次被浏览器验证页阻挡 | 作者明确研究参数滞后导致的representation drift和recurrent state staleness。本次只核验作者摘要，不声称重读全文、burn-in细节或证明。支持将递归表示变化与hidden语义列为待检验机制；并不证明T453实际出现hidden复用错误，也不证明冻结GRU可修复导航。 |
| Pfrommer、Zhang、Tu、Matni，*TaSIL: Taylor Series Imitation Learning*，NeurIPS 2022，[正式全文](https://papers.neurips.cc/paper_files/paper/2022/file/7f10c3d66c3b7863a9cda255dcac5bb7-Paper-Conference.pdf)，§2、§3，Assumptions 3.1–3.2及Theorems 3.1–3.2 | §2明确指出，即使专家数据上的population imitation risk小，学生反馈诱导的轨迹分布仍可明显改变。其正面结论需要专家闭环局部增量输入到状态稳定性、策略正则性、足够小的误差以及相应阶数的导数匹配等条件；强稳定条件下零阶匹配亦可能充分。Stage尚无这些条件或碰撞裕度的验证，不能把有限锚点MSE、平均KL或一次输出分解当成轨迹/安全保证。 |

## 固定完整history应回答什么

以下为结合上述材料的本地计算推断，不是论文对Stage的实测结论。相同观测、前动作、时间及reset序列固定时，各检查点必须用自己的encoder＋GRU从自身真实h0重建hidden；中段父hidden不能作为新参数hidden的准确替身。这样测到的是同一已记录输入历史上的函数变化。候选实际上闭环执行会改变后续观测、动作历史和hidden；本诊断没有生成这些候选轨迹，因此不能估计反馈放大、碰撞概率或新成功率。只有三条history，不能推及完整329或未见起态。

## 两种模块顺序的精确分解仍是描述

令A为encoder＋GRU整体，B为动作head。对同一完整历史前缀H，定义`Fij(H)=B_j(A_i(H; h0_i))`，0为对照检查点、1为比较检查点。A和B替换时保持所有其他参数及输出读取定义一致。逐时刻、逐drive/steer输出有：

```text
Δ = F11 - F00
先A后B：Δ = (F10 - F00) + (F11 - F10)
先B后A：Δ = (F01 - F00) + (F11 - F01)
交互项：I = F11 - F10 - F01 + F00
```

两顺序均须报告重构残差和有符号分量；分量范数无需相加为总范数，抵消时“贡献比例”可能很大。`I`给出顺序依赖：A在B1下的差减去在B0下的差正好为I，故单一顺序不能给出唯一模块归因。若同时报告裁剪前输出及实际有界动作，两套都应按各自完整Fij重算，不能把裁剪前分量直接当执行动作分量。

这些是模块函数混合的代数反事实，没有对环境闭环作干预。混合head和表示也可能不在训练时共同适配的组合上；所得大小依赖模块边界、检查点对、历史和输出坐标。它可以定位“在这些父history上，哪个组合改变了输出、何时出现及是否相互抵消”，不能证明某模块造成了T453的47项最终损失，不能把输出差解释为导航因果效应或保持模块后的性能。A作为整体还不能区分encoder和GRU各自作用。

因此T454只提供有限、可复算的功能定位；冻结GRU等改进至多是后续待证假设。双顺序精确分解和标准函数保持均不作为算法新颖性或安全证明。

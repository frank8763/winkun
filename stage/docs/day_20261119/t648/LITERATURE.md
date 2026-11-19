# T648一手材料与采用边界（2026-11-19）

完整检索/相关原论文章节与作者源码复核见 ../t647/PRIMARY_METHOD_REVIEW.md。此轮再次查看官方GEM源码页与SIL正式论文摘要页；未声称新全文穷尽。

- GEM，https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py ，原作者实现为SGD+局部梯度约束。T648不用其QP/不复制代码，而检查真实有限Actor位移、实际响应并审计自定义Adam提交。不能把GEM局部假设当闭环安全定理。
- SIL，https://proceedings.mlr.press/v80/oh18b.html ，正式算法是离策略actor-critic；本轮成功筛选监督经验没有V与正优势权重，不称SIL精确复现。
- DAgger，https://proceedings.mlr.press/v15/ross11a.html ，自身动作改变后续状态分布；原名义历史响应约束仍不能覆盖所有新闭环状态，所以保留原起点原生验收。
- A-GEM、Behavior Transformers、AWAC/IQL/EWC依据及反例边界见前述PRIMARY_METHOD_REVIEW。任务均衡、整轨迹留出、有限容差都非独立原创；若响应保护仍闭环退化，结果否定本次代理充分性，不证明导航无解。

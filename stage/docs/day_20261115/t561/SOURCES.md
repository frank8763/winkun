# T561：尖锐连续动作蒸馏的一手文献与机制边界

2026-11-15。仅 Stage 文献工作；未训练、未调用原生模拟器、未检查运行进程。以下阅读到正文指定节，不能把摘要、作者示例或坐标回归结果当作本项目导航保证。文献机制、项目事实和下一对照的推断分别陈述。

## 本地问题与当前证据

[T559 FINDINGS_AND_NEXT](../t559/FINDINGS_AND_NEXT.md)记录：冻结因果 GRU 与 120 维观测形成 184 维输入，40450 参数的 184→128 tanh→128 tanh→2 平滑头，固定相同 27 条成功历史和 1500 次 Adam。平均 MSE 降约 93.57%，但仅 14/27 历史通过逐历史命令 RMS≤.02、max≤.1；最坏 RMS=.119415830、max=1.056344718。自主原生验收未执行，不能记为失败率，也不能说这个架构理论上无法表示标签。

[T560 RESULT](../t560/RESULT.json)记录 6779 行、6173 个唯一保存特征、303 组精确重复且冲突数为 0。该有限样本事实排除了这些精确重复组中的标签矛盾；没有排除近邻冲突、遗漏状态、尺度影响或闭环分布外问题。最坏两条主历史中，所需纠正超过 .1 的决策占 83.95%/87.32%，因此“只有罕见短段未被采到”不足以解释全部结果。有限特征距离上的 secant 是描述量，不是全局 Lipschitz 下界、不可表示证明或因果结论。

## 正文证据

### 1. 连续动作监督有先例，但动作误差需要任务依据

Levine, Finn, Darrell & Abbeel, *End-to-End Training of Deep Visuomotor Policies*, JMLR 2016，已读 §4.3（印刷页 12）。高维观测到连续力矩策略的监督目标含策略均值与轨迹控制器均值之间的加权二次误差；权重来自控制器条件协方差的精度矩阵，与局部 cost-to-go 曲率有关。旧分布样本另做重要性处理。[原论文 §4.3](https://www.jmlr.org/papers/volume17/15-522/15-522.pdf#page=12)

**对本项目的推断：**大多数动作拟合良好不能抵消少数关键命令偏差；但不能把当前残差大小直接称为 cost-to-go 权重。T559 没有该论文的控制器分布、局部曲率和分布匹配合同，不能沿用其收敛结论，也不建议引入运行时控制器。

### 2. 频谱偏置是可信假说，不是 T559 的已证病因

Rahaman et al., *On the Spectral Bias of Neural Networks*, ICML 2019，已读 §2、§3.1、§4。正文分析的是 ReLU 网络；§3.1 的一维正弦混合实验使用 6 层、宽 256 的 ReLU 和 full-batch 梯度下降，观察到低频先拟合及高频对参数扰动更敏感。§4 强调数据流形几何会改变可学习频率。[原论文](https://proceedings.mlr.press/v97/rahaman19a/rahaman19a.pdf)

**边界：**T559 是有限宽 tanh、Adam、相关历史和混合 184 维特征，不能把该实验外推成定理。沿轨迹动作快速变化不自动等于“特征空间高频”：它也可能来自状态快速移动、历史编码、特征单位或目标切换。当前证据支持检验表示/优化机制，不支持宣布谱偏置已被唯一定位。

### 3. Fourier features 论文明确限制了适用场景

Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*, NeurIPS 2020，已读 §3–5、Fig.3–4。§3 的 NTK 近似涉及宽度趋无穷和小学习率；§4 明确研究稠密、低维欧氏坐标，区别于稀疏高维机器学习输入。映射可调核带宽；带宽太宽会产生过拟合/混叠。§5 指出稠密 Fourier 基所需采样随维度指数增长，因此采用随机特征。[原论文 §3–5](https://proceedings.neurips.cc/paper_files/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf#page=3)

**边界：**184 维冻结混合特征不满足其密集坐标设置；随机特征可计算不等于已经解决高维覆盖。新增映射还改变第一层参数量、输入尺度与优化条件。论文不保证本项目误差门或导航成功，也不构成必须增加编码的依据。

### 4. SIREN：可检查的同容量替代，需要配套初始化

Sitzmann et al., *Implicit Neural Representations with Periodic Activation Functions*, NeurIPS 2020，已读 §1、§3.1–3.2、§4。以空间/时空坐标拟合图像、音频、形状及 PDE；正弦网络在其任务上改善细节和导数表示。§3.2 说明随意均匀初始化导致较差准确性/收敛，分析激活统计，并报告 ω₀=30 在其应用中有效；输出频率分布的形式化分析不在该文范围内。[原论文正文](https://arxiv.org/html/2006.09661)

该网络在代数上可接受任意输入维数；正文没有提供 184 维因果历史特征到车辆命令的实证或导航保证。其激活统计假设也不自动覆盖相关的冻结特征。光滑正弦能够表达快速变化，与小扰动动作稳定、有限数据泛化和闭环安全是不同问题。

已读作者[官方实现 modules.py](https://github.com/vsitzmann/siren/blob/master/modules.py)的 `Sine`、`FCBlock`、`sine_init`、`first_layer_sine_init`，及作者[explore_siren.ipynb](https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb)的 `SineLayer`/`Siren`（2026-11-15 读取，master 可变化）。作者可执行约定是：

- 隐层计算 sin(ω·(Wx+b))，示例 first/hidden ω 均为 30。
- 第一层权重 W∼U[−1/n,1/n]；后续隐层 W∼U[−√(6/n)/ω,+√(6/n)/ω]，n 为该层 fan-in。
- `outermost_linear=True` 时输出为仿射层，输出权重仍按 √(6/n)/hidden_ω 初始化，不对输出再做 sine。
- 作者示例只重设权重；bias 保留 `nn.Linear` 初始值。自行设零 bias 需要明确记作偏离，不能默称逐实现复现。

**对 T561 的推断：**184→128 sine→128 sine→2 与原头参数量同为 40450，固定同一数据、backbone、损失和 1500 步可检验“正弦＋规定初始化”的整包替代。初始化分布不同，所以不是纯激活函数的因果消融。不要同时加 Fourier 映射、导数损失、样本重权或额外步数；固定 ω=30 也只是预先选定的作者默认值，不是已经验证的最优值。

### 5. 导数监督需要真实目标导数；高频也可能使其更差

Czarnecki et al., *Sobolev Training for Neural Networks*, NeurIPS 2017，已读 §2、§3、§4.1–4.2。方法将真实目标函数的输入导数加入监督，可随机投影 Jacobian。§4.1 报告低数据、高频 Ackley 函数的反例：导数监督在测试上可能更差。§4.2 的政策蒸馏为 Atari 动作概率分布的 KL＋导数匹配，按 episode 留出；不是连续车辆命令闭环验收。[原论文正文](https://arxiv.org/html/1706.04859)

**边界：**动作时间差分不是 184 维输入 Jacobian。特征邻接 secant 混入轨迹方向与状态变化，不能充当完整偏导标签。导数可计算也不保证正确的恢复动作；本次若仅换头，不应顺手引入 Sobolev 项后归因于 sine。

### 6. 难例采样是已有替代机制，也有明确反证

Loshchilov & Hutter, *Online Batch Selection for Faster Training of Neural Networks*，arXiv v4 2016，已读 §4、§7–8、Algorithm 3。按最近损失排序并提高难例采样概率，定期刷新；作者方案不做梯度重要性校正。§8 明示主要评价限于 MNIST/OCR，依赖损失排名相对稳定，CIFAR-10 初步结果尚未优于 shuffling；选择压力在其他数据上不明。[原论文正文](https://arxiv.org/html/1511.06343)

**项目推断：**这是可以单独检验的优化分配机制，不能宣称原创或保证降低最坏命令误差。若均匀风险不变，需明确采样校正；若故意不校正，就是重加权目标。只追最高残差可能牺牲其余历史，T560 也不支持把最坏两条仅当罕见段。与换头同时做会混淆归因。

### 7. 保存历史拟合与闭环分布是两道门

Ross, Gordon & Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011，已读 §1、§3–4 和 Algorithm 3.1。动作改变后续输入分布；DAgger 在学习器访问状态上获得专家标签并聚合。其界依赖 no-regret、损失/样本和专家覆盖等条件。[原论文](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)

**边界：**固定 27 条历史的监督替代不满足 DAgger 数据收集合同；不能套用其保证，也不因此加入运行时教师。保持 T559 原逐历史门槛只是第一关，通过后仍须按另行注册的原生闭环、按成功 ID 保持和留出条件报告；低损失不能替代这些证据。

## 可支持的有限判断

现有证据可支持一次固定容量、固定步数的 SIREN 头替代作为可证伪候选，不支持“高频已定位”“必然学会”或“高维限制已解决”。失败也只能否定这次特定配置和预算，不能推出 sine/神经策略普遍无效。难例覆盖、导数监督和 Fourier 映射均为既有方法或已有近邻；本项目组合与工程实现不自动构成原创算法、泛化、跨场地或实体车有效性。

# T621 实际训练路径加速验证（零更新）

本机Ryzen9 8945HX为16物理核/32逻辑CPU，RTX5060 Laptop8GB；已安装PyTorch2.11+cu128，sm120实际float64矩阵和真实完整历史GRU梯度路径可执行，无安装/更换环境。当前无其他GPU compute进程。用户最新授权充分利用CPU/GPU，取代后续新协议GPUoff/双CPU限制；历史协议不改。

实际同批、同初值、float64：CPU1 Critic4.6366s/Actor2.1245s；CPU2 1.1909/2.1901；CPU4 1.2967/2.3176；GPU .4162/.1210。仅单批梯度，不报全程训练倍数。峰值281872896B约269MiB。原CPU实现/新CPU/GPU损失和梯度一致，GPU重复梯度逐位一致，所有参数/Adam/targets未变。GPU相对原CPU1对应11.1倍/17.6倍；CPU增加线程并非必然更快。

旧依赖导入会隐藏CUDA、eval GRU无法cuDNN反传；在新适配进程恢复用户指定CUDA，只有无dropout的可训练branch切train模式。两次拒绝和原错误保留，不改结束实验。无native/optimizer更新。后续GPU训练必须独立影子与历史重构，不把单批数值等价当完整优化器训练已验证。

确定性设置依据[PyTorch CUDA环境变量](https://docs.pytorch.org/docs/main/cuda_environment_variables.html)、[循环网络确定性说明](https://docs.pytorch.org/docs/stable/cudnn_rnn_determinism.html)。Blackwell构建背景见[官方2.7发布说明](https://pytorch.org/blog/pytorch-2-7/)，本机兼容性以实际运行结果为准。

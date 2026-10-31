# T218 启动失败与 T219 独立修复协议

T218已FAILED，零物理步/命令/Actor、V、RND更新，不重启。父进程BrokenPipe发生在首次发权重；单进程零步调用原生加载器复现ModuleNotFoundError: g5b。原加载器依赖/tmp/stage_daad_g5b_selector_recovery_v1，目录已缺失，具体消失原因未查明。不能将此次工程启动失败算作RND导航反证。

历史T71恢复日志给出g5b SHA256 d218726bf0b08c19d0fbfee0e09821f62b6c0f95fb3575dfa0d45855880bb5f6。从用户原Stage备份zip只取16,271字节模块，逐字节哈希匹配，保存t219_rnd_runtime/native_compat/g5b.py；注册显式加载同模块，不改历史源文件、不运行其训练入口。原生加载核验通过；四个真实spawn子进程加载后就绪握手，全部正常退出，零物理/优化器。启动时再次握手，异常独立写日志/收据，避免只留下BrokenPipe。最终工作进程须验证退出，必要时只杀本次子进程。

T218零更新却导出4份初始权重；逐张量确认分别等于原T194与T217保留输入后，只去掉这些重复初始化容器，释放4,683,728字节。原failure/summary/protocol/log及原始模型不动，副本容器SHA、状态SHA、元数据、原始输入引用与核验均在stage/runs/t218_compact_rnd/initialization_copy_receipt.json；不是删除学到的模型或唯一失败证据。

T219是新注册的修复实现，不是重启T218。保持T218科学协议、2026103218种子和全部训练超参数，见t218_notes.md：同初始MLP/独立V/原始LEFT/原生动作域/无保持，唯一科学因素为新颖度奖励0与.05回合上限，最多4轮32新训练回合，120总命令/2.2M所有物理步/900秒、末180秒验收、6MiB输出/global1GiB/free40GiB。全部数据重放与每批shadow重算通过才接受。T159/T194不覆盖，不自动晋升/续训/扩距。

2026-10-31注册前重新检索：RND原论文摘要与作者ppo_agent实现（https://arxiv.org/abs/1810.12894 ; https://github.com/openai/random-network-distillation/blob/master/ppo_agent.py），仅借鉴固定随机目标、训练期预测误差新颖度，非原Atari复现、不是新算法。反证来源ICML2024 DRND正式摘要明确指出RND奖励分配不一致（https://proceedings.mlr.press/v235/yang24w.html）；本轮不临时换成DRND，不将离线误差比当可靠计数或导航成功。Python官方multiprocessing说明spawn重新导入与Pipe端点生命周期（https://docs.python.org/3/library/multiprocessing.html），用于工程握手设计，不是科研贡献。阅读范围仅上述摘要/代码片段/官方文档；本轮未声称全网穷尽或论文全文阅读。

当前GitHub远端新增备份未核验；原上下文已确认连接器不可调用且本机网络超时，不能把本地提交称作已上传。

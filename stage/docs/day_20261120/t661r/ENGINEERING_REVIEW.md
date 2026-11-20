# 原T661失败与T661R修复依据

原T661实际启动2026-11-20 09:25:53；学习13秒内在live/shadow一致性处失败，0物理步、6管理命令、Actor live/shadow各4次、Value各16次，independent0。原raw全远端逐成员读回SHA通过，服务inactive，模型T501保留；不是导航验收失败。

已用真实成熟Value Adam检查点和原M.load零更新重现共享CPU tensor。GPU模型的Adam.step仍位于CPU，原load_state_dict加载同一state对象到两个模型，step.data_ptr相同；数值改变一方会同步改变另一方。完整Actor训练使用冻结旧Value优势，因此本次报错位置在完整trace一致性，不能据此判断Actor方向本身无效；原运行未保存最终proposal，不补造缺失结果。

新独立root的load为每一个模块和优化器逐份deepcopy。实际GPU零更新回归核验原红条件、深拷贝后step独立、修改一份不会改变另一份或保存state；随后完整64轨迹、387参照、优势detach/真实梯度、runtime/worker/transport/CPU暂停恢复/GPU有界等待回归均PASS。没有通过再次调用optimizer.step来“预检”，未花额外优化器额度。完整训练一致性仍待真实运行验证。

代码：stage/experiments/t661r_value_state_isolation/t661r_learning.py中的load；实际复现路径prepare.py。原注册代码与run保持只读，失败报告保留。新scope仅修工程隔离，算法参数不变。

由于原T661 live/shadow明确额度已消费，新执行不能擅自从independent额度挪成live/shadow。待核定补live/shadow各4 Actor/16 Value及6管理命令；独立审计上限不增加，物理、总时间、存储和绝对硬截止均不增加。请求原因来自实际消费账目与用户原硬上限，不是技能要求审批。

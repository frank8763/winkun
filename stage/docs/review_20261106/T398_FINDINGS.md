# T398：预热候选损伤旧中心，六米能力仍有关键缺测

2026-11-06。原始summary/audit/模型/失败不改。主运行20:30:16–20:36:48，392.130652秒，27主命令＋15零物理预检＝42命令，517493全部物理步。固定200监督live＋200shadow逐位一致；独立audit重建200历史Adam通过，预检8合成丢弃Adam另列。真实RL、新学生随机采集、当前学生教师再查询均为0。

预热候选在旧八中心7/8，唯一dev_task6_13接触，按原协议整Actor/Adam回退并停止。原4秒恢复标签的拟合误差下降不是闭环成功。warm/proposed.pt.gz仍保留：SHA3f8f1adbd7a0966a55239fbc3ae825500c4a38f66a7844dc97aa7406689042b3，policy c8eb7d5a7bb64cb65d64e20b6796d9c0c2557fcfd195aa95671f14417376183f。

之后post_warm/final的53秒超时、54秒GOAL和原起点LEFT接触/RIGHT GOAL，实际都属于已回退的T393 policy5c43ac025862bc607178cbd90b4f17231e1b8976feec88a99168d2b772ce4ae6；不是预热候选的六米测试，也不是新学习收益。candidate.pt.gz是回退父，不能替代warm/proposed.pt.gz。被拒候选的53/54恢复及原起点六米尚未测量，无法断言“学到但只被guard挡住”或“完全没学到”。T291取消旧硬门、T284/T285稀疏投影均已有负证据，不重复包装为创新。

完整新主任务54/old275/fresh16/独立reload78均未执行null，不填0%失败率；六米稳定=False，7/8未进入。原0.57m/56s与主0.70m/80s不混合。safeT255、原人控、全部失败、五米HD视频保持。

所有原生父/worker/driver/timeout已退。独立audit PASS。43原件451570B已RAM归档并完整远端二进制逐成员SHA读回；归档SHA e101b7ab458da32dfcff932f62fdef0500b53d5501ba2df1b07c02ee723faa42，见release_receipt_T398.json。最终源精确同步见t224_archive/latest_source_sync_receipt.json，原driver报告不覆盖。

下一先用已保存拒绝候选作零新优化器、固定四条件及原生精确重生诊断。父历史基线按同policy/同任务/同prefix复用，不充新独立样本。根据真实能力变化，再选择旧行为保持与恢复获取如何分离；不先堆Critic、相机或重开SAC。

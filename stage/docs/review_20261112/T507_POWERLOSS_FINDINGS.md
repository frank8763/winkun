# T507 意外关机检查

2026-11-12 北京时间20:34:26附近机器重启。T507旧inhibitor/driver/native父worker均已消失，不是仍在后台训练；原目录未重启、未覆盖status/summary或补造完成记录。

完成部分：首块32监督live+32shadow；同一候选18项开发检查全安全GOAL，含旧八中心8/8、原7场地保持及3新任务成功（sector0_offset1、sector7_offset1、sector7_offset2），18原生exact全部数组/积分相同。候选文件 block0_proposal.pt.gz 完整SHA ba3d7a21467b16686bd2e583f5e12a75e07e15c80c6ff0fb889ced69bf2dbe18。已做独立0native/0新学习的32丢弃历史Adam重构，全Actor/Adam哈希、完整history/动作密度/奖励/NPZ与接受规则通过。研究候选保留，原正式接受T505不覆盖。

中断处：开始派发剩余14场地中的前2项，未保存任何可用新case结果；零字节0019.npz保留，既不称有效超时也不称成功。两项未完成请求为sector0_offset2和sector0_offset3，其余12项尚未派发。最终全field24精确重放未开始。完整field24=null，旧275/six54/eight54/fresh16/78未执行；不能称10/24最终成功率，更不能以已测10/10称全场地100%。

已完整物理记录683424，最近落盘共享counter下界747195；断电时内存counter丢失，实际总物理只能界定747195–831302，不能补造精确值。主已派发命令40（含启动2，未正常退出2），预检6零物理另计。全部原始文件包括空文件/原status/log/模型保持不变，完整SHA清单在独立只读复核目录。

复核代码 stage/experiments/t507_powerloss_review/review.py，独立输出 stage/runs/t507_powerloss_review；仅检查/历史重构/RAM备份，0新native/0新学习。该复核不恢复训练、不重新跑已结束目录。远端状态以T507_POWERLOSS_REPORT.json及release_receipt_T507_powerloss.json为准。

下一次若获继续指令，应为冻结此同一候选的剩余14项和完整24项重放单独注册有界验收协议，先核算实际磁盘与剩余时间；不重新做训练，不将原断电请求算已完成、不扩大旧训练预算。

# 保留模型与停止位置

当前固定课程 Actor：`stage/current/outputs/stage_t159/consolidated_actor.pt`，SHA256 `d0065843f6766407038bae679b32177b20991a429010bb3cb83872413f8ea570`。

T161 final：`stage/current/outputs/stage_t161/episode_011.pt`，SHA256 `d62e94c060449a0ca2ed94db3748532703f54e071a2c14d7ad3bb73ea4dba7ee`。

T164 用户暂停后的最新完整边界：

| 组 | 文件 | 已保存新增物理步 |
|---|---|---:|
| gamma=.99 | `stage/current/outputs/stage_t164/gamma099/episode_008.pt` | 170199 |
| 2秒折扣视野 | `stage/current/outputs/stage_t164/gamma2s/episode_008.pt` | 165498 |

这两个文件含Actor、Critic、target网络、优化器、replay与RNG；具体SHA256见 `stage/manifests/paused_training.json`。它们未完成原定预算，未作最终配对评估。回合中未保存的实时模拟器状态无法恢复，现有训练脚本也不直接支持这次暂停后的恢复入口，需下一步明确实现并验证后才继续。

以上文件本地已就位，Git忽略大文件；远端在Release对应压缩包。不要将Actor-only文件作为TD3完整恢复断点。

# T338实际结果

固定48秒宽场景恢复pilot，非原起点六米验收。主运行：FAILED。

初始完整旧275与恢复控制：

```json
{
  "old275": {
    "success": 274,
    "collision": 0,
    "timeout": 1
  },
  "recovery48": {
    "success": 0,
    "collision": 0,
    "timeout": 1
  },
  "control56": {
    "success": 1,
    "collision": 0,
    "timeout": 0
  },
  "shared_initial_behavior_preflight": true
}
```

两臂结果（随机训练、确定性恢复与原起点分列）：

```json
{}
```

物理步 4178976；命令 288；live/shadow Adam {"live_actor": 0, "shadow_actor": 0}。

独立审计 False；全部原件远端逐SHA读回 True。最终完整old275/six54/fresh16/reload78本pilot未执行，不将缺测当失败率，不晋升/不进7或8。CAPG/PPO已有方法，不称原创或保证成功。

# Stage 2026-11-03 夜间实际结果

主运行 COMPLETE；全部物理 10555201；全部命令 594。

两臂结果：

```json
{
  "fixed": {
    "accepted_blocks": 0,
    "stop_reason": "TWO_CONSECUTIVE_OLD20_REJECTIONS",
    "final_prefix": 0,
    "wide5": {
      "success": 0,
      "collision": 5,
      "timeout": 0
    },
    "original2": {
      "success": 0,
      "collision": 1,
      "timeout": 1
    }
  },
  "reverse": {
    "accepted_blocks": 2,
    "stop_reason": "TWO_CONSECUTIVE_OLD20_REJECTIONS",
    "final_prefix": 240,
    "wide5": {
      "success": 0,
      "collision": 5,
      "timeout": 0
    },
    "original2": {
      "success": 0,
      "collision": 1,
      "timeout": 1
    }
  }
}
```

原六米完整门槛：

```json
{
  "chosen": "reverse",
  "old275": {
    "success": 274,
    "collision": 1,
    "timeout": 0
  },
  "six54": {
    "success": 1,
    "collision": 27,
    "timeout": 26
  },
  "fresh16": null,
  "reload78": null,
  "stable": false
}
```

独立审计通过：True；全部原件远端逐SHA读回：True。七八米未训练。

Actor/Q实际live/shadow Adam：{"live_actor": 7000, "live_critic": 9000, "shadow_actor": 7000, "shadow_critic": 9000}。缺测不是失败率。标准SAC/RLPD/反向课程不称原创。

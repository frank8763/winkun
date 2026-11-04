# T339实际结果

固定48秒宽场景恢复pilot，非原起点六米验收。主运行：COMPLETE。

冻结初始控制（初始old275为T338历史复用）：

```json
[
  {
    "prefix": 240,
    "success": false,
    "collision": false,
    "timeout": true,
    "remaining_m": 2.5986770632524316
  },
  {
    "prefix": 280,
    "success": true,
    "collision": false,
    "timeout": false,
    "remaining_m": 0.19988593900759286
  },
  {
    "prefix": 200,
    "success": false,
    "collision": false,
    "timeout": true,
    "remaining_m": 2.972339079630215
  },
  {
    "prefix": 150,
    "success": false,
    "collision": true,
    "timeout": false,
    "remaining_m": 2.268498845187819
  },
  {
    "prefix": 0,
    "success": false,
    "collision": true,
    "timeout": false,
    "remaining_m": 2.2541135268101806
  }
]
```

一次更新前后确定性控制结果：

```json
{
  "cached_complete_native_exact": true,
  "before": [
    {
      "prefix": 240,
      "success": false,
      "collision": false,
      "timeout": true,
      "remaining_m": 2.5986770632524316
    },
    {
      "prefix": 280,
      "success": true,
      "collision": false,
      "timeout": false,
      "remaining_m": 0.19988593900759286
    },
    {
      "prefix": 200,
      "success": false,
      "collision": false,
      "timeout": true,
      "remaining_m": 2.972339079630215
    },
    {
      "prefix": 150,
      "success": false,
      "collision": true,
      "timeout": false,
      "remaining_m": 2.268498845187819
    },
    {
      "prefix": 0,
      "success": false,
      "collision": true,
      "timeout": false,
      "remaining_m": 2.2541135268101806
    }
  ],
  "after": [
    {
      "prefix": 240,
      "success": false,
      "collision": false,
      "timeout": true,
      "remaining_m": 2.734952780811379
    },
    {
      "prefix": 280,
      "success": true,
      "collision": false,
      "timeout": false,
      "remaining_m": 0.1999984619043416
    },
    {
      "prefix": 200,
      "success": false,
      "collision": false,
      "timeout": true,
      "remaining_m": 2.9677953338910372
    },
    {
      "prefix": 150,
      "success": false,
      "collision": true,
      "timeout": false,
      "remaining_m": 2.2666961204658285
    },
    {
      "prefix": 0,
      "success": false,
      "collision": true,
      "timeout": false,
      "remaining_m": 2.2543755358678346
    }
  ],
  "old20": {
    "success": 19,
    "collision": 0,
    "timeout": 1
  },
  "learned48_diagnostic": false
}
```

物理步 951889；命令 40；live/shadow Adam {"live_actor": 8, "shadow_actor": 8}。

独立审计 True；全部原件远端逐SHA读回 True。最终完整old275/six54/fresh16/reload78本pilot未执行，不将缺测当失败率，不晋升/不进7或8。CAPG/PPO已有方法，不称原创或保证成功。

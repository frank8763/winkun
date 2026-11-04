# T351 actual result

Native state: COMPLETE. Commands 415; physical steps 6121459; live/shadow updates {'live': 8, 'shadow': 8}.

Result (null means unexecuted):

```json
{
  "initial": {
    "old275": {
      "success": 240,
      "collision": 35,
      "timeout": 0
    },
    "old20": {
      "success": 9,
      "collision": 11,
      "timeout": 0
    },
    "wide5": {
      "success": 2,
      "collision": 2,
      "timeout": 1
    },
    "original2": {
      "success": 0,
      "collision": 2,
      "timeout": 0
    }
  },
  "final": {
    "old20": {
      "success": 9,
      "collision": 11,
      "timeout": 0
    },
    "wide5": {
      "success": 2,
      "collision": 2,
      "timeout": 1
    },
    "original2": {
      "success": 0,
      "collision": 2,
      "timeout": 0
    },
    "old_success_ids": [
      "dev_task0_13",
      "dev_task1_13",
      "dev_task2_13",
      "dev_task3_13",
      "dev_task4_00",
      "dev_task4_13",
      "dev_task5_13",
      "dev_task7_13",
      "dev_task7_26"
    ],
    "wide_success_ids": [
      "wide_dev_2",
      "wide_dev_4"
    ]
  },
  "old275": null,
  "six54": null,
  "fresh16": null,
  "reload78": null,
  "stable_six": false
}
```

Independent audit=True; complete raw remote byte readback=True; full-six gate=False. Overnight/7m/8m were NOT started by this driver. No automatic continuation of failed budget. Stage only; Phase isolated.

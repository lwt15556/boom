# 识图样本与评分

调试截图保存在 `_debug/screenshots/probes` 和 `_debug/red_scout_samples`。
每个包含 `result.json` 的目录都可以旁边放一个人工复核文件 `review.json`：

```json
{
  "expected": "miss",
  "note": "空白水域，不能识别为已命中"
}
```

`expected` 允许 `hit`、`miss`、`unknown`。运行：

```powershell
.venv\Scripts\python.exe tools\evaluate_vision_samples.py --root _debug --json _debug\vision_report.json
```

报告分别统计：

- `decisions`：程序最终判定的 hit/miss/unknown 数量；
- `evidence_kinds`：动态命中、静态残骸、完整潜艇证据来源；
- `unknown_rate`：低置信度结果占比；
- `review_accuracy`：仅对已经人工标注的样本计算；
- `by_level`：按关卡的标注准确率。

没有 `review.json` 的样本只计入识别分布，不会被当作“正确”。建议按以下目录标签整理人工样本：

```text
01_red_marker_as_hit
02_neighbor_wreck_missed
03_empty_water_as_hit
04_complete_submarine_missed
05_completed_submarine_downgraded
06_l_shape_not_cleared
07_l_shape_cleared_wrong_cell
08_animation_frame_false_hit
```

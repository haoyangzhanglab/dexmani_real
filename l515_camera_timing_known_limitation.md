# L515 RGB/Depth 发布时序：已知限制

状态：已记录；RGB 降帧根因待独立诊断。在诊断结果确认前，不修改 production camera
cadence、exposure/gain、repeated-color publication 或 skew filtering。

## 实测事实

2026-08-22 使用 L515 `f1382055`、640×480、nominal 30 FPS、Short Range
preset 5、confidence 1，在 `hard_depth_edge` 场景采集 300 个 frameset：

- depth frame number 300/300 次更新，设备帧号速率约 30.13 Hz；
- color 只有 199 个不同 frame number，101 次沿用上一张 color；
- 新 color 的相邻设备时间戳固定约 59.96 ms，即约 16.68 Hz；
- 全部 frameset 的 `abs(depth_ts-color_ts)`：P50 12.53 ms、P95 40.51 ms、
  P99 42.59 ms；
- 只统计 color frame number 更新的 pair：P50 8.19 ms、P95 15.58 ms、
  P99 16.45 ms，全部不超过 16.67 ms。

采集证据位于本地未跟踪目录
`l515_native_shadow_20260822_222537/`。该目录是大型实验产物，不属于源码。

## 当前语义与影响

native depth XYZ、`T_color_from_depth`、color distortion projection 的空间几何
不因帧率不同而失效；但重复 color 与新 depth 并非同时曝光。静态场景通常不受影响，
运动中的手或物体可能出现点云颜色错位。

- 点云 XYZ 以 depth timestamp 为准；
- RGB 只是 color timestamp 下的 projected measurement；
- `rs.align()` 或自定义空间投影都不能消除这个时间差；
- raw v21 分别保存 depth/color frame number 与 timestamp，避免隐藏该限制；
- 安全、碰撞和 workspace 判断不得依赖 projected RGB。

三类 timestamp 应区分：

- XYZ geometry 对应 native depth exposure/time；
- projected RGB 对应 native color exposure/time；
- camera pair `source_monotonic_ns` 是 `min(depth_mapped_source, color_mapped_source)`，
  用于 conservative pair freshness / causality，不等价于 XYZ 的唯一物理采样时间。

## 当前决策

本轮不修改 camera worker 的发布节拍，不丢弃重复 color，不增加未经单独评审的
skew threshold。processed v4 可以使用这些 native pair，但其 metadata 只声明
`native_color_projection`，不声明同步曝光或完美可见性。

## 后续独立议题

后续讨论应独立比较：

1. 仅发布 color frame number 更新的 pair；
2. RGB auto-exposure priority、exposure 与 gain 的实际 readback；
3. 固定曝光维持 30 FPS 对亮度和噪声的影响；
4. 对 learned policy 的动态颜色错位容忍度；
5. realtime worker 在真实 L515 输入下的 source-to-publish 频率与延迟预算。

# L515 RGB/Depth 发布时序：已知限制

状态：根因已确认（2026-08-23 第二轮消融），并已把 `auto_exposure_priority=0`
设为 production 默认。RGB 降帧是 sensor 端曝光限速，不是下游丢帧。

## 根因与第二轮消融结论（2026-08-23）

使用 `diagnose_l515_rgb_timing.py`（non-mutating）与 `run_l515_ablation.py`
（guide §32 Branch A）在偏暗真实场地确认：

- baseline（priority ON）：color unique rate 16.68–17.92 Hz，per-frame
  `actual_exposure` P50 = 59682 µs（≈60 ms），frame number 连续（Δn≈1），
  normalized period ≈60 ms，luminance ≈75/255；
- 消融（priority OFF，AE 仍 ON）：color unique rate **30.15 Hz**，
  `actual_exposure` P50 = 29841 µs（≈30 ms），luminance ≈75/255（几乎不变）。

结论：Auto Exposure 在 **Auto-Exposure Priority=ON（默认）** 时把曝光拉到 ≈2 个
帧周期（~60 ms），把 color 传感器物理限速到 ~16.7 Hz。关掉 priority 后曝光压回
≈1 个帧周期（~30 ms），帧率恢复 30 Hz；亮度由增益补偿、几乎不变，代价是暗场下
噪声上升（增益约翻倍）。`actual_exposure` per-frame metadata 才是可信值，AE=ON
时 `get_option(EXPOSURE)` 的 readback（如 166）是手动种子、不代表实际曝光。

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

- **Auto-Exposure Priority 默认关闭**：`RealSenseConfig.auto_exposure_priority`
  默认 `0.0`（OFF），在 `RealSense._apply_color_config()` 于 pipeline 启动后写入
  color 传感器并回读校验。Auto Exposure 保持 ON；OFF 只把曝光压在帧周期内、用
  增益补偿，从而在暗场维持 30 Hz RGB。设 `None` 可退回设备默认。
- 本轮不修改 camera worker 的发布节拍，不丢弃重复 color，不增加未经单独评审的
  skew threshold。processed v4 可以使用这些 native pair，但其 metadata 只声明
  `native_color_projection`，不声明同步曝光或完美可见性。

## 剩余风险

priority OFF 后暗场 RGB 噪声上升（增益补偿）；诊断只测了平均亮度、未测 SNR。
若下游（点云着色、ArUco、触觉对齐）在暗场对噪声敏感，优先补 workspace 照明
（AE 自然缩短曝光、降增益），而不是重新打开 priority。

## 后续独立议题

后续讨论应独立比较：

1. 仅发布 color frame number 更新的 pair；
2. ~~RGB auto-exposure priority、exposure 与 gain 的实际 readback~~ — 已由
   2026-08-23 消融确认（见「根因与第二轮消融结论」）；
3. ~~固定曝光维持 30 FPS 对亮度和噪声的影响~~ — 亮度几乎不变，噪声上升已记录
   为剩余风险，但未做 SNR 定量；
4. 对 learned policy 的动态颜色错位容忍度；
5. realtime worker 在真实 L515 输入下的 source-to-publish 频率与延迟预算。

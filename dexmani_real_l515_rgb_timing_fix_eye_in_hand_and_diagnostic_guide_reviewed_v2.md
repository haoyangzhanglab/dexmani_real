# DexMani Real：L515 RGB 时序、Eye-in-Hand 几何修复与帧率诊断实施指南（Reviewed v2）

> Review baseline: `haoyangzhanglab/dexmani_real@e28aa3e72646e12e065050ec24bbc750033506a1`  
> Review date: 2026-08-23  
> Current camera path: Intel RealSense L515, current calibration is `eye_to_hand`  
> Current storage pipeline: raw v21 → processed HDF5 v4 → Policy Zarr v2

## 1. 目标与范围

本轮只做两类工作：

1. **修复不依赖“是否由自动曝光导致 RGB 降帧”即可确定的问题**；
2. **新增一个尽量低开销、默认不修改 RGB camera control 的 timing diagnostic**，等待人工在真实场地运行后，再决定是否修改 Auto Exposure、Auto-Exposure Priority、Exposure、Gain、stream format、pipeline/sync 策略或数据契约。

核心原则：

> **先修 contract 和 timestamp measurement 的确定性问题；再测量；不要把根因假设写进 production behavior。**

---

## 2. Review 结论：对上一版指南的修正

### 2.1 `source_monotonic_ns = min(depth, color)` 不应修改

当前 raw v21 已明确声明：

```text
camera_pair_source_monotonic_ns_semantics
= minimum_of_depth_and_color_mapped_source_times
```

因此：

```python
source_monotonic_ns = min(
    depth_source_monotonic_ns,
    color_source_monotonic_ns,
)
```

是 intentional **pair-oldest / worst-case freshness timestamp**，适合用于 pair-level causality 和 conservative freshness，不是当前 bug。

物理语义应区分：

```text
XYZ geometry time    = t_depth
RGB measurement time = t_color
pair freshness time  = min(t_depth_mapped, t_color_mapped)
```

本轮不修改 camera/pointcloud SHM timestamp contract，也不升级 raw/processed schema 来增加 per-stream host-mapped timestamp。

---

### 2.2 `eye_in_hand` 问题纳入本轮 mandatory correctness fix

进一步复核 `data_processing/pipeline.py` 后确认，这个问题不只影响 point-cloud。

当前 `eye_in_hand` 路径使用：

```python
arm_ee = reader.h5f["arm_ee"][source_index]
```

并把 control-grid row 的 EEF pose 直接组合成相机外参。其物理时间语义不成立：

```text
processed RGB camera_extrinsic
    需要 wrist pose at native color exposure time t_color

processed point-cloud XYZ → xarm_base
    需要 wrist pose at native depth exposure time t_depth
```

而 raw v21 当前没有持久化：

```text
depth_source_monotonic_ns
color_source_monotonic_ns
```

这两个 per-stream host-mapped source time，因此无法把 arm pose 精确插值到 `t_depth` / `t_color`。

现有：

```text
arm_ee[source_index]
```

只代表 control-grid row 对应的 robot state，不能作为 wrist-camera exposure-time pose 的替代。

因此本轮应直接采用 **fail closed**：

```text
JOINT profile:
    eye_in_hand 仍允许处理，因为不生成 camera geometry

RGB / POINTCLOUD / RGB_PC profile:
    eye_in_hand 必须在写 processed artifact 前明确拒绝

eye_to_hand:
    完全保持现有行为
```

本轮不尝试实现 pose interpolation，因为当前 raw v21 缺少完成该功能所需的 per-stream host-mapped exposure time。未来若要正式支持 eye-in-hand camera modalities，应先扩展 timestamp contract，再实现：

```text
T_base_eef(t_color)  for RGB camera_extrinsic
T_base_eef(t_depth)  for point-cloud geometry
```

这属于 correctness 修复，不需要升级 processed schema：输出 layout 不变，只拒绝原本会产生错误几何的输入。

---

### 2.3 `wait_return_monotonic_ns` 的测量点是本轮明确 bug

当前 `RealSense.read()`：

```python
frames = self.frame_queue.wait_for_frame(timeout_ms).as_frameset()
wait_return_monotonic_ns = time.monotonic_ns()
```

timestamp 实际发生在：

```text
frame_queue.wait_for_frame()
→ as_frameset()
→ monotonic timestamp
```

而字段名、注释和 raw metadata 都表达“immediately after wait”。

该 timestamp 后续用于：

```text
DeviceClockMapper.host_receive_ns
source_monotonic_ns mapping
backlog / delivery-delay diagnostics
```

因此应修正。

---

### 2.4 raw-v21 semantic metadata 使用了错误的 SDK 接口名称

当前：

```text
camera_wait_return_monotonic_ns_semantics
= host_monotonic_immediately_after_wait_for_frames
```

但 production 使用：

```python
frame_queue.wait_for_frame()
```

不是：

```python
pipeline.wait_for_frames()
```

因此 metadata 与真实 runtime path 不一致，必须修复。

---

### 2.5 `ACTUAL_FPS` 只能作为辅助证据

librealsense public API 当前定义 `ACTUAL_FPS = actual FPS × 1000`，但其历史版本中曾改进过该 metadata 的计算。

L515 环境可能运行旧版 librealsense，因此：

> **Primary evidence = frame number + device timestamp。**

`ACTUAL_FPS` 只作为辅助信号，不能单独确认 AE / sensor cadence。

---

### 2.6 color-only 模式不能假设 queue 返回值一定可直接 `as_frameset()`

RGB-D production path 可以要求 composite frameset。

但 diagnostic 的 color-only control 应同时接受：

```text
composite frameset containing color
or
single color/video frame
```

否则 diagnostic 本身可能受 SDK aggregation behavior 影响。

---

### 2.7 新增 `examples/` 文件必须更新 `repo_map.md`

新增：

```text
examples/diagnose_l515_rgb_timing.py
```

时必须同步更新 `repo_map.md`。

---

## 3. Phase A：确定性修复

### A1. 修正 `wait_return_monotonic_ns` 打点位置

文件：

```text
dexmani_real/sensor/realsense.py
```

当前：

```python
frames = self.frame_queue.wait_for_frame(timeout_ms).as_frameset()
if not frames:
    raise RuntimeError(...)

wait_return_monotonic_ns = time.monotonic_ns()
```

目标：

```python
queued_frame = self.frame_queue.wait_for_frame(timeout_ms)
wait_return_monotonic_ns = time.monotonic_ns()

frames = queued_frame.as_frameset()
if not frames:
    raise RuntimeError(...)
```

使 `wait_return_monotonic_ns` 真正对应 `frame_queue.wait_for_frame()` 返回后的 host monotonic timestamp。

不要顺便修改：

- queue capacity；
- stream FPS；
- `source_monotonic_ns=min(...)`；
- RGB exposure control；
- depth/color pairing；
- skew threshold。

---

### A2. 修正 `camera_wait_return_monotonic_ns_semantics`

文件：

```text
dexmani_real/recording/episode_schema.py
```

当前：

```text
host_monotonic_immediately_after_wait_for_frames
```

建议改为：

```text
host_monotonic_immediately_after_frame_queue_wait_for_frame_return
```

---

### A3. 补充 generic camera health 字段的 stream ownership 语义

当前 production 同时维护 depth/color clock mapper，但以下 generic 字段来自 **depth mapper**：

```text
camera_generation
clock_reset
duplicate
frame_gap
```

本轮不改 schema/dataset 名称，只建议在 `SEMANTIC_META_ATTRS` 增加：

```python
"camera_generation_semantics":
    "depth_stream_clock_mapper_generation",

"camera_clock_reset_semantics":
    "depth_stream_clock_mapper_reset",

"camera_duplicate_semantics":
    "depth_stream_duplicate_detection",

"camera_frame_gap_semantics":
    "depth_stream_frame_number_gap",

"camera_backlog_s_semantics":
    "host_wait_return_minus_pair_oldest_mapped_source_time",
```

这不改变 field value，只消除 contract 歧义。

不要把 `camera_duplicate` 重新定义成 `color repeated`。

---

### A4. `eye_in_hand` processed camera geometry fail closed

文件：

```text
dexmani_real/data_processing/pipeline.py
```

#### A4.1 当前错误路径

当前 `_depth_transform_for_index()` 对 `eye_in_hand`：

```text
camera_T_eef_from_depth
+
arm_ee[source_index]
→ T_xarm_base_from_depth
```

然后该结果被两条路径消费：

```text
RGB profile
→ _color_transform_for_index(...)
→ camera_extrinsic

POINTCLOUD / RGB_PC profile
→ _PointCloudDeriver.derive(...)
→ T_xarm_base_from_depth
```

因此不能只在 point-cloud worker/deriver 层修复。

#### A4.2 新增统一 contract guard

建议新增一个纯 helper，例如：

```python
def _require_supported_processed_camera_pose(reader: EpisodeReader) -> None:
    camera_type = str(reader.h5f["meta"].attrs.get("camera_type", ""))

    if camera_type == "eye_to_hand":
        return

    if camera_type == "eye_in_hand":
        raise ValueError(
            "processed RGB/camera_extrinsic/pointcloud for eye_in_hand requires "
            "arm pose evaluated at native color/depth exposure times; raw v21 "
            "does not persist per-stream host-mapped camera source timestamps"
        )

    raise ValueError(f"unsupported camera_type {camera_type!r}")
```

具体函数名可按 repository style 微调，但错误信息必须说明：

```text
不是“不支持 wrist camera”
而是“当前 raw timing contract 不足以正确生成 exposure-time camera pose”
```

#### A4.3 在 output file 创建前 fail

在 `_write_processed_episode()` 中：

```python
if config.profile.needs_rgb or config.profile.needs_pointcloud:
    _require_supported_processed_camera_pose(reader)
```

这一步必须发生在：

```python
with h5py.File(path, "w") as output:
```

之前。

目的：

- 不创建部分 processed artifact；
- `JOINT` profile 不受影响；
- camera-modal profile 对 eye-in-hand 立即明确失败。

#### A4.4 删除 silent approximation

`_depth_transform_for_index()` 不再允许：

```python
arm_ee[source_index]
```

构造 eye-in-hand transform。

它应只接受当前可严格支持的 `eye_to_hand` static transform，或者复用上述 guard 后返回：

```text
camera_T_xarm_base_from_depth
```

因此原本仅用于这一 approximation 的：

```python
quat_wxyz_to_rotmat
rot6d_to_quat_wxyz
```

如果在 `pipeline.py` 没有其他用途，应删除对应 import。

#### A4.5 不升级 schema

本修复：

```text
raw schema       = v21 不变
processed schema = v4 不变
Policy Zarr      = v2 不变
```

因为没有改变 artifact layout/field semantics，只是禁止生成时间语义不正确的 camera geometry。

#### A4.6 验收矩阵

必须验证：

| camera_type | JOINT | RGB | POINTCLOUD | RGB_PC |
|---|---:|---:|---:|---:|
| eye_to_hand | pass | pass | pass | pass |
| eye_in_hand | pass | **fail closed** | **fail closed** | **fail closed** |
| unknown/invalid | camera geometry profile 时 fail | fail | fail | fail |

对于 eye-in-hand camera-modal profile，异常必须在 processed output file 创建前发生。

---

## 4. Phase A：修复 L515 limitation 文档

文件：

```text
l515_camera_timing_known_limitation.md
```

### A5. `raw v20` → `raw v21`

当前文档仍写 `raw v20`，实际 schema 已是 raw v21，直接修正。

### A6. 明确三类 timestamp

建议写清：

```text
- XYZ geometry 对应 native depth exposure/time；
- projected RGB 对应 native color exposure/time；
- camera pair source_monotonic_ns 是
  min(depth_mapped_source, color_mapped_source)，
  用于 conservative pair freshness / causality；
- pair source_monotonic_ns 不等价于 XYZ 的唯一物理采样时间。
```

### A7. 修改当前状态描述

建议改为：

```text
已记录；RGB 降帧根因待独立诊断。
在诊断结果确认前，不修改 production camera cadence、
exposure/gain、repeated-color publication 或 skew filtering。
```

---

## 5. Phase A：明确旧 `inspect_l515.py` 的职责边界

当前 `examples/inspect_l515.py` 同时做 native RGB-D timing、depth baseline、SciPy filters、SDK pointcloud、optional plane fitting 和可选 frame array storage。

它适合 geometry/depth-quality inspection，但不适合作为精确定位 RGB sensor 为什么只有 16.7 Hz 的唯一工具。

建议只做一个很小的 docstring/focused-doc update：

> `inspect_l515.py` 的 acquisition loop 包含 geometry/depth computation；精确 stream-cadence / RGB FPS root-cause diagnosis 请使用 `diagnose_l515_rgb_timing.py`。

不要为 timing diagnosis 大改该脚本。

---

## 6. Phase A 完成标准

| 项目 | 要求 |
|---|---|
| RGB exposure behavior | 不变 |
| Auto Exposure | 不变 |
| Auto-Exposure Priority | 不变 |
| Gain | 不变 |
| Stream FPS/config | 不变 |
| Camera publication cadence | 不变 |
| Repeated color policy | 不变 |
| `source_monotonic_ns` | 仍为 pair-oldest `min()` |
| SHM schema | 不变 |
| raw schema | 仍为 v21 |
| processed schema | 仍为 v4 |
| pointcloud algorithm | eye_to_hand 数学/采样算法不变 |
| eye_to_hand processed camera profiles | 行为不变 |
| eye_in_hand JOINT profile | 仍允许 |
| eye_in_hand RGB/POINTCLOUD/RGB_PC | 在写 artifact 前 fail closed |
| wait-return measurement | 修正到 queue wait 真正返回后 |
| timing metadata | 与真实 production path 一致 |

---

# 7. Phase B：新增独立 RGB timing diagnostic

## B1. 文件与职责

新增：

```text
examples/diagnose_l515_rgb_timing.py
```

唯一职责：

> **以尽量低的 host-side processing overhead 采集 RealSense stream timing、frame number、per-frame metadata 和 RGB option readback。**

不负责：

- pointcloud；
- depth filtering；
- alignment；
- calibration；
- robot；
- episode recording；
- policy；
- 修改 production config。

---

## 8. Diagnostic side-effect contract

不要称它为绝对“read-only hardware script”，因为 `pipeline.start()` 本身会连接和启动相机。

更准确的表述：

> **non-mutating camera-control diagnostic**

默认允许：

- 连接 RealSense；
- 启动 requested streams；
- `get_option()`；
- 读取 frame/metadata；
- 写本地 diagnostic output；
- 停止 pipeline。

默认禁止：

```python
sensor.set_option(...)
```

尤其禁止修改：

```text
enable_auto_exposure
auto_exposure_priority
exposure
gain
power_line_frequency
global_time_enabled
visual_preset
confidence_threshold
```

---

## 9. CLI 设计

建议：

```text
--serial
--mode {rgbd,color}
--width
--height
--fps
--queue-capacity
--warmup-seconds
--duration-seconds
--timeout-ms
--sample-luma-every
--label
--output-dir
```

默认：

```text
width = 640
height = 480
fps = 30

mode=rgbd:
    queue_capacity = 2

mode=color:
    queue_capacity = 1

warmup_seconds = 10
duration_seconds = 20
timeout_ms = 5000
sample_luma_every = 30 unique color frames
```

所有 resolved settings 写入 `report.json`。

---

## 10. RGB-D 模式：匹配 production stream path，但不写 option

建议：

```python
pipeline = rs.pipeline()
config = rs.config()

config.enable_device(serial)
config.enable_stream(
    rs.stream.depth,
    width,
    height,
    rs.format.z16,
    fps,
)
config.enable_stream(
    rs.stream.color,
    width,
    height,
    rs.format.bgr8,
    fps,
)

queue = rs.frame_queue(queue_capacity)
profile = pipeline.start(config, queue)
```

复现：

```text
Depth = Z16
Color = BGR8
nominal FPS = 30
pipeline + user frame_queue
```

第一轮 diagnostic **不调用** production 的：

```text
set_global_time()
_apply_l515_depth_config()
```

因为它们会写 device option。

因此 report 中应明确：

```text
stream_path = production_profile_matched
camera_options = observed_not_modified
```

不要声称“完全等价 production setup”。

---

## 11. Color-only control

只启用：

```python
config.enable_stream(
    rs.stream.color,
    width,
    height,
    rs.format.bgr8,
    fps,
)
```

用于回答：

> 移除 depth stream 和 RGB-D aggregation 后，color 自身在 high-level pipeline 中是什么 cadence？

---

## 12. Queue 返回值解析必须健壮

RGB-D 模式可要求 composite frameset，并验证 depth/color 都存在。

Color-only 模式应实现 helper：

```text
queued frame
├─ 如果是 frameset → 取 color
└─ 如果是 single color/video frame → 直接使用
```

如果既不是 color frame，也不是包含 color 的 frameset，应明确报错并打印 profile stream type / format。

---

## 13. Capture loop 必须极简

核心：

```python
queued_frame = queue.wait_for_frame(timeout_ms)
host_wait_return_ns = time.monotonic_ns()

# only frame extraction + scalar metadata
```

禁止主 timing loop 做：

```text
rs.align()
rs.pointcloud.calculate()
depth filtering
SciPy
OpenCV encode
MP4 / PNG / JPEG
full RGB/depth ownership copy
plane fitting
large disk write
```

---

## 14. Color sensor option snapshot

不要依赖 `query_sensors()[1]`。

推荐：

```text
device.query_sensors()
→ 遍历 sensor.get_stream_profiles()
→ 找到包含 color stream profile 的 sensor
```

check-then-read：

```text
enable_auto_exposure
auto_exposure_priority
exposure
gain
power_line_frequency

enable_auto_white_balance
white_balance

brightness
contrast
gamma
saturation
sharpness
backlight_compensation
```

每个字段记录：

```text
exposed_by_wrapper
supported_by_sensor
value
range.min
range.max
range.step
range.default
```

至少采：

```text
after pipeline start
after warmup
after capture
```

不要在 capture loop 每帧调用 `sensor.get_option()`。

---

## 15. Per-frame metadata

Color 动态读取：

```text
frame_counter
frame_timestamp
sensor_timestamp
actual_exposure
gain_level
auto_exposure
time_of_arrival
backend_timestamp
actual_fps
exposure_priority
power_line_frequency
```

Depth 至少读取：

```text
frame_counter
frame_timestamp
sensor_timestamp
time_of_arrival
backend_timestamp
actual_fps
```

必须：

```python
frame.supports_frame_metadata(...)
```

之后再读。

unsupported 用统一 NaN/sentinel，不允许让诊断失败。

---

## 16. Metadata 单位必须显式

librealsense public API 当前定义：

```text
actual_exposure    → microseconds
actual_fps         → FPS × 1000
sensor_timestamp   → microseconds
frame_timestamp    → microseconds
backend_timestamp  → microseconds
```

建议保存 normalized field：

```text
color_actual_exposure_us
color_actual_fps_hz
color_sensor_timestamp_us
color_frame_timestamp_us
color_backend_timestamp_us
```

`frame.get_timestamp()` 转换成：

```text
*_device_timestamp_s
```

并单独保存：

```text
*_timestamp_domain
```

---

## 17. 关键指标一：unique stream FPS

不能用 `frameset_count / host_duration` 代表 RGB FPS。

按 frame number 去除 repeated observation 后：

\[
FPS =
\frac{N_{unique}-1}
{t_{last}-t_{first}}
\]

输出：

```text
depth_unique_rate_hz
color_unique_rate_hz
```

并输出 unique-frame interval 的 p50/p95/p99/max。

---

## 18. 关键指标二：frame-number gap

相邻 distinct frame：

\[
\Delta n=n_i-n_{i-1}
\]

分类：

```text
Δn = 1  → contiguous
Δn > 1  → skipped/missing observed frame numbers
Δn <= 0 → reset/rollback/anomaly
```

输出：

```text
depth_frame_gap_event_count
depth_missing_frame_number_total

color_frame_gap_event_count
color_missing_frame_number_total

frame_number_reset_or_rollback_count
```

---

## 19. 关键指标三：normalized per-frame period

定义：

\[
T_{normalized} = \frac{\Delta t_{device}}{\Delta n}
\]

### sensor/cadence 本身变慢的典型形态

```text
frame number:
100, 101, 102, 103

timestamp:
0, 60, 120, 180 ms
```

则：

\[
\Delta n=1,\quad T_{normalized}\approx60ms
\]

### consumer 只观察到部分帧的典型形态

```text
frame number:
100, 102, 104, 106

timestamp:
0, 66, 132, 198 ms
```

则：

\[
\Delta n=2,\quad T_{normalized}\approx33ms
\]

后者更应调查 queue drop / SDK pipeline / host / USB，而不是直接归因 Auto Exposure。

---

## 20. 关键指标四：repeated color

仅对 RGB-D frameset：

```python
color_repeated = (
    current_color_frame_number
    == previous_frameset_color_frame_number
)
```

输出：

```text
frameset_count
depth_unique_count
color_unique_count

color_repeat_count
color_repeat_ratio
```

必须区分：

```text
100,100,101,101
```

和：

```text
100,102,104
```

前者是 repeated observation，后者是 skipped observed frame number。

---

## 21. 关键指标五：RGB-D device timestamp skew

定义：

\[
\Delta t_{rgbd}=t_d-t_c
\]

**只在：**

```text
depth_timestamp_domain == color_timestamp_domain
```

时计算。

若 domain 不同：

```text
rgbd_device_skew_s = NaN
cross_stream_device_timestamp_domains_match = false
```

不要直接相减不同 clock domain。

输出：

```text
rgbd_skew_all_ms:
    count/p50/p95/p99/max

rgbd_skew_new_color_ms:
    count/p50/p95/p99/max
```

---

## 22. `ACTUAL_FPS` 使用规则

若支持，输出 depth/color `actual_fps_hz` 的 p50/p95/p99，但只能作为辅助。

证据优先级：

```text
1. frame number
2. device timestamp
3. normalized period
4. option / actual exposure metadata
5. actual_fps metadata
```

---

## 23. Sparse RGB luminance measurement

真实场地偏暗，需要建立 baseline 亮度统计。

不要每帧 copy RGB。

例如每 30 个 unique color frame 才采一次图像，BGR8 luma：

```text
Y ≈ 0.114 B + 0.587 G + 0.299 R
```

记录：

```text
luma_mean
luma_p05
luma_p50
luma_p95
black_ratio
highlight_clip_ratio
```

heuristic：

```text
black_ratio = fraction(Y < 16)
highlight_clip_ratio = fraction(Y > 245)
```

这些只是 diagnostic heuristic，不是图像质量标准。

---

## 24. 第一版输出文件

```text
diagnostics/<run_name>/
├── report.json
├── frame_timing.npz
└── options.json
```

第一版不强制实现 notification callback。

若后续 frame-number gap 明显，再增加 librealsense debug log / `rs-data-collect` / notifications。

---

## 25. `frame_timing.npz` 推荐字段

```text
host_wait_return_monotonic_ns

depth_frame_number
depth_device_timestamp_s
depth_timestamp_domain

color_frame_number
color_device_timestamp_s
color_timestamp_domain

color_is_repeated
depth_frame_number_delta
color_frame_number_delta

rgbd_device_skew_s

depth_frame_counter
depth_sensor_timestamp_us
depth_actual_fps_hz

color_frame_counter
color_sensor_timestamp_us
color_actual_exposure_us
color_gain_level
color_auto_exposure
color_exposure_priority
color_actual_fps_hz

color_backend_timestamp_us
color_time_of_arrival_us
```

unsupported metadata 用统一 sentinel/NaN，并在 report 中保存 support mask。

---

## 26. `options.json`

建议结构：

```text
device:
    name
    serial
    firmware
    product_line
    usb_type_descriptor
    librealsense_version

active_profiles:
    depth
    color

color_sensor:
    after_start
    after_warmup
    after_capture
```

必须保存 resolved active profile：

```text
width
height
fps
format
stream index
```

---

## 27. `report.json` 核心 summary

至少：

```text
mode
label
requested_fps
queue_capacity
warmup_seconds
capture_duration_seconds

host_output_rate_hz

depth_unique_rate_hz
color_unique_rate_hz

color_repeat_count
color_repeat_ratio

depth_frame_gap_event_count
depth_missing_frame_number_total

color_frame_gap_event_count
color_missing_frame_number_total

color_unique_interval_ms:
    p50/p95/p99/max

color_normalized_period_ms:
    p50/p95/p99/max

rgbd_skew_all_ms:
    count/p50/p95/p99/max

rgbd_skew_new_color_ms:
    count/p50/p95/p99/max

color_actual_exposure_us:
    count/p50/p95/p99/max

color_gain_level:
    count/p50/p95/p99/max

color_actual_fps_hz:
    count/p50/p95/p99

luminance:
    sample_count
    mean
    p05
    p50
    p95
    black_ratio
    highlight_clip_ratio
```

---

## 28. 不自动输出“根因已确认”

第一版只输出 measurements / support flags / evidence flags。

不要自动打印：

```text
ROOT CAUSE = AUTO EXPOSURE
```

更合适：

```text
evidence:
    color_frame_numbers_contiguous: true/false
    color_normalized_period_near_33ms: true/false
    color_normalized_period_near_60ms: true/false
    auto_exposure_enabled: value/unknown
    auto_exposure_priority: value/unknown
    actual_exposure_near_60ms: true/false/unknown
```

最终根因由实验结果 review 判断。

---

# 29. 第一轮人工运行 protocol

## 29.1 环境

在真实偏暗实验场地运行。

保持：

- 正常实验照明；
- 相机位置不变；
- `realsense-viewer` 关闭；
- 不启动其他占用相机的程序；
- 不需要启动 xArm/XHand/Quest；
- 第一轮不改变 AE/exposure/gain。

静态场景即可，本轮先诊断 stream cadence。

---

## 29.2 RGB-D baseline

建议跑 3 次：

```bash
python examples/diagnose_l515_rgb_timing.py \
  --serial f1382055 \
  --mode rgbd \
  --width 640 \
  --height 480 \
  --fps 30 \
  --queue-capacity 2 \
  --warmup-seconds 10 \
  --duration-seconds 20 \
  --label dark_rgbd_baseline_01 \
  --output-dir diagnostics/dark_rgbd_baseline_01
```

随后 `_02`、`_03`。

---

## 29.3 Color-only control

同样跑 3 次：

```bash
python examples/diagnose_l515_rgb_timing.py \
  --serial f1382055 \
  --mode color \
  --width 640 \
  --height 480 \
  --fps 30 \
  --queue-capacity 1 \
  --warmup-seconds 10 \
  --duration-seconds 20 \
  --label dark_color_only_01 \
  --output-dir diagnostics/dark_color_only_01
```

---

# 30. 第一轮结果判别

## Case 1：强烈支持 exposure-limited sensor cadence

同时看到：

```text
color frame number 基本连续：Δn≈1
color normalized period ≈ 60 ms
color unique rate ≈ 16.7 Hz

actual_exposure ≈ 60 ms
Auto Exposure = ON
Auto-Exposure Priority = ON

color-only 仍 ≈ 16.7 Hz
```

则进入第二轮 controlled ablation：

```text
Auto Exposure = ON
Auto-Exposure Priority = OFF
```

仍建议表述为“强烈支持 exposure-limited cadence”，而不是仅凭 option readback 宣布根因。

---

## Case 2：更像 consumer / queue / pipeline / transport 丢帧

如果：

```text
observed unique color interval ≈ 60~67 ms
但 color frame number 经常 Δn≈2
且 normalized period ≈ 33 ms
```

优先调查：

```text
frame_queue drop
pipeline aggregation
host CPU scheduling
USB/driver transport
```

不要先改 Auto-Exposure Priority。

---

## Case 3：RGB-D 16.7 Hz，color-only 30 Hz

优先怀疑：

```text
multi-stream interaction
pipeline synchronization / aggregation
frame borrowing
queue/consumer behavior
```

下一步做 direct color sensor / `rs-data-collect` / pipeline sync diagnostics。

---

## Case 4：RGB-D 和 color-only 都 16.7 Hz

RGB/color-side 原因概率上升。

结合 frame-number continuity、actual exposure、AE state、AE priority、gain、USB evidence 继续判断。

---

## Case 5：frame number 大量跳变

例如：

```text
100, 102, 105, 107...
```

优先进入 host/SDK/USB drop investigation。

建议下一步使用：

```text
rs-data-collect
librealsense debug log
USB controller/cable inspection
```

若出现 `Incomplete Frame`，USB/controller/driver 优先级进一步上升。

---

# 31. 第一轮结果回来前禁止的 production 修改

人工 diagnostic 结果 review 前，不做：

```text
Auto Exposure = OFF

Auto-Exposure Priority = OFF
（它是有价值的第二轮候选，但现在不直接进 production）

manual exposure
manual gain

only publish new-color frames
drop RGB-D pair by skew threshold

change camera worker cadence
change control frequency

change pointcloud source timestamp

change raw schema
change processed schema
change Policy Zarr observation
```

---

# 32. 第二轮预留分支

## Branch A：支持 AE / exposure-limited cadence

优先实验：

```text
Auto Exposure = ON
Auto-Exposure Priority = OFF
```

比较：

```text
RGB unique FPS
RGB-D skew
actual exposure
gain
luminance
black ratio
motion blur
```

若偏暗，再讨论 workspace illumination / gain / exposure budget / explicit lower RGB FPS。

## Branch B：支持 pipeline/queue drop

进一步做：

```text
direct color sensor acquisition
rs-data-collect
queue capacity ablation
CPU/load isolation
```

## Branch C：支持 USB/driver

检查：

```text
USB mode
controller
cable
port
kernel / librealsense backend
transport logs
```

## Branch D：支持 format/conversion path

再单独比较 BGR8 vs YUYV，但不放进第一轮 baseline。

---

# 33. 本轮建议修改文件

```text
dexmani_real/sensor/realsense.py
dexmani_real/recording/episode_schema.py
dexmani_real/data_processing/pipeline.py
l515_camera_timing_known_limitation.md
examples/inspect_l515.py               # 仅职责说明，可选小改
examples/diagnose_l515_rgb_timing.py   # 新增
repo_map.md
```

若新 diagnostic 成为正式支持的 user-facing workflow，再评估 `README.md` 是否增加入口命令。

---

# 34. 明确不修改

本轮不要改：

```text
dexmani_real/sensor/pointcloud.py
dexmani_real/sensor/pointcloud_process.py

dexmani_real/shm/camera_ring.py
dexmani_real/shm/causal_reader.py

dexmani_real/teleop/camera_freshness.py
dexmani_real/teleop/episode_samples.py

dexmani_real/data_processing/zarr_export.py

deployment observation / policy adapter
```

除非实施时发现与本轮确定性修复直接相关的新 bug。

---

# 35. Verification

仓库目前没有通用 unit-test suite；不要把 example program 当测试。

coding agent 在不连接硬件的情况下至少执行：

```bash
git status --short

python -m compileall -q dexmani_real examples

git diff --check
git diff --stat

git diff -- \
  dexmani_real/sensor/realsense.py \
  dexmani_real/recording/episode_schema.py \
  dexmani_real/data_processing/pipeline.py \
  l515_camera_timing_known_limitation.md \
  examples/inspect_l515.py \
  examples/diagnose_l515_rgb_timing.py \
  repo_map.md
```

纯 offline review：

```text
- CLI import 不应在 import-time 连接 RealSense；
- constructor 不打开硬件；
- 只有 main execution path start pipeline；
- finally 中 stop pipeline；
- option query 先 check support；
- diagnostic 默认没有 sensor.set_option；
- timing loop 无 full-frame copy / pointcloud / image encoding；
- cross-stream skew 仅在 timestamp domain 相同时计算；
- color-only 不假设 frameset；
- actual_fps 已做 ×1/1000 归一化；
- output field name 带单位；
- repo_map 已加入新文件；
- eye_to_hand camera-modal processing 仍走原 static transform；
- eye_in_hand + JOINT 不触发 camera-pose guard；
- eye_in_hand + RGB/POINTCLOUD/RGB_PC 在 output file 创建前 fail closed；
- pipeline.py 不再用 control-grid `arm_ee[source_index]` 合成 wrist-camera exposure-time transform。
```

**不要由 coding agent 自动运行硬件 diagnostic。**

硬件运行由人工后续执行。

---

# 36. Handoff：人工运行后提供什么

优先提供每组：

```text
report.json
options.json
```

需要进一步分析时再提供：

```text
frame_timing.npz
```

建议至少：

```text
3 × RGB-D baseline
3 × color-only baseline
```

之后再按：

```text
Observation
→ Root-cause hypothesis update
→ Controlled ablation
→ Production fix
→ Validation
```

推进。

---

# 37. 最终实施顺序

```text
Step 1
修 wait_return timestamp measurement point

Step 2
修 raw-v21 timing semantic metadata

Step 3
修 eye_in_hand processed camera geometry：
JOINT 保留，RGB/POINTCLOUD/RGB_PC fail closed

Step 4
修 L515 limitation 文档的 v20/v21 与 timestamp 语义

Step 5
新增 diagnose_l515_rgb_timing.py
默认不写 camera option

Step 6
更新 repo_map.md

Step 7
只做 offline compile/diff verification，
并验证 eye_in_hand fail-closed matrix

Step 8
人工真实暗光场地运行：
3× RGB-D + 3× color-only

Step 9
review report.json/options.json

Step 10
只有证据支持后，才进入 AE Priority / exposure / gain /
pipeline / USB / format 的单变量修复实验
```

---

# 38. 核心结论

当前最合理的工程策略不是：

```text
“假设 Auto Exposure 是根因，然后直接关闭”
```

而是：

```text
修掉已确定的 timestamp/metadata bug
        ↓
建立低开销、non-mutating camera-control timing diagnostic
        ↓
用 frame number + device timestamp 判断
sensor cadence 还是 observed-frame loss
        ↓
再结合 exposure / AE Priority / gain / color-only control
        ↓
决定第二轮 production fix
```

这样可避免把可能属于 exposure、pipeline sync、frame_queue、USB/driver 或 format conversion 的问题提前固定成错误根因。

---

# References

## dexmani_real

- Baseline commit:  
  https://github.com/haoyangzhanglab/dexmani_real/commit/e28aa3e72646e12e065050ec24bbc750033506a1
- RealSense driver:  
  https://github.com/haoyangzhanglab/dexmani_real/blob/e28aa3e72646e12e065050ec24bbc750033506a1/dexmani_real/sensor/realsense.py
- Raw v21 schema:  
  https://github.com/haoyangzhanglab/dexmani_real/blob/e28aa3e72646e12e065050ec24bbc750033506a1/dexmani_real/recording/episode_schema.py
- Processed pipeline / eye-in-hand guard owner:  
  https://github.com/haoyangzhanglab/dexmani_real/blob/e28aa3e72646e12e065050ec24bbc750033506a1/dexmani_real/data_processing/pipeline.py
- Existing L515 limitation note:  
  https://github.com/haoyangzhanglab/dexmani_real/blob/e28aa3e72646e12e065050ec24bbc750033506a1/l515_camera_timing_known_limitation.md
- Agent engineering contract:  
  https://github.com/haoyangzhanglab/dexmani_real/blob/e28aa3e72646e12e065050ec24bbc750033506a1/AGENTS.md

## librealsense

- Frame metadata:  
  https://github.com/realsenseai/librealsense/blob/master/doc/frame_metadata.md
- Public metadata enum / units:  
  https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/h/rs_frame.h
- Frame management / frame queue:  
  https://github.com/realsenseai/librealsense/blob/master/doc/frame_lifetime.md
- Frame buffering management:  
  https://github.com/realsenseai/librealsense/wiki/Frame-Buffering-Management-in-RealSense-SDK-2.0
- Exposure vs FPS discussion:  
  https://github.com/realsenseai/librealsense/issues/1957
- Pipeline missing-frame / previous viable frame discussion:  
  https://github.com/realsenseai/librealsense/issues/5675

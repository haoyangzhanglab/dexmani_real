# 修复方案：`examples/` 目录 Code Review

- **日期**：2026-08-15
- **对象**：`docs/code-review-examples-2026-08-15.md` 的 24 CONFIRMED（2 high / 9 medium / 13 low）+ 1 PLAUSIBLE
- **范围**：`examples/` 全部 9 个入口脚本；不触碰 13 条 REFUTED
- **产出**：本方案文档（未改动任何代码、未运行硬件）

> **✅ 已实施（2026-08-15）**：本方案已分 6 阶段落地（10 文件，372+ / 377−），并通过一轮多智能体
> 对抗性复审（7 review + 逐条 verify）。复审确认 **7 条新发现（1 high / 2 medium / 4 low）已全部修复**、
> 2 条被 refute。详见文末 [「实施状态与修正」](#实施状态与修正2026-08-15)。

---

## 三处跨文件主题（报告 §"跨文件模式"）

1. **硬件生命周期缺 `try/finally`** → `keyboard_teleop.py:754-819` 是正确的参照范式，统一推广。
2. **标定写入缺确认/备份** → `calibrate_camera.py:520-525` 的 `shutil.copy2` 备份是正确范式，统一推广。
3. **provenance 声称校验但未真校验** → `replay_episode.py` 的 config SHA 只做格式校验、`--acc/--joint-speed` 覆盖形同虚设。

**关键基础设施（已核对）**：
- `shutdown_processes` → `runtime/processes.py:139 shutdown_processes_verified`：写 `is_running=False` → join/escalate → `_close_shared_storage(shared)` → `shared.close()`（`processes.py:95-101`），返回 `ShutdownReport`。
- `SharedStorage.create()` / `.close()`：`shm/shared_storage.py:275/414`，`close()` 幂等、best-effort unlink POSIX 段。
- `validate_arm_feedback`：`teleop/keyboard.py:573-610`，校验 connected/state_valid/源时间戳/新鲜度/形状/有限性。
- `atomic_json_dump`：`recording/transaction.py:51-76`，mkstemp→dump→fsync→os.replace→fsync(parent)，原子覆写。
- `resolve_runtime_config`：`config/runtime.py:216-279`，产出 `canonical_json`（**全部** validated section）+ `sha256`。

---

## ⭐ 关键发现：replay provenance 比报告所述更糟——它读的是从未写入的元数据键

深入追踪 `resolved_config_sha256` 的写入/读取链后发现一个报告未识别、但**决定 M2/M3 修法**的根因：

**`episode_recorder.py` 的 `_write_meta_attrs`（L282-317）只写 `resolved_config_sha256`，从不写
`hand_available` / `joint_max_acc` / `joint_max_speed` / `arm_loop_hz` / `jerk_management` 这 5 个键。**
（`rg` 全库核对：这 5 个键在 `episode_recorder.py` 与 `collect_teleop.py` 中**零写入**，只作为 `replay_episode.py` 的**读取**出现。）

因此任何真实 v16 episode 上，`load_trajectory`（L291-301 用 `_optional_*` 读缺键）会得到：
`trajectory.joint_max_acc / joint_max_speed / arm_loop_hz / jerk_management / hand_available` **恒为 `None`**。后果：

| 读取点 | 当前行为（缺键 = None） |
|---|---|
| `_verify_trajectory_provenance` 完整性检查 L798-807 | `missing` 恒含 4 项 → **每次 live replay 都 raise** |
| `require_explicit_hand_mode` L778-780 | `hand_available is not True` 恒真 → **非 `--no-hand` 每次 raise** |
| `TrajectoryData.has_hand` L232-235 | `hand_available is True` 恒假 → **live replay 永不 spawn 手 worker / 不回放手动作流**（L1034 / L1734） |

**结论**：当前 live replay 在到达 provenance 之前就被 `require_explicit_hand_mode` 拒，是 **100% 不可用**。
报告 M2 把控制器参数（acc/speed/jerk）当成"已存在的相等校验"——实际它们读缺键，不是"不充分"而是"恒失败"。

**这个发现决定两件事**：
1. **M2/M3 的正确修法不是"从 episode 元数据重建控制器参数再比较"**（那些元数据根本不存在），而是
   **`resolved_config_sha256` 与 replay 侧 `base_runtime.sha256` 相等比较**——唯一完整、始终存在的 config 指纹。
2. **手模式证明不能依赖 `hand_available` 键**，必须由 config 哈希（`policy.hand_enabled` 已含于 canonical_json）承载，
   replay 侧用 `--no-hand` 重建该字段即可。

---

## 四、`examples/replay_episode.py`（核心，联合修复 M2+M3 + 同一根因的手模式链路）

### 修复目标（一句话）

用 **`resolved_config_sha256` == replay 侧重建的 recording-equivalent config 哈希** 作为唯一 config provenance，
删除读缺键的控制器参数比较与死代码 `replay_runtime_hash`，并修掉被缺键 `hand_available` 破坏的手模式判定链，
使 live replay 端到端可跑、且 fail-closed 于 workspace/table/IK/模型等任何真实改动。

### 录制侧哈希的精确语义（重建依据，已核对 `collect_teleop.py:382-390`）

录制时 `resolved_config_sha256 = runtime.sha256`，其中（键为**完整字段名**的 dotted string，非 `arm.max_acc`/`arm.max_speed` 缩写）：
```
cli_overrides = {arm.max_joint_acceleration_deg_per_s2: args.acc(=None 常态),
                 arm.max_joint_velocity_deg_per_s: args.speed(=None),
                 policy.hand_enabled: False if --no-hand else None,
                 policy.recording_enabled: False if --no-record else None}
```
正常录制（有手、无 acc/speed 覆盖、有录制）下，四个覆盖全为 `None` → **录制哈希 ==
`resolve_runtime_config(yaml_path=config).sha256` == replay 侧 `base_runtime.sha256`**。
唯一使录制哈希偏离 config 文件、且 replay 可控的字段是 `policy.hand_enabled`（`--no-hand` 时置 False）。

### 具体改动

**A. 删除 `replay_runtime_hash`（L161-191）** —— 死/错实现：哈希 `canonical_yaml` + replay-only 字段
（source/speed/no_hand/jerk），与录制侧 `canonical_json` 哈希**永远对不上**；L855-861 调用结果被丢弃、main 仅打印。

**B. `_resolve_replay_runtime`（L1874-1914）—— 计算 recording-equivalent config 哈希**
把 L1907-1913 的 `config_sha256 = replay_runtime_hash(...)` 替换为：
```python
config_sha256 = runtime.sha256
```
`ReplayRuntimeSelection.config_sha256` 语义从"yaml+replay 哈希"改为"recording-equivalent config 哈希"。

> **修正（复审后）**：原稿用 `config_sha256 = provenance_runtime.sha256`（`base_runtime` 或 `--no-hand`
> 时 `hand_enabled=False` 的重建），把 `--acc/--joint-speed/--arm-ip` 排除在 provenance 外。复审发现这
> 与录制侧不一致：`collect_teleop` 把 `--acc/--speed/--no-hand` 覆盖烙进其 `runtime.sha256`，故
> (a) 用覆盖录制的 episode 永远无法回放、(b) 回放侧覆盖与录制不一致时**假通过**（provenance 验的
> config 与真实运行的 config 不同）。最终改为对真实回放 `runtime.sha256` 哈希：`--acc/--joint-speed/
> --no-hand/--arm-ip` 现在**参与** provenance（覆盖与录制一致才放行）。

**C. `_verify_trajectory_provenance`（L785-834）—— 用 config 哈希相等比较替换读缺键的控制器块**
- 保留 `_is_sha256` 格式预检（L795-796，给损坏 episode 更清晰报错）。
- **删除** L798-821 整块（`required_controller` 完整性 + `np.isclose` + `jerk_management != "unmanaged"`）——它们读恒为 `None` 的缺键，且是 M3 的病根。
- 新增相等比较：
  ```python
  if trajectory.resolved_config_sha256 != provenance_sha256:
      raise ValueError("live replay config provenance mismatch: recorded resolved config differs from replay config")
  ```
- 保留 action_source / num_frames / arm_actions 形状 / model 哈希（L787-794、L823-833）不变。
- 签名改为 `_verify_trajectory_provenance(trajectory, runtime, *, provenance_sha256: str)`（唯一调用点 L854 同步改 keyword 传参，见 F）。

**D. `require_explicit_hand_mode`（L774-782）—— 删除读缺键的 `hand_available` 检查**
删 L778-780（`if trajectory.hand_available is not True: raise "missing hand_available metadata"`）；
保留 L781-782 的 `has_hand_actions`（数据集存在性）检查。手模式（录制时 `hand_enabled`）已由 C 步的 config 哈希相等比较承载：
无手 episode + 非 `--no-hand` replay → `hand_enabled` 不一致 → config 哈希不匹配 → 被拒（与既有 `--no-hand` 语义一致）。

**E. `TrajectoryData.has_hand`（L232-235）—— 改为按数据集存在性判定**
`return self.hand_available is True and self.action_hand_joint is not None` →
`return self.action_hand_joint is not None`。消费点 L1034/L1734（`trajectory.has_hand and not no_hand`）随之变为
"有手动作流且未 `--no-hand`"→ 正确 spawn 手 worker 并回放手动作流。手数据完整性/新鲜度由现有 worker 健康门 + config 哈希兜底。
（改后 `has_hand` 与 `has_hand_actions` 语义重复，可让 `has_hand` 委托 `has_hand_actions` 或删其一；`load_trajectory` L381 三元里的
`"dataset-only"` 分支随之不可达。）

**F. 线程传递 `config_sha256`（含两处必须一并改的调用点，否则 `TypeError`）**
`LiveReplayConfig`（L1447）新增 `config_sha256: str` 字段 → `main` 构造处（L2115-2121，`run_live_replay(...)` 调用内）传
`selection.config_sha256` → `run_live_replay` → `verify_live_replay_preflight(..., provenance_sha256=config.config_sha256)` → C 步函数。
**注意**：C 步把 `_verify_trajectory_provenance` 改成 `*, provenance_sha256` 后，它唯一调用点 L854 是位置传参
`_verify_trajectory_provenance(trajectory, runtime)`，须同步改为
`_verify_trajectory_provenance(trajectory, runtime, provenance_sha256=provenance_sha256)`；且 `verify_live_replay_preflight`
（L837-843，现为 `*, no_hand, speed_factor`）须在 keyword-only 参数里加 `provenance_sha256: str`。

**G. 删除 `verify_live_replay_preflight` 内死调用（L855-861）** —— `replay_runtime_hash(...)` 结果被丢弃，随 A 一并删。
（删后 `action_source` 局部变量（L854）与 `_verify_trajectory_provenance` 的返回值（L834 `return trajectory.action_source`）变死代码，
可让函数改返回 `None` 或保留原样。）

**H. 清理残留死字段/死回退（保持全局一致，可选但推荐）** —— A-G 修完后，`TrajectoryData` 上 5 个缺键字段
（`hand_available` L223、`joint_max_acc`/`joint_max_speed`/`arm_loop_hz` L224-226、`jerk_management` L227）与
`_resolve_replay_runtime` 里 `trajectory.joint_max_acc`/`joint_max_speed` 回退分支（L1880-1882、L1891-1893，恒走 config 默认）
都成了死代码。删除这些字段与其 `load_trajectory` 读取（L291-301）与构造（L367-372）；L2085 的
`hand_metadata` 打印改从 `traj.has_hand_actions` 派生（"yes"/"no"），不再读恒为 None 的 `hand_available`。

### 行为变化（fail-closed 收紧，符合报告预期）
- workspace / table（含解析后的 `plane_abcd`）/ IK 阈值 / loop_hz / 手模式等任何 config 文件改动 → 拒绝 live replay。
- `--acc` / `--joint-speed` / `--arm-ip` 现在**参与** provenance：回放须传与录制一致的覆盖（覆盖匹配才放行），否则 config 哈希不等 → fail-closed（修正了原"replay-only、不参与"的设计，见 B 步修正）。
- `--no-hand` 只对"无手录制"的 episode 放行（与既有 `--no-hand` 语义一致），反之被 config 哈希拒绝。
- **前提**：replay 须传与录制相同的 `--config` YAML 及相同的 `--acc/--joint-speed/--no-hand`（常态下两者都用默认值，即匹配），否则 config 哈希不等 → fail-closed。

### 可选替代（默认不做）
- **录制侧补写元数据**：在 `episode_recorder._write_meta_attrs` 增加 `hand_available`/`joint_max_acc`/`joint_max_speed`/`arm_loop_hz`/`jerk_management`（schema v16 加法式、向后兼容），使 replay 可重建录制时的 acc/speed 覆盖。代价是触及 `dexmani_real/recording/` 与 v16 契约，超出本报告 `examples/` 范围。

### L1（L686）— `np.savez_compressed` / `json.dump` 非原子结果写
- `save_replay_data`（L681-688）：先写 `npz_path.with_suffix(".npz.tmp")` 再 `os.replace` 到 `npz_path`。
- `save_results`（L740-743）：`json.dump` 改用 `from dexmani_real.recording.transaction import atomic_json_dump`。

### L2（L1269）— `previous_hand_cmd` 只写不读 dead store
删除两处赋值（L1269、L1367）及变量声明。

### 复查·send_mask（L337）— `send_mask` 未纳入 shape 校验
`load_trajectory` 的 shape 校验 `arrays` dict（L342-354）中，当 `send_mask is not None` 时追加
`("send mask", (send_mask, (total_frames,)))`，`flag_action_queued` 帧数不足时在进入 `_replay_live` 前即 fail。

---

## 一、`examples/calibrate_camera.py`

### H1（L1073）— `main()` 无 `try/finally`，异常时臂 worker 变 Mode 6 孤儿 + shm 泄漏
现状：`exit_code = _run_calibration(...)`（L1073）后，清理逻辑（L1078-1094）只在正常返回路径执行；
`shared.close()` 从未被直接调用（`shutdown_processes` 内部会 close，但异常时 `shutdown_processes` 也被跳过）。
`_run_calibration` 内部 `try` 从 L769 才开始，此前 `_start_camera`/`keys.start()`/`compute_eef_pose_world`/`cv2.namedWindow` 均可能抛异常。

修复：把 `_run_calibration` 调用放进 `try/finally`，`finally` 内做与 `keyboard_teleop.py:774-809` 一致的清理。
```python
exit_code = 1
try:
    exit_code = _run_calibration(shared, runtime, planner, safety_gate, workspace,
                                 arm_process, args.serial, calib_cfg, aruco_cfg)
finally:
    started = [p for p in processes if p.pid is not None]
    if started:
        try:
            clean_exit = exit_code == 0
            report = shutdown_processes(shared, started,
                                        graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                                        disarm_if_clean=clean_exit)
            if clean_exit and not report.clean:
                logger.error("verified shutdown invalidated the clean control exit: %s", report)
                exit_code = 1
        except RuntimeError:
            logger.critical("child process remains alive; leaving SharedStorage linked", exc_info=True)
            exit_code = 1
    else:
        try:
            if not shared.close():
                _set_fault(shared, "SharedStorage cleanup was incomplete"); exit_code = 1
        except Exception:
            _set_fault(shared, "SharedStorage cleanup failed"); exit_code = 1
print(f"  calibration session exit code: {exit_code}")
return exit_code
```
早退路径（L1058-1067 的 `shutdown_processes(shared, processes); return 1`）在 try 之前，保持不变（不会双关）。

### M5（L725）— `_event_solve` 不捕获异常，一次坏 ENTER 中断整个交互会话
`_calibrate_and_select` 退化几何抛 `RuntimeError("all hand-eye methods failed")`（L396）、`_save_cameras_json`
对损坏 JSON 抛 `JSONDecodeError`（L504），均逃逸到主循环 `try`（仅 `except KeyboardInterrupt`）。

修复：仿照 `_event_capture`（L693-706）的优雅 catch-and-report，把 `_event_solve` 内
`_calibrate_and_select(...)` 与 `_save_cameras_json(...)` 调用包进 `try/except Exception`：
`logger.warning("solve failed", exc_info=True)` + `print(f"FAILED — {exc}, skipped")` + `return`，不写、不 ACCEPTED。

### M6（L708）— `_event_capture` 臂反馈校验过弱（只查 `np.isfinite(eef_pos)`）
主循环（L825-858）用 `validate_arm_feedback` + `error_code` 门；`_event_capture` 只查 eef_pos 有限性，
会拿陈旧臂姿态配对新 marker，污染手眼求解。

修复：把 `_event_capture` 里 L708-711 的弱检查替换为主循环同款 `validate_arm_feedback(...)` 调用
（`policy.arm_state_stale_threshold_s` 已在作用域内；`validate_arm_feedback` 已 import，L81），
`feedback_issue is not None` 时 `print("FAILED — ... skipped"); return`。`_event_capture` 是采样而非运动，
不必加 `error_code` 门（保持最小改动，仅补齐 freshness/connected/state_valid 语义）。

---

## 二、`examples/pointcloud_process_example.py`

### H2（L367/370）— 静默无备份覆写生产标定 `desk_plane.json`
现状：`main()`（L541，`def main() -> None`，**无 argparse**）无条件调 `_calibrate_desk`（L566），
内部 L370 直接 `save_desk_plane`（→ `atomic_json_dump` 原子覆写 `dexmani_real/config/desk_plane.json`），
无确认、无备份；该文件被 `PointCloudProcessor.__init__`（`pointcloud_processor.py:176-179`）自动加载，是碰撞/homing/replay 共用输入。

修复（本文件无 argparse，保持交互风格）：
1. `_calibrate_desk` 内，`save_desk_plane` 之前加交互确认：`input("Overwrite dexmani_real/config/desk_plane.json? [y/N] ")`，非 y 则跳过写（仍打印拟合平面）。
2. 写前用 `shutil.copy2` 备份（仿 `calibrate_camera.py:520-525` 的 `.bak.<时间戳>` 命名），`import shutil`。
   （若倾向非交互，可改用 `--write-desk-plane` flag —— 需为此文件新增 argparse；默认走交互确认，最小改动。）

### M8（L316）— `_run_2d_filters` 声称 "byte-identical" 却跳过 medianBlur + NaN 恢复
生产 `process()`（`pointcloud_processor.py:211-225`）先 medianBlur（`depth_median_enabled=True` 默认）再恢复 NaN；
诊断对原始 depth 直接跑 GaussianBlur/Laplacian，无效边界巨大梯度被误判为边缘，边缘移除计数高估。

修复（DRY + 保证一致）：把 median+NaN 恢复提取为可复用静态方法
`PointCloudProcessor.apply_depth_median(depth_m, enabled)`（`pointcloud_processor.py` 内），
`process()` 的 L217-225 改调它，`_run_2d_filters` 在 LoG 之前也调它（`cfg.depth_median_enabled` 门控）。
同时修正 L315-316 误导性状态行（打印 "median filter: ON"，但实际未应用）——字面 "byte-identical" 其实在 L249（edge filter 注释），二者是同一错误的两种表述，都改为如实反映诊断实际是否应用了 median。

### L9（L440）— 单帧计时读私有 `_t_*`，`process()` 早退 None 时各阶段显示 0/"disabled"
`process()` 多个早退 `return None` 在 `_t_* +=`（`pointcloud_processor.py:431-441`）之前，`pipeline_total` 却有真实时长。

修复：`_run_pipeline` 里 `result = processor.process(...)` 之后判 `if result is None`：
打印 "process() early-returned (no pointcloud); per-stage timing unavailable"，并跳过/标记各阶段分解，
避免 `_tprint`（L116-125）把 0 显示成 "(disabled)" 造成误导。

---

## 三、`examples/calibrate_vr_heading.py`

### M4（L253）— `SharedStorage.create()` 后无 `try/finally`
settle/countdown/collect（L270-324）期间异常逃逸，`shared.close()` 只在 happy path（L332）和 `_fatal_exit`（L219）执行。

修复：把 settle/countdown/collect 包进 `try/finally`，`finally` 复用现有 shutdown 逻辑
（L326-332：`shared.is_running.value=False` → `vr_proc.join` → `terminate` → `shared.close()`），
并 `print` 保留 `_fatal_exit` 语义。

### M9（L391）— poor 档 transform 照样覆写 `vr_transform.json`，砖掉 teleop
`_quality_grade` 返回 `grade`；L381-392 无条件 `atomic_json_dump` 覆写 `_OUTPUT_PATH`，
而运行时 `load_vr_transform(reject_poor=True)`（`teleop/vr_transform.py:102-159`）会拒绝 poor 档。

修复：
1. 写前若 `quality["grade"] == "poor"`，默认拒绝写（打印 "poor quality, not written; re-collect or use --force"），
   仅当传入 `--force` flag 才写。
2. 任何写入前用 `shutil.copy2` 备份 `_OUTPUT_PATH`（仿 `calibrate_camera`），`import shutil`。

### L9（L377）— sanity check `corrected[0] < 0.98` 恒假（死代码）
`T=R_z(-θ)` 与 `mean_fwd` 由同一 θ 构造，`T@mean_fwd` 恒 `[norm, ~0]`，`corrected[0]` 恒 ≈1。

修复：删除该恒假分支（warning 在 L378）及对应 `sanity_min_corrected_x` 配置字段（L90；dataclass 声明 L77-78，`__post_init__` L92-98），
`_quality_grade` 的 `std_deg` 已足够表达对齐质量；保留 `corrected` 计算仅用于打印 "T·forward"。

### L11（L232）— `--duration` 绕过 `HeadingCalibrationConfig.__post_init__`
CLI `--duration`（L230-240）直接 `time.monotonic() + args.duration`（L286），`__post_init__`（L92-98）只校验默认值。

修复：`args = parser.parse_args()` 后显式校验：
```python
if not math.isfinite(args.duration) or args.duration <= 0:
    parser.error("--duration must be a positive finite number of seconds")
```
（`import math`；与 `replay_episode.py` 的 `_positive_finite_float` 思路一致。）

---

## 五、`examples/visualize_episode.py`

### M7（L682）— `EpisodeVisualizer` 在 try/finally 之外构造
`__init__`（L161-205）先 `self._reader = EpisodeReader(...)`（打开 3 h5py + VideoDecoder，L172），
其后 `read_camera_all` 帧数不匹配（`episode_reader.py:465-469`）或缺 `depth_scale`（L203）抛 `ValueError`；
`viz.close()` 只包住 `log_step` 循环。

修复（在 `__init__` 内自清理，最稳）：`self._reader = EpisodeReader(h5_path)` 之后，把 `__init__` 剩余体
包进 `try: ... except BaseException: self.close(); raise`。`close()`（L634-643）已幂等
（`hasattr(self, "_reader")` 守卫）。这样构造失败不再泄漏，`main()` 现有 try/finally 无需改动。

### L5（L544）— `/pointcloud` 无条件渲染全零占位帧
修复：`_log_camera` 的预计算分支（L542-547）渲染前读 `flag_pointcloud_valid`（已预加载进 `self._state`）：
`valid = self._state.get("flag_pointcloud_valid")`；`valid is None or bool(valid[step_idx])` 时才 `rr.log`，
否则跳过（不画原点黑团）。

### L6（L492）— 把 "SDK-scaled, physical unit unverified" 的派生序列标成 "(N)" 牛顿
修复：`_log_static` 的 `_force_series`（L490-495）标签去掉 "(N)"，改为中性单位
如 `"(SDK-scaled)"`（与 `robot/types.py:53` 措辞一致），三轴标签相应改为 `Fx/Fy/Fz` 不带单位。

---

## 六、其余 Low（单文件、小改动）

- **`collect_teleop.py:577`**：`if group is None and not shared_closed:` → 删掉 `and not shared_closed`
  （`shared_closed` 只可能经 `group.shutdown()` 变真，届时 `group` 必非 None）。
- **`keyboard_teleop.py:661`**：删除不可达的 "XHand 超时不确定态" 守卫块（L661-664）；
  `hand_enabled=True` 时 `is_ready("hand")` 已单调为真，`not is_ready("hand")` 恒假。
- **`realsense_record_example.py:621`**：重试循环改为真实退避（`delay = 1.0*(attempt+1)` 即 1s/2s），
  且仅 `attempt < 2` 时打印 "retrying"（最后迭代不 retry 不打印）。
- **`xhand_control_example.py:191`**：`read_state` 按 `finger_id` 选 sensor —— 加
  `_FINGERTIP_TO_SENSOR_IDX = {2:0, 5:1, 7:2, 9:3, 11:4}`，`sensor = state.sensor_data[_FINGERTIP_TO_SENSOR_IDX[finger_id]]`。
- **`xhand_control_example.py:259`**：RS485 分支捕获枚举结果：`ports = xhand_exam.enumerate_devices("RS485")`，
  无 ports 则报错返回；否则 `open_device("RS485", ports[0])`（对齐 EtherCAT 用 `ethercat_ports[0]`）。
- **`xhand_control_example.py:164`**：`print(f"  serial_number: {''.join(info.serial_number[:16])}")`
  （`DeviceInfo_t.serial_number` 是 32 个单字符 str 的 list，直接打印是 list repr；`''.join` 是正确修法。注：仓库并无现成 `''.join`
  可"对齐"——生产 `robot/xhand.py:718-720` 走 `get_serial_number`（返回 str），不读该 list 字段。）

### PLAUSIBLE（需真机确认，防御性修复）
- **`realsense_record_example.py:451`**：点云生成 `except ValueError` 扩为 `except (ValueError, RuntimeError)`
  （open3d `voxel_down_sample`/pytorch3d `sample_farthest_points` 可能抛 `RuntimeError`），异常时跳过并继续汇总/`viewer.close()`。

---

## 不做（明确排除）

- 13 条 **REFUTED**：不改（含复查撤回的 Medium #1 `collect_teleop.py:574`、Low #4 `keyboard_teleop.py:669`）。
- **覆盖缺口（未升级为 finding）**：`visualize_episode.py --info` 空 HDF5 的 `keys[0]`/零除、
  `calibrate_camera.py` 备份时间戳秒级并发覆盖 —— 可选，默认不在本轮。
- **录制侧补写元数据**（见"四·可选替代"）：默认不改 `episode_recorder` / v16 契约；手模式与控制器参数 provenance 全部在 replay 侧用 config 哈希解决。

---

## 验证（未运行硬件）

1. **编译**：`conda run -n real_robot python -m compileall -q dexmani_real examples`。
2. **静态核对**：`git status --short` + `rg` 核对每个改动点的引用路径/符号仍存在（尤其 `replay_episode.py` 内
   `replay_runtime_hash` 已无引用、`hand_available`/`joint_max_*` 只读字段的消费点已更新）。
3. **离线可跑路径**：
   - `python examples/replay_episode.py <episode_dir> --dry-run`（走 `_resolve_replay_runtime` + `validate_offline`，
     验证 `Replay config:` 打印的是 recording-equivalent config 哈希，非 replay_runtime_hash）。
   - `python examples/visualize_episode.py <episode_dir>`（离线 Rerun，验证 M7/L5/L6）。
4. **专项自检（离线）**：
   - **replay provenance**：对一个真实 episode 目录，`--dry-run` 后确认 config 哈希 = `resolve_runtime_config(yaml).sha256`
     （有手录制时）；手工篡改 episode 的 `resolved_config_sha256`（或换一个 workspace/table 不同的 config 文件）验证相等比较 fail-closed。
   - **手模式**：`--no-hand` 下 `_resolve_replay_runtime` 应重建 `hand_enabled=False`；非 `--no-hand` 且 episode 有手数据时
     `require_explicit_hand_mode`/`has_hand` 不再因缺 `hand_available` 抛错。
   - `pointcloud`/`vr_heading` 的写入门控：仅静态确认 `input()`/`--force`/`shutil.copy2` 逻辑与 import。
5. **不运行**：任何 `--live` 运动命令、标定写入、SDK/相机硬件调用。真机验收项单独列出：
   - `calibrate_camera` 无相机场景的挂死/泄漏路径；
   - `calibrate_vr_heading` poor 档拒绝 + `--force` 路径；
   - `xhand_control_example` 的 RS485 端口枚举与 serial 打印（需 SDK）；
   - `realsense_record_example` open3d `RuntimeError` 是否真触发（PLAUSIBLE）；
   - **replay live 端到端**：一个真实 episode 的 `--live`（有手 / `--no-hand` 各一次），确认手 worker spawn 与手动作流回放恢复正常。

---

## 实施状态与修正（2026-08-15）

本方案已分 6 阶段实施，改动 **10 个文件**（`dexmani_real/sensor/pointcloud_processor.py` + `examples/` 9 个入口脚本）。
实施后跑了一轮多智能体对抗性复审（7 个 review agent + 逐条 verify），在方案之外**新确认 7 条缺陷**，
全部修复；另有 2 条被 refute。

### 复审确认并已修复

| # | 级别 | 位置 | 问题 | 修复 |
|---|---|---|---|---|
| 1 | high | `replay_episode.py:save_replay_data` | `np.savez_compressed` 会对不 `endswith(".npz")` 的文件名自动追加 `.npz`，原 `with_suffix(".npz.tmp")` 实际写到 `replay_data.npz.tmp.npz`，随后 `os.replace` 恒 `FileNotFoundError` → 每次 live replay 到结果持久化都崩 | `tmp_path = npz_path.with_suffix(".tmp.npz")`（已用临时目录实测通过） |
| 2 | medium | `replay_episode.py:_resolve_replay_runtime` | provenance 用 `base_runtime.sha256`（排除 `--acc/--joint-speed/--arm-ip`），与录制侧把覆盖烙进 `runtime.sha256` 不一致：覆盖录制的 episode 无法回放，且回放侧覆盖不同时假通过 | `config_sha256 = runtime.sha256`（真实回放 runtime） |
| 3 | low | `replay_episode.py:--joint-speed/--acc` | 帮助文本残留已删的 "episode metadata fallback" 措辞 | 改为 "defaults to the runtime config value" |
| 4 | low | `pointcloud_process_example.py:_run_pipeline` | 注释声称早退时各阶段显示 "unavailable"，实际摘要仍显示 "(disabled)" | 早退时各 3-D 阶段以 `NaN` 标记，`_tprint`/摘要渲染 "(unavailable)" |
| 5 | low | `realsense_record_example.py` | 重试循环最后一次失败仍空等 3s（sleep 在 `attempt<2` 守卫外） | 把 `time.sleep` 移入 `attempt < 2` 守卫 |

### 复审 refute（无需改）

- `calibrate_camera.py` 的 `main()` 缺 `except Exception` fault-latch：既存形态、非本改动引入，`try/finally` 已是严格改进。
- `pointcloud_process_example.py` 备份时间戳秒级并发覆盖：单进程单次调用下不可达。

### 验证

- `conda run -n real_robot python -m compileall -q dexmani_real examples` → OK。
- 静态残留核对通过（`provenance_runtime`/`replay_runtime_hash`/`_optional_*`/`sanity_min_corrected_x`/`sensor_data[0]` 全库零残留）。
- 仍**不运行**硬件：`--live` 运动、标定写入、SDK/相机调用；replay live 端到端需真机。

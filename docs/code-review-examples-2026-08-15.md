# Code Review 报告：`examples/` 目录

- **日期**：2026-08-15
- **范围**：`examples/` 全部 9 个入口脚本（7320 行）
- **方法**：57 个 agent（10 finder + 对抗性验证 skeptic + completeness critic），38 个候选，逐条读真实代码并交叉核对内部 `dexmani_real` 模块 API。
- **产出**：仅报告，未改动任何代码，未运行硬件。
- **结论**：38 候选 → **24 CONFIRMED**（2 high / 10 medium / 12 low）+ **1 PLAUSIBLE** + **13 REFUTED**。
- **复查（第二轮独立对抗核对，12 个全新 skeptic）**：净结论不变（24 / 1 / 13），但 **2 撤 2 升** —— 复核后实为 **2 high / 9 medium / 13 low** + 1 PLAUSIBLE + 13 REFUTED。详见文末「复查修订」。

---

## 🔴 High（2）

### 1. `calibrate_camera.py:1073` — 主流程无 `try/finally`，异常时臂 worker 变成 Mode 6 孤儿进程 + 共享内存泄漏

`main()` 里 `_run_calibration(...)`（L1073）没有被 `try/finally` 包裹，`shutdown_processes` + `shared.close()` 只在正常返回时执行。而 `_run_calibration` 内部的 `try` 在 L769 才开始，此前有多个可抛异常的调用：`_start_camera`（L652，无相机/串口错误时 `pipeline.start()` 抛 `RuntimeError`）、`keys.start()`（L662）、`compute_eef_pose_world`（L673）、`cv2.namedWindow`（L689）。

- **触发**：无相机或错误 `--serial` 运行 → `pipeline.start()` 抛异常 → 跳过 `shutdown_processes`。
- **后果**：`arm_loop` 子进程是 `daemon=False` 且在 Mode 6（L1070），靠 `while shared.is_running.value` 循环（`arm_loop.py:682`），而 `is_running=False` 只在 `shutdown_processes_verified` 里写 —— 于是臂持续伺服、SharedStorage 不 unlink、父进程 atexit join 可能挂死。
- **佐证**：同目录 `keyboard_teleop.py`（L754-819）正确地用 `try/except/finally` 包住整个 session，这是不一致而非 by-design。
- **修复**：把 `_run_calibration` 调用放进 `try/finally`，`finally` 里调 `shutdown_processes`（保留现有 `except RuntimeError` 处理）。

### 2. `pointcloud_process_example.py:367/370` — 静默、无备份地覆写生产标定 `desk_plane.json`

`main()`（L566）无条件调 `_calibrate_desk`，执行一次 RANSAC 后立即 `save_desk_plane` 写到 `dexmani_real/config/desk_plane.json`，全程无确认、无备份。

- **消费方**：`PointCloudProcessor.__init__`（`pointcloud_processor.py:177-179` 自动加载）、`camera_process`（`table.plane_abcd`，源自 `config/defaults.py:163`）—— 是碰撞模型、homing、replay 预检共用的生产输入。
- **违反契约**：CLAUDE.md §7 明确该入口需 "explicit confirmation for desk-plane write"。
- **已澄清（skeptic 反驳成功的部分）**：写本身**是原子的**（`save_desk_plane` → `atomic_json_dump`，mkstemp→fsync→os.replace），"非原子写"前提不成立；确证缺陷是**缺确认 + 缺备份**。对比 `calibrate_camera.py:520-525` 用 `shutil.copy2` 写前备份。
- **修复**：写前 `input()` 确认 + `shutil.copy2` 备份（或改为 `--write` 显式 flag）。

---

## 🟠 Medium（10）

1. **`collect_teleop.py:574`** — `finally` 里 `group.shutdown()` 抛 `RuntimeError` 后裸 `raise`，覆盖正在传播的原始异常与其 traceback（`session.py:49`/`processes.py:165,170` 确实会抛）。原始错误只剩 L564 的日志。修复：先记录原始异常再决定是否 re-raise（`raise ... from None` 语义）。**⚠️ 复查撤回 → REFUTED**（裸 `raise` 在 L576；原始异常已被 except 记录、退出码恒 1，无掩蔽。见文末「复查修订」）。

2. **`replay_episode.py:795`**（safety-gating）— `replay_runtime_hash` 计算后**从未参与任何比较**（L855 丢弃）；episode 记录的 `resolved_config_sha256` 只做 `_is_sha256` 格式校验，**从不与当前 config 相等比较**。只有 3 个控制器参数（acc/speed/jerk）+ URDF/SRDF 哈希做相等校验；workspace 边界、table 参数、IK 阈值等改动会静默通过 provenance 并进入 worker 启动。docstring 声称的 "config hash" 检查名不副实。

3. **`replay_episode.py:816`** — `--acc`/`--joint-speed` CLI 覆盖**在 live 模式下必然被 provenance 拒绝**：`_resolve_replay_runtime` 把覆盖值写入 `arm.max_*`，随后 `_verify_trajectory_provenance` 拿它和 episode 记录值做 `np.isclose`，一旦不同就 raise。即 `--acc 500`（比记录 900 更保守的"收窄"）也会被拒，与 CLI help 承诺矛盾，覆盖 flag 在唯一有用的场景下不可用。

4. **`calibrate_vr_heading.py:253`** — `SharedStorage.create()` 之后无 `try/finally`（L253–332），settle/countdown/collect 期间的 `KeyboardInterrupt` 或 numpy 异常直接逃逸，`shared.close()` 不执行 → POSIX 共享内存环 + `mp.Queue` feeder 不 unlink（VR proc 是 `daemon=True` 会随父进程终止，proc 半边较轻）。

5. **`calibrate_camera.py:725`** — `_event_solve` 不捕获 `_calibrate_and_select`（退化几何时 L396 `raise RuntimeError("all hand-eye methods failed")`）或 `_save_cameras_json`（已有 `cameras.json` 损坏时 L504 `json.load` 抛 `JSONDecodeError`）的异常 → 一次坏 ENTER 就中断整个交互会话，并连带触发 High #1 跳过 shutdown。与 `_event_capture` 的优雅 catch-and-report 不一致。

6. **`calibrate_camera.py:708`** — `_event_capture` 只校验 `np.all(np.isfinite(eef_pos))`，跳过了主循环 `validate_arm_feedback` 强制要求的 `connected/state_valid/error_code/源新鲜度`。arm 瞬态读失败时会发布 `connected=0/state_valid=0` 但保留最后有限 `eef_pos/rot6d` 的帧，于是陈旧臂姿态会被配对新检测到的 marker，污染手眼求解。

7. **`visualize_episode.py:682`** — `EpisodeVisualizer` 在 `try/finally` **之外**构造（finally 只包住 `log_step` 循环）；`__init__` 在 L172 打开 3 个 h5py + VideoDecoder，其后 `read_camera_all` 帧数不匹配抛 `ValueError` 或缺 `depth_scale` 抛 `ValueError` 时，`close()` 永不执行，资源泄漏。

8. **`pointcloud_process_example.py:316`** — `_run_2d_filters` 声称与生产管线 "byte-identical"，但**跳过了 medianBlur + NaN 恢复**（`depth_median_enabled=True` 默认开启）。生产 `process()` 先中值滤波再把无效像素恢复为 NaN，NaN 经 LoG 不会在无效/有效边界产生梯度；本诊断直接对原始 depth（无效=0.0）跑 LoG，边界处巨大梯度被误判为边缘，打印的移除计数/边缘图**高估**了边缘移除，误导阈值调参。

9. **`calibrate_vr_heading.py:391`** — 计算出的 transform 即便 `grade='poor'`（std≥5°）也照样 `atomic_json_dump` 覆写 `vr_transform.json`，无确认无备份；而运行时 preflight `load_vr_transform(reject_poor=True)` 会拒绝 poor 档 → 一次差采集**静默毁掉好标定并砖掉 teleop**。对比 `calibrate_camera.py` 有 `.bak` 备份。

10. **`xhand_control_example.py:191`** — `read_state` 无论 `finger_id` 传多少都硬编码读 `state.sensor_data[0]`（拇指），而 `sensor_data` 是位置序 thumb→little 的 5 元素数组（`robot/xhand.py:1124-1127` 确认），`finger_id=5` 应读 `sensor_data[1]` → 打印的 calc_force/temperature 张冠李戴。

---

## 🟡 Low（12）

1. `replay_episode.py:686` — `np.savez_compressed` / `json.dump` 非原子结果写，崩溃留损坏文件。
2. `replay_episode.py:1269` — `previous_hand_cmd` 只写不读的 dead store（两处赋值零读取）。
3. `keyboard_teleop.py:661` — "XHand 超时不确定态" 守卫是**不可达死代码**（`is_ready('hand')` 单调，`hand_enabled` 语义已使其恒 False）。
4. `keyboard_teleop.py:669` — 冗余的初始臂状态轮询：L669 读到 qpos 即丢弃，L287 立刻重读同一状态（退化场景下双倍 15s 等待）。**⚠️ 复查撤回 → REFUTED**（故意的 pre-flight fail-fast，非冗余 bug。见文末「复查修订」）。
5. `visualize_episode.py:544` — 预计算 `/pointcloud` 路径无条件渲染全零占位帧（`flag_pointcloud_valid=False`）成原点黑团，未读已加载的 valid flag。
6. `visualize_episode.py:492` — 把 `hand_tactile_sum`（`robot/types.py:53` 标注 "SDK-scaled, physical unit unverified"）的派生序列标签成 "(N)" 牛顿，属单位过度声明。
7. `realsense_record_example.py:621` — 重试日志谎报退避：打印 "retrying in 2s/3s" 却固定 `sleep(1.0)`，且最后一次迭代打印 "retrying" 实际不再重试。
8. `collect_teleop.py:577` — `if group is None and not shared_closed` 中 `not shared_closed` 是死项（`shared_closed` 只可能经 `group.shutdown()` 变真，而那时 `group` 必非 None）。
9. `pointcloud_process_example.py:440` — 单帧计时提取读私有 `_t_*` 累加器，`process()` 早退返回 None 时各 3-D 阶段显示 0/"disabled" 而 `pipeline_total` 却有时长，误导预算调参。
10. `calibrate_vr_heading.py:377` — sanity check `corrected[0] < 0.98` 恒假（`T=R_z(-θ)` 由同一 θ 构造，`T@mean_fwd=[1,0]` 恒成立），"re-run recommended" 警告是死代码。
11. `calibrate_vr_heading.py:232` — `--duration` 绕过 `HeadingCalibrationConfig.__post_init__` 校验，负值/超大值被接受，产生误导的 "0 frames collected" 或无界采集循环。
12. `xhand_control_example.py:259` — RS485 路径枚举了可用端口却**丢弃结果**，仍硬编码开 `/dev/ttyUSB0`（对比 EtherCAT 路径正确用 `ethercat_ports[0]`）。

---

## ⚪ PLAUSIBLE（1，需真机/SDK 确认）

- `realsense_record_example.py:451` — 点云生成只 `except ValueError`；open3d `voxel_down_sample` / pytorch3d `sample_farthest_points` 抛 `RuntimeError` 时会逃逸并中止整个交互循环、跳过 `viewer.close()`/统计汇总（代码结构静态可证，但 open3d 在合法输入下是否真抛 `RuntimeError` 属 vendor 运行时行为，故定 PLAUSIBLE）。

---

## ❌ REFUTED（13，一句话为何不成立）

| 位置 | 原主张 | 反驳要点 |
|---|---|---|
| `xhand_control_example.py:267` | 序列号 bytes/int vs str 比较 | SDK `get_serial_number` 经 pybind11 返回 `tuple[ErrorStruct, str]`，是 str 非 bytes |
| `xhand_control_example.py:282` | 无 try/finally 泄漏设备 | `sys.exit(1)` 只在 open 失败时触发（无可泄漏）；无静态可到的中途异常 |
| `collect_teleop.py:121` | heartbeat 键缺失 KeyError | `heartbeat_timeouts` 默认工厂恒含 recorder/camera 且 `__post_init__` 强制完整性 |
| `keyboard_teleop.py:377` | 可恢复错误+退出键→FAULT | `if/elif` 短路，退出优先于可恢复错误 → 干净退出 |
| `calibrate_camera.py:288` | 坐标系命名相反 | 库内统一 X2Y 约定下 `T_cam2base == T_base_camera` 同向，且代码自洽 |
| `visualize_episode.py:543` | pointcloud (N,6) 颜色契约 | recorder schema + writer 均强制 float32 0..1，契约端到端一致 |
| `replay_episode.py:337` | send_mask 长度未校验 | ⚠️ 复查升回 CONFIRMED(low)：`send_mask` 未纳入 shape 校验循环，`flag_action_queued` 帧数不足时 `send_mask[frame_idx]` 抛 IndexError（见文末「复查修订」） |
| `replay_episode.py:1880` | 元数据回退未做正数校验 | `resolve_runtime_config` 会跑 `ArmConfig.__post_init__`（0<v≤500, 0<a≤50000） |
| `replay_episode.py:1249` | 手反馈新鲜度用 liveness 超时 | 手 heartbeat 与 state 帧同循环写入，是同一信号；与生产 `teleop/loop.py:186` 一致 |
| `visualize_episode.py:372` | depth/RGB 分辨率可能不一致 | `CameraStreamWriterConfig.__post_init__` 强制 `depth_shape == rgb_shape[:2]` |
| `visualize_episode.py:642` | close 吞异常 | 离线 viewer 的 best-effort teardown，by-design |
| `collect_teleop.py:162` | 同上（手新鲜度） | 同 `replay:1249`，设计一致而非 bug |
| `xhand_control_example.py:164` | 序列号按 codepoint 打印 | ⚠️ 复查升回 CONFIRMED(low)：`std::array<char,32>` 暴露为单字符 str 的 list，`print([:16])` 打印 list repr 而非拼好的序列号（见文末「复查修订」） |

---

## 🔁 跨文件模式（合并视角）

1. **硬件生命周期缺 `try/finally`**：`calibrate_camera.py`（孤儿 Mode 6 worker）与 `calibrate_vr_heading.py`（shm 泄漏）都有；`keyboard_teleop.py` / `collect_teleop.py` 是对的范式，可作为参照统一。
2. **标定写入缺确认/备份**：`desk_plane.json`（pointcloud）与 `vr_transform.json`（calibrate_vr_heading）无门控；`calibrate_camera.py` 的 `shutil.copy2` 备份是对的，应推广。
3. **provenance/config 声明未落实**：`replay_episode.py` 的 config SHA 只做格式校验、`--acc/--joint-speed` 覆盖形同虚设 —— 是同一"声称校验但未真校验"主题的两处体现。

## ⚠️ 覆盖缺口（critic 提出、未升级为已核实 finding）

- `visualize_episode.py --info`：空 HDF5 时 `keys[0]` `IndexError`、`t_frames` 为零除 —— 退化输入鲁棒性未测。
- `calibrate_camera.py _save_cameras_json`：备份时间戳为秒级（并发保存会互相覆盖 `.bak`）、合并时对损坏 JSON 无专门 catch（部分已被 Medium #5 覆盖）。

## 📋 未验证项（需真机确认，未运行硬件）

- SDK 真实返回类型（`get_serial_number` 的 str 判定已静态确认；`realsense_record_example.py:451` 的 open3d `RuntimeError` 需实跑）。
- `calibrate_camera.py` 异常路径的实际挂死/泄漏行为需在无相机场景验证（静态已证结构，未跑）。
- 本报告不涉及任何 `--live` 运动命令、标定写入或 SDK 调用。

---

## 🔁 复查修订（第二轮独立对抗核对）

12 个全新独立 skeptic（按文件分组 + 3 条 HIGH 额外第二票）逐条重新裁定 38 条结论，**未被告知原判定**。净结论不变（24 CONFIRMED / 1 PLAUSIBLE / 13 REFUTED），但以下 **4 条 delta** 修订原结论：

### 撤回（CONFIRMED → REFUTED，原报告过度确认）

- **Medium #1 `collect_teleop.py:574`** — 非异常掩蔽。`except Exception`(L563) 已先 `logger.error(exc_info=True)` 记录原始异常并 `return 1`；finally 里的裸 `raise` 只替换挂起的 return，`main()` 的 except(L407) 又把它转回 `return 1`。原始异常已记录、退出码恒 1 → **无信息丢失**，是刻意的 shutdown-failure 升级。裸 `raise` 实际在 **L576**（非 574）。
- **Low #4 `keyboard_teleop.py:669`** — 是故意的 pre-flight fail-fast 检查（`return False` + `logger.error`），与循环内读取语义不同（`_set_fault`），属轻度重复而非 bug。

### 升回（REFUTED → CONFIRMED，均 low，原报告误驳回）

- **`replay_episode.py:337`** — `send_mask` 切片到 `[:total_frames]` 但**未纳入 L342–354 的 shape 校验循环**；`flag_action_queued` 帧数不足时 `send_mask[frame_idx]` 抛 `IndexError` → "unexpected replay failure" fault。防御性校验缺口（仅损坏 episode 可达）。原"`require_valid()` 保证长度"的反驳依据不足。
- **`xhand_control_example.py:164`** — `DeviceInfo_t.serial_number` 是 `std::array<char,32>`，pybind11 暴露为 32 个单字符 str 的 list；`print([:16])` 打印 list repr（`['0','1','2','R',…]`）而非拼好的序列号，SDK 自身用 `''.join`。原"codepoint"措辞不准（是字符非 int 码点），但**打印方式错误真实**。

### 附带修正（结论不变）

- `pointcloud:367` → 真正的 `save_desk_plane` 调用在 **L370**（L367 是注释）。
- `visualize:492` → 键是 `hand_contact_*`（由 `hand_tactile_sum` 派生），非字面 `hand_tactile_sum`。
- `xhand:282` → 维持 REFUTED，但确实存在 post-open 异常/中断的静态可达路径，靠进程退出回收 fd + close best-effort 而无可报告影响（原"无静态可达路径"不精确）。
- 两条 HIGH 均获第二轮独立票 CONFIRMED，维持。

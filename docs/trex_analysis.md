# T-Rex → DexMani 迁移分析

> 基于 T-Rex `hardware_code/` 与 DexMani 源码的逐项对比。所有行号经 2026-07-13 实码验证。

---

## 1. 架构对比

| 维度 | T-Rex | DexMani |
|------|-------|---------|
| 机器人 | Dexmate Vega-1 双臂 + 2×Sharpa Wave (22-DOF) | xArm7 (7-DOF) + XHand (12-DOF) |
| 控制频率 | 30Hz (IK) / 300Hz (平滑+发送) | 50Hz (全部) |
| 平滑 | 关节空间速度限幅 / Ruckig OTG | 笛卡尔 EMA (pos α=0.8, rot α=0.4) |
| 碰撞检测 | Pinocchio @ 300Hz | MPlib (`_check_teleop_collision_gate`) |
| 相机存储 | MP4 (libx264rgb CRF 18) | HDF5 raw `/rgb(T,H,W,3)` |
| 数据布局 | `episode_NNNN/` 目录, HDF5+MP4+MKV | 单文件 `episode_YYYYMMDD_HHMMSS.h5` |
| 成功/失败 | 目录移动 (`success/` `failure/`) | HDF5 `/meta` 属性 + JSON |
| 写入模式 | 阻塞 put, 无界队列, 每100帧 flush | `put_nowait`, 有界队列(2000), 无 flush |
| 状态机 | INIT→WAITING→IN_EPISODE | IDLE⇄TELEOP⇄PAUSED+EMERGENCY |

---

## 2. P0 — 安全和数据完整性（7 项，~208 行）

| # | 问题 | T-Rex 做法 | DexMani 现状 | 修复 | 行数 |
|---|------|-----------|-------------|------|------|
| 1 | **Diff IK 无关节空间速度限幅** | `arm_hand_control.py:901-916` — 每关节 `np.clip(delta, -limit, +limit)` | `ik.py:544-577` diff IK 直接输出 dq，90°跳变限制仅在位置 IK fallback (`ik.py:144`) | `ik.py:575` dq 计算后加 `np.clip(delta, -limit, +limit)` | 15 |
| 2 | **零 flush 调用** | `data_writer.py:579-581` — 每 100 帧 `hdf5_file.flush()` | `TimestampAlignedBuffer` 仅在 `stop_episode()` 写盘。崩溃=数据全部丢失 | 每 100 帧 flush + SWMR 崩溃恢复 | 50 |
| 3 | **ArmInnerLoop 跟踪误差监控（A4 已实现）** | `full_robot_action_loop` 每 tick 检查 `|actual-target|>10.0rad` | `inner_loop._monitor()` 现比对 `|current-target|`，超过 `tracking_error_warn_rad`（`ArmInnerLoopConfig`，默认 0.35 rad）时在 status 告警（被动，不置 `_error_state`）— A4 已实现 | `_send_target()` 前加 `np.abs(current-target)>threshold` 检测 | 20 |
| 4 | **validate_action 缺少碰撞检测** | 300Hz `pin.computeCollisions(stop_at_first=True)` | `validate.py:26` 声明了 `env_collision_check` 参数但**从未调用**（死代码），调用者 (`controller.py:335-337`) 也未传入 | 在 `validate_action()` 体内接入碰撞检查 | 80 |
| 5 | **CameraRingBuffer torn-read** | ZMQ 单帧原子消息 | `ring_buffer.py:354-416` 读 `slot_seq`(L366) 后拷贝 RGB/深度(L391-414)，但**不重读序列号**验证写入端未覆盖 | 拷贝后重读 `slot["sequence"]`，不一致则丢弃 | 8 |
| 6 | **disconnect() 无异常保护** | DexControl `__exit__` 用 try/except 包裹每个组件 | `interface.py:109-111` 顺序调用 `arm.disconnect()` → `hand.disconnect()`，前者异常则后者被跳过 | 各自包 try/except | 5 |
| 7 | **双臂/手 tracker 无全量校验** | 双 tracker 任意缺失则跳过整帧 | `controller.py:310-313` 按 `ik_ok`/`retarget_ok` 独立门控，但 `send_action()` 在 L347-350 **无条件调用** | `all_components_ok` 检查门控 send | 30 |

---

## 3. P1 — 架构和鲁棒性（14 项，~1,130 行）

| # | 问题 | T-Rex/DexControl 做法 | DexMani 现状 | 修复 | 行数 |
|---|------|----------------------|-------------|------|------|
| 8 | **无自定义异常体系** | `RobotError`, `ConnectionError`, `HardwareError`, `SafetyError` 等类型化异常 | 零自定义异常类。`except Exception` 裸捕获 (`inner_loop.py:315-316`, `xarm7.py:86` flag 方式) | `exceptions.py` — `DexManiError` + 6 子类 | 80 |
| 9 | **错误处理约定未文档化** | DexControl `arm.py:50-58` 类级 docstring 约定 | 三种不一致模式：`tuple[bool,str]` (`validate.py`) / `ValueError` (`interface.py`) / `_error_state` flag (`inner_loop.py`) | `CLAUDE.md` 增加错误处理约定章节 | 15 |
| 10 | **无 strict 配置键校验** | YAML 加载时拒绝未识别键 | `serialization.py:169-171` 明确文档："Extra keys ... silently ignored"。拼写错误静默回退默认值 | `_strict` flag, `extra_keys` 检查, 抛 `ValueError` | 40 |
| 11 | **无仓库根路径配置解析** | CLI 入口将 `--config` 相对仓库根解析 | 无通用 `--config` 路径解析。仅 `camera_calib.py:105-112` 有相机特定实现 | `resolve_config_path(name)` 工具函数 | 25 |
| 12 | **无 ExecutionTimer** | 上下文管理器 + 装饰器，累积 mean/min/max/count | `time.perf_counter()` 散落在 8+ 文件中，重复 `t0=...; dt=...-t0` 样板 | `utils/timer.py` | 50 |
| 13 | **机器人模型定义分散 7 处** | `robot_descriptions.py` 集中管理：关节名、限位、URDF 路径、工厂函数、导入时存在性检查 | `xhand.py:27-40,147-149`, `xarm7.py:37-46`, `types.py:23,155-160`, `collision_model.py:59-62`, `simulation/xarm7_xhand.py:58,86-88` | `robot/robot_descriptions.py` — 集中描述 + 工厂函数 + `Path.exists()` 检查 | 200 |
| 14 | **无两阶段初始化** | `wait_for_active()` 轮询硬件确认数据流已启动 | `interface.py:103-107` connect() 立即返回，不验证状态数据实际到达 | `wait_for_live_data(timeout=5.0)` 轮询 `get_state()` | 40 |
| 15 | **无组件名校验** | `validate_component_names()` 提供 "Unknown 'X'. Available: [...]" | 无校验。拼写错误导致不透明 `KeyError` | `validate_camera_name()` 检查 `_names` | 20 |
| 16 | **无实时可视化** | 独立渲染线程显示相机/关节/力矩 | `vr_teleop_shm.py` 零 `cv2.imshow`。仅事后 Rerun 查看器 | `teleop/viz/live_viz.py` — 独立线程 OpenCV 仪表板 | 300 |
| 17 | **无触觉独立线程** | 守护线程以可配频率拉取，lock 保护缓冲 | `controller.py:305` 主循环内同步拉取触觉数据 | `TactileFetchThread` — 可选独立线程 | 80 |
| 18 | **运行时不可切换 data_dir** | 允许跨 episode 切换输出目录 | `episode_recorder.py:39-43` data_dir 为不可变构造参数。`vr_teleop_shm.py:148-151` 硬编码为 `"episodes"` | `set_data_dir(path)` + 热键 | 20 |
| 19 | **无 Rich 硬件健康面板** | `rich` 表格：连接状态、关节限位、固件版本 | 零 `rich` 导入。`connect()` 返回裸 dict | `print_hardware_status()` — 带颜色格式表格 | 100 |
| 20 | **无单肢冻结热键** | 支持独立冻结单手 | `keyboard.py` 仅 6 键：B/C/S/H/Q/ESC | `F`=冻结手, `A`=冻结臂 | 60 |
| 21 | **无聚合元数据目录** | Parquet 文件支持跨 episode 过滤，无需打开 HDF5 | HDF5 `/meta` 属性 + JSON，`load_episodes()` 逐个打开文件 | `metadata_catalog.py` — JSONL 增量追加，`filter_episodes()` | 100 |
| 22 | **DLS IK 无姿态代价** | `ik_utils.py:82-98` 两个 `PostureTask` (cost=0.2+0.05) + DAQP | `ik.py:494-499` 仅最小化 `‖J dq − e‖²`，零空间漂移 | 增广雅可比：`J_aug = [J; √w·I]`, `e_aug = [e; √w·(q−q_pref)]` | 30 |
| 23 | **缺 Jlog6 解析 Jacobian 修正** | Pink 用 `Jlog6` 对 SO(3) 对数映射做右 Jacobian 逆修正 | `pose_utils.py:75-82` 已有 `_quat_to_rotvec()` 对数映射，**仅缺 Jacobian 修正因子** | `Jlog6_inv @ J_angular` (Pinocchio 提供 `pin.Jlog6`) | 15 |

---

## 4. P2 — 数据管线和策略就绪（19 项，~2,320 行）

| # | 问题 | T-Rex 做法 | DexMani 现状 | 修复 | 行数 |
|---|------|-----------|-------------|------|------|
| 24 | **无 Episode 质量分析工具** | 随机采样生成 7 种诊断图（跟踪误差、状态-目标叠加） | 仅有 Rerun 交互式查看器，无批量自动分析 | `tools/analyze_episode_quality.py` — 批量指标 + matplotlib | 150 |
| 25 | **无 Delta-base action 导出** | action 表示为相对 chunk-start 的增量 | `export_hdf5_to_zarr.py:65-66,201` 仅绝对 qpos | `--delta-base` 标志 | 40 |
| 26 | **仅 z-score 归一化** | q01/q99 百分位 + 缩放到 [-1,1] | `export_hdf5_to_zarr.py:378-404` 仅 mean/std | `--norm-method {zscore,percentile}` | 30 |
| 27 | **无多数据集合并工具** | `ProcessPoolExecutor` 并行合并，episode ID 前缀防冲突 | 单 `--data_dir` (L497)，无并行，无少样本采样 | 多 `--data-dir`, ID 前缀, `--num-trajectories`, 并行加载 | 120 |
| 28 | **无消费者端图像缩放** | 不同消费者可接收不同分辨率 | 全链路 640×480，无 `cv2.resize` | `read_latest_camera(target_resolution=...)` | 30 |
| 29 | **无 LeRobot v3.0 导出** | `convert_inlab_to_lerobot.py` → Parquet+MP4 | 纯 HDF5，无 Parquet/LeRobot 支持 | `tools/export_to_lerobot.py` | 300 |
| 30 | **Episode 回放** | 从 HDF5 重解 IK 获取关节目标 | 无回放功能 | 直接回放 `/action_arm_joint` + IK 重解 fallback | 300 |
| 31 | **回放安全检查** | 4-6 层：300Hz 安全循环、碰撞环境、Event 终止、速度限幅 | 无回放安全 | `ReplaySafetyMonitor` — 4 层（力矩偏差/工作空间/E-Stop 线程/关节跳变） | 80 |
| 32 | **回放初始化** | 碰撞感知的运动规划启动 | `planner.py:178-251` 已有 `plan_path()`（9 项校验），可直接复用 | `ReplayInitializer` — 3 阶段（中立位→确认→起始位） | 80 |
| 33 | **无策略推理 ZMQ 桥接** | ZMQ REQ/REP → 外部推理服务器 | 已有 `SharedSyncPrimitives` 握手 + ZMQ (`retarget_server.py`)，缺推理桥接 | ZMQ 推理桥接模块 | 150 |
| 34 | **无频率解耦基础设施** | 策略 30Hz / 执行 300Hz + 插值 | 50Hz 单一速率。`types.py:303-304`: "VR 原生 50Hz → 无需解耦" | `ActionInterpolator`（策略模式时使用） | 120 |
| 35 | **无 Delta-EEF action 空间** | 策略输出相对于 chunk-start 的 delta EEF + chunk 边界 FK 漂移修正 | `solve_teleop_ik()` 接收绝对姿态 | `solve_delta_eef_ik(delta_pos, delta_rot6d, current_eef, current_qpos)` | 100 |
| 36 | **Ruckig OTG** | `arm_hand_control.py:760-780` — v/a/jerk 限制在线轨迹生成 | xArm Mode 6 固件已有轨迹规划。Ruckig 增加预测层 | 基准测试：EMA vs EMA+速度限幅 vs Ruckig | 150 |
| 37 | **ACT 时序聚合** | `eval_trex_async.py:75-96` 指数加权平均多个 chunk 预测 | 无。每帧独立 IK 求解 | `aggregate_chunks()` 纯函数 | 22 |
| 38 | **单文件 vs 目录结构** | 每 episode 目录 + MP4/MKV + 成功/失败子目录路由 | 单文件 `episode_YYYYMMDD_HHMMSS.h5`，平坦结构 | 每 episode 目录 + 成功/失败子目录 | 200 |
| 39 | **相机 MP4 编码** | libx264rgb CRF 18 (~0.02MB/帧 vs ~0.9MB 原始) | 原始 uint8，无视频编码 | 基准测试驱动：选编码器、调 CRF、测 PSNR/SSIM | 400 |
| 40 | **_tick_mode4() 过时注释** | N/A（DexMani 特有） | 3 处引用不存在的 `_tick_mode4()`，且错误声称 250Hz | 替换为正确描述（固件 Mode 6 负责限速） | 3 |

---

## 5. 不应迁移

| 项 | 原因 |
|----|------|
| T-Rex 的无界阻塞队列 | DexMani 的有界 `put_nowait` + 丢弃更安全，不会阻塞控制循环 |
| 30Hz IK 循环 | DexMani 50Hz 在速度和 IK 稳定性间平衡更好 |
| Sharpa 手部初始化 (`speed_coeff=0.3`) | XHand 硬件特定，不适用 |
| 触觉数据线程 | DexMani 无触觉传感器 |
| Vive tracker 集成 | DexMani 使用 Quest VR |

---

## 6. 实施路线图

| 阶段 | 项数 | 行数 | 时间 |
|------|------|------|------|
| **Phase 1:** 安全加固 (P0) | 7 | ~208 | 3-5 天 |
| **Phase 2:** 架构基础 (P1) | 8 | ~470 | 1-2 周 |
| **Phase 3:** 操作体验 (P1) | 6 | ~660 | 1-2 周 |
| **Phase 4:** 数据管线 (P2) | 8 | ~715 | 1-2 周 |
| **Phase 5:** 策略就绪 (P1-P2) | 11 | ~1,605 | 待定 |
| **合计** | **40** | **~3,658** | |

**Phase 1-3**（~1,338 行，3-4 周）覆盖全部安全和操作体验差距。**Phase 4-5**（~2,320 行）为数据管线和策略就绪项，可在策略训练开始前实施。

### Phase 1: 安全加固（P0）

| # | 项 | 行数 |
|---|----|------|
| 1 | Diff IK 每关节速度限幅 | 15 |
| 2 | 录制周期性 flush | 50 |
| 3 | ArmInnerLoop 跟踪误差监控 ✅ 已实现 (A4) | 20 |
| 4 | validate_action 接入碰撞检测 | 80 |
| 5 | CameraRingBuffer torn-read 检测 | 8 |
| 6 | disconnect() try/except 包裹 | 5 |
| 7 | 双臂/手 tracker 全量校验门 | 30 |

### Phase 2: 架构基础（P1）

| # | 项 | 行数 |
|---|----|------|
| 8 | 自定义异常体系 | 80 |
| 9 | 错误处理约定文档 | 15 |
| 10 | Strict 配置键校验 | 40 |
| 11 | 仓库根路径配置解析 | 25 |
| 12 | ExecutionTimer 工具 | 50 |
| 13 | 集中式机器人模型描述 | 200 |
| 14 | 两阶段组件初始化 | 40 |
| 15 | 组件名校验 | 20 |

### Phase 3: 操作体验（P1）

| # | 项 | 行数 |
|---|----|------|
| 16 | Rich 硬件健康面板 | 100 |
| 17 | 实时可视化线程 | 300 |
| 18 | 单肢冻结热键 | 60 |
| 19 | 运行时切换 data_dir | 20 |
| 20 | 触觉独立拉取线程 | 80 |
| 21 | 聚合元数据目录 | 100 |
| 22 | DLS IK 姿态代价 | 30 |
| 23 | Jlog6 Jacobian 修正 | 15 |

### Phase 4: 数据管线（P2）

| # | 项 | 行数 |
|---|----|------|
| 24 | Episode 质量分析工具 | 150 |
| 25 | Delta-base action 导出 | 40 |
| 26 | q01/q99 百分位归一化 | 30 |
| 27 | 多数据集合并工具 | 120 |
| 28 | 消费者端图像缩放 | 30 |
| 29 | LeRobot v3.0 导出 | 300 |
| 30 | Episode 回放 | 300 |
| 31 | 回放安全检查 | 80 |

### Phase 5: 策略就绪（P1-P2）

| # | 项 | 行数 |
|---|----|------|
| 32 | 回放初始化 | 80 |
| 33 | ZMQ 推理桥接 | 150 |
| 34 | 频率解耦基础设施 | 120 |
| 35 | Delta-EEF action 空间 | 100 |
| 36 | Ruckig OTG 基准测试 | 150 |
| 37 | ACT 时序聚合 | 22 |
| 38 | 每 episode 目录结构 | 200 |
| 39 | 相机 MP4 编码 | 400 |
| 40 | _tick_mode4() 注释修复 | 3 |

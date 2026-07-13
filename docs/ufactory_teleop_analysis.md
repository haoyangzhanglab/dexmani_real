# ufactory_teleop → DexMani 可迁移改进（最终版）

**日期**：2026-07-13 | **方法**：全代码库逐行比对 + 三轮 agent fact-check（35 agent，500+ 工具调用），所有行号已代码核实。

---

## 代码库对比

| | ufactory_teleop | DexMani |
|---|---|---|
| 规模 | ~15 源文件，~1500 行核心 | ~80 源文件，成熟模块化架构 |
| 控制模式 | Mode 1 / 6 / **7** | **仅 Mode 6** |
| 遥操作方案 | Pika Sense / UMI / GELLO | VR (Quest) |
| 数据记录 | **无** | HDF5 多流记录 |
| 安全机制 | 基础 error code 检查 | 多层：validate + collision + desk safety + emergency stop |
| 机器人抽象 | 单一 `UFRobot` (~265行) | `RobotInterface` + `ArmInnerLoop` + `XArm7` + `XHand` |

---

## 发现总览

| 优先级 | 项目 | 文件 | 行数 | 风险 | 来源 |
|--------|------|------|------|------|------|
| **P0** | 软启动斜坡 | `inner_loop.py` | ~15 | 零 | 已有 |
| **P0** | 运行时 Mode 重入守卫 | `inner_loop.py` | ~10 | 低 | 已有 |
| ~~**P0**~~ ✅已实现 | 启动帧跳过 | `collection_config.py`+`episode_recorder.py` | ~10 | 零 | 已有 |
| **P1** | Arm Error Code 语义映射 | `error_codes.py`(新) + `inner_loop.py` + `xarm7.py` | ~80 | 零 | **新** |
| **P1** | Arm 电机温度安全门 | `inner_loop.py` + `xarm7.py` | ~25 | 低 | **新** |
| **P1** | Mode 切换后错误码验证 | `inner_loop.py` + `xarm7.py` | ~8 | 零 | **新** |
| **P1** | YAML 驱动配置 | `yaml_loader.py`(新) + 入口脚本 | ~100 | 低 | 已有 |
| **P1** | Mode 7 笛卡尔可选 | 4 文件 | ~150 | 低 | 已有 |
| **P1** | 手部关节归一化工具 | `hand_utils.py`(新) | ~30 | 零 | 已有 |
| **P2** | Command Counter 通信看门狗 | `inner_loop.py` | ~20 | 低 | **新** |
| **P2** | 固件级关节限位推送 | `inner_loop.py` + `xarm7.py` | ~20 | 低 | **新** |
| **P2** | 末端执行器类型抽象 | `types.py` + `interface.py` | ~50 | 零 | 已有 |
| **P2** | EMA Alpha CLI 可配置 | `vr_teleop_shm.py` | ~5 | 零 | 已有 |
| **P2** | Held 帧去重后处理 | `post_processor.py` | ~40 | 零 | 已有 |
| **P2** | Dead-Man 开关（键盘） | `keyboard_handler.py` | ~20 | 低 | 已有 |
| **P3** | 连接时自动预置位 | 2 文件 | ~5 | 低 | 已有 |
| **P3** | 线性速度限制因子 | `xarm7.py` | 1 | 零 | 已有 |
| **P3** | SDK Import 错误提示 | `xarm7.py` + `realsense.py` | ~30 | 零 | 已有 |
| Ref | 双臂组合架构 | — | — | — | 已有 |

**合计：18 项待办（13 已有 + 5 新）+ 1 已实现（启动帧跳过），~610 行。**

---

## P0 — 安全与正确性（2 项待办 + 1 已实现，~35 行）

### P0-1: 软启动斜坡

**问题**：DexMani 从第一条指令起全速 (1.57 rad/s)。ufactory 前 20 条指令降速 (0.2 rad/s) 消除启动 jerk。

| ufactory (`uf_robot.py:206`) | DexMani (`inner_loop.py:341`) |
|---|---|
| `jnt_spd = 0.2 if self._cmd_cnt < 20 else self._joint_speed` | `speed=self._cfg.joint_max_speed` — 固定值 |

**方案**：`ArmInnerLoopConfig` 加 `ramp_steps: int = 0` + `ramp_start_speed: float = 0.3`，`_send_target` 内线性插值速度。`ramp_steps=0` 保持现有行为。

---

### P0-2: 运行时 Mode 重入守卫

**问题**：DexMani 仅在 init 时检查 `arm.mode==6` (`inner_loop.py:196-207`)。若示教器急停导致固件退出 Mode 6，目前仅 `inner_loop.py:385` 的 `_monitor()` 被动节流告警（不触发 error state、不自动重入）。ufactory 每帧检查并自动重入。

| ufactory (`uf_robot.py:209-212`) | DexMani (`inner_loop.py:330-368`) |
|---|---|
| `if arm.mode != 6: set_mode(6); set_state(0)` | `_send_target()` 无 mode 检查 |

`arm.mode` 是 SDK 缓存属性（纳秒级），零网络开销。

**方案**：`_send_target()` 开头加 mode 检查，偏离即调 `_init_mode()` 重入，超 3 次失败触发 error state。

---

### P0-3: 启动帧跳过

**问题**：DexMani 从第一帧开始记录。操作员按下"开始"后约 0.5-1s 处于过渡姿态——这些帧对策略训练是噪声。ufactory 的 ViveTracker 跳过前 100 帧 (`vive_tracker.py:182`)。

**已实现**：`skip_initial_frames` 已加入 `collection_config.py:26`（默认 10 帧），经 `collection_loop.py` 透传给 recorder；跳过逻辑在 `episode_recorder.py:201`，位于 `max_frames` 检查 (`:193`) 之后且被跳过的帧提前返回 False，因此不计入 `max_frames` 上限。

---

## P1 — 显著改进（6 项，~393 行）

### P1-1: Arm Error Code 语义映射 ⭐新

**问题**：DexMani 在 6 处以上日志点输出原始整数错误码（`inner_loop.py:287,296,363-365`，`xarm7.py:150-152,223-226,310-313`）。`error_code=11` 需查 xArm SDK 文档才能理解。

**方案**：新增 `robot/xarm7/error_codes.py`，维护 ~40 error + ~99 warn code 映射字典，`decode_error(code) -> str`。修改 6 处日志点。

---

### P1-2: Arm 电机温度安全门 ⭐新

**问题**：Arm 读取温度 (`xarm7.py:194` `arm.temperatures`) 但从不检查阈值。inner_loop 完全不读温度（`get_joint_states()` 只返回位置/速度/力矩）。XHand 同样读温度但不检查（`xhand.py:707`）。xArm7 固件硬 cutoff ~80°C——软件层无预警。

**方案**：`ArmInnerLoop._run()` 中每 50 周期 (~1s) 读 `arm.temperatures`，>75°C 告警，>78°C 急停。

---

### P1-3: Mode 切换后错误码验证 ⭐新

**问题**：`robot_init()` (`xarm7.py:310-313`) 和 `clear_error()` (`xarm7.py:149-152`) 在操作后验证 `get_err_warn_code()`，但 `_set_mode()` (`xarm7.py:328-336`) 和 `_init_mode()` (`inner_loop.py:388-427`) 不做此检查——mode 切换失败静默继续。

**方案**：在 `_set_mode()` 和 `_init_mode()` 的 mode transition 之后，复用已有的错误检查模式（~4 行/处）。

---

### P1-4: YAML 驱动配置

**问题**：DexMani 入口脚本 209 行硬编码配置。ufactory 用 `-c config.yaml` + `**splat` 实例化 dataclass。

**方案**：新增 `config/yaml_loader.py`，利用已有 `FromDictMixin`。入口精简到 ~40 行。

---

### P1-5: Mode 7 笛卡尔可选

**问题**：DexMani 仅 Mode 6（软件 IK → 关节角度 → 固件）。ufactory 默认 Mode 7（笛卡尔目标 → 固件内部 IK + 轨迹平滑），消除软件 IK 环节。

**方案**：`ArmInnerLoopConfig.control_mode: int = 6`，mode 7 时跳过 `solve_teleop_ik()`，直接 `set_position_aa()`。默认保持 Mode 6。

---

### P1-6: 手部关节归一化工具

**问题**：XHand 关节量程差异 7:1（J3 index_abd: 0.349 rad vs J1 thumb_j1: 2.443 rad，来自 `xhand.py:124-166`）。馈入原始弧度扭曲策略梯度。

**方案**：新增 `utils/hand_utils.py`（`normalize_hand_qpos` / `denormalize_hand_qpos`），不修改 HDF5 记录格式，在数据加载层应用。

---

## P2 — 锦上添花（7 项，~178 行）

### P2-1: Command Counter 通信看门狗 ⭐新

**问题**：现有 `target_timeout_s=0.2` 只检测"控制器没发指令"。若 SDK 接受但固件未执行——`cmd_num` 停滞——完全静默。DexMani 仅在 `xarm7.py:192` 读取 `cmd_num`，从不监控。

**方案**：`_run()` 中比较连续 `arm.cmd_num`，停滞 >50 周期 (~1s) 触发 error state。

---

### P2-2: 固件级关节限位推送 ⭐新

**问题**：DexMani 在 Python 层 `np.clip` 关节限位（`validate.py:47-49`，`xarm7.py:346-352`），但从未调用 `arm.set_reduced_max_joint_range()` 推送到固件。全文搜索零结果。

**方案**：`_init_mode()` 和 `_configure_collision_params()` 中调用 `set_reduced_max_joint_range()`（弧度→度转换后）。

---

### P2-3: 末端执行器类型抽象

**问题**：DexMani 全代码库硬编码 XHand。ufactory 用 `GripperType(IntEnum)` + `GripperParam` 统一 6 种末端执行器（`uf_robot.py:20-47,231-264`）。

**方案**：`robot/types.py` 加 `EndEffectorType(IntEnum)`，`RobotInterface` 构造分发。纯抽象层，不改行为。

---

### P2-4: EMA Alpha CLI 可配置

**问题**：Cartesian EMA 参数 (`pipeline.py:46-47` `alpha_pos=0.8`, `alpha_rot=0.4`) 已可配但未暴露 CLI。GELLO 基线验证了 Mode 6 固件轨迹规划可独立满足关节空间平滑——笛卡尔管线的 EMA 松弛实验需快速调参。

**方案**：入口脚本增加 `--ema-alpha-pos` / `--ema-alpha-rot` argument（~5 行）。

---

### P2-5: Held 帧去重后处理

**问题**：VR 数据不可用时，DexMani 发送上一帧并标记 `flag_held=True`——安全正确，但连续 held 帧形成重复 (s,a) 对造成过采样偏差。ufactory 直接 `continue` 跳过。

**方案**：新增 `recording/post_processor.py` 的 `deduplicate_held_frames()`，超过阈值连续 held 帧降采样。

---

### P2-6: Dead-Man 开关（键盘方案）

**问题**：Pika Sense 用物理按键做遥操作使能——按下激活，释放暂停。DexMani 仅有键盘（B/C/S/H/Q/ESC）。Quest HTS SDK 为裸手追踪，不提供控制器按钮数据，硬件 dead-man 需 OpenXR 集成（~200 行）。

**方案**：键盘使能键（持续按住 `Space` 激活，松开冻结），~20 行。

---

### P2-7: 碰撞灵敏度 — 不修改

ufactory 设置 `collision_sensitivity=0`（关闭，`uf_robot.py:137`），DexMani 使用 `1`（`xarm7.py:54`，`inner_loop.py:193`）。**维持现状**——固件碰撞检测是 DexMani 唯一的接触后安全层，不可移除。所有软件检查都是接触前的几何检查。

---

## P3 — 小改进（3 项，~36 行）

### P3-1: 连接时自动预置位

ufactory `connect()` 自动移动 `start_joints`（`uf_robot.py:127-133`）。DexMani 需手动按 H。方案：`auto_home: bool = False`（默认关闭）。

### P3-2: 线性速度限制因子

ufactory 调用 `set_linear_spd_limit_factor(2.0)`（`uf_robot.py:136`）。DexMani 未调用。影响仅 SDK 阻塞式移动——加 1 行使复位加速。

### P3-3: SDK Import 错误提示

XHand SDK 已有 try/except + 引导（`xhand.py:11-17`），XArm7 和 RealSense SDK 缺失。方案：补齐 import 包装。

---

## 参考：双臂组合架构

ufactory 用 L:/R: YAML 分段 + 独立线程 + 1Hz 存活监控实现双臂遥操作（`uf_robot_umi_teleop_dual.py`，仅 50 行）。DexMani 的 `ArmInnerLoop` 已线程化且独占连接——无架构冲突。需要双臂时启动两个 `TeleopController` 实例即可。

---

## DexMani 已优于 ufactory 的方面

| 方面 | ufactory | DexMani |
|------|----------|---------|
| 数据记录 | 无 | HDF5 多流 (state+action+VR+camera+flags) |
| 触觉 | 无 | tactile_force (5×120×3) + tactile_sum (5×3) |
| 中间表示 | 仅 joint action | target_eef + vr_wrist + vr_landmarks |
| Return-to-home | 无 | 三阶段路径规划，任意时刻触发 |
| 元数据 | 无 | schema_version, camera calib, task labels, operator |
| 错误处理 | 返回码（手动检查） | 异常捕获 → error_state → emergency stop |
| 多相机 | 无 | SHM 多相机采集 + 帧对齐 |

---

## 合并执行计划

| 批次 | 内容 | 项数 | 行数 | 文件 |
|------|------|------|------|------|
| **第一批**（当天） | P0: 软启动斜坡 + Mode 重入守卫（启动帧跳过已实现） | 2 | ~25 | 1 |
| **第二批** | P1 安全+可观测: Error Code 映射 + 温度安全门 + Mode 切换验证 | 3 | ~113 | 5 (2新) |
| **第三批** | P1 能力: YAML 配置 + Mode 7 + 手部归一化 | 3 | ~280 | 6 (2新) |
| **第四批** | P2: 通信看门狗 + 固件限位推送 + 末端执行器抽象 + EMA CLI + Held 去重 + Dead-Man | 6 | ~155 | 6 |
| **第五批** | P3: 自动预置位 + 速度限制因子 + Import 提示 | 3 | ~36 | 4 |

---

## 核查修正记录

| 修正项 | 原值 | 修正后 | 原因 |
|--------|------|--------|------|
| Error Code 行数 | 65 行 | ~80 行 | 实际需映射 ~140 个 code |
| CmdNum 看门狗行数 | 30 行 | ~20 行 | 逻辑简单，无需复杂状态机 |
| Post-Mode Error Check 优先级 | P2 | P1 | 8 行零风险，与已有 `robot_init` 模式一致 |
| 温度安全门 XHand 描述 | "已有 send-error circuit breaker" | 两者均无温度安全门 | `_CONSECUTIVE_ERROR_RECONNECT_THRESHOLD` 是发送错误断路器，非温度断路器 |
| inner_loop 温度读取 | 暗示读取 | 完全不读温度 | `get_joint_states()` 只返回位置/速度/力矩 |
| 新发现总数 | Synthesis 混排 19 项 | 5 项为新 | 其余 14 项来自前次报告 |

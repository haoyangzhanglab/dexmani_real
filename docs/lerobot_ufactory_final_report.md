# LeRobot × UFACTORY → DexMani 迁移分析 — 最终报告

> **合并版**：Round 1（19 项发现，4 维度）+ Round 2（8 新维度，43 项洞察），经两轮对抗性验证。
> **核查规模**：42 个独立 agent，逐项引用双方代码库具体文件和行号。
> **最后更新**：2026-07-13

---

## 目录

1. [执行摘要](#一执行摘要)
2. [架构差异](#二架构差异)
3. [P0 发现（12 项 —— 必须修复）](#三p0-发现12-项)
4. [P1 发现（9 项 —— 应该修复）](#四p1-发现9-项)
5. [P2 / 延迟项](#五p2--延迟项)
6. [已审查并拒绝的发现](#六已审查并拒绝的发现)
7. [统一实施路线图](#七统一实施路线图)
8. [隐蔽成本](#八隐蔽成本)

---

## 一、执行摘要

对 `lerobot_robot_ufactory`（~6,400 行 Python）的完整挖掘，覆盖 **xArm 控制、遥操作、数据记录、策略部署、多机器人架构、设备抽象、相机基础设施、测试/键盘工具、变换数学、配置架构** 共 10 个维度。

### 关键数字

| 指标 | 数值 |
|------|------|
| 总发现数 | 33（Round 1: 19，Round 2: 14 可操作） |
| **P0 发现** | **12**（合并后） |
| **P1 发现** | **9**（合并后） |
| 已拒绝/降级 | 8（含 1 项事实错误） |
| P0 总 LOC | ~1,349 |
| P1 总 LOC | ~435 |
| P2（延迟） | 90 |
| **实用路线图** | **~1,784 LOC** |

### 验证统计

| 结论 | P0 | P1 | 合计 |
|------|----|----|------|
| CONFIRMED | 8 | 0 | **8** |
| OVERSTATED（已修正） | 8 | 7 | **15** |
| INCORRECT（已拒绝） | 0 | 1 | **1** |
| Round 1 已有等效/不适用 | — | — | **11** |

---

## 二、架构差异

| 维度 | lerobot_ufactory | DexMani |
|------|-----------------|---------|
| 控制模式 | mode 6 (关节) + mode 7 (笛卡尔) | mode 6 (关节) |
| 状态反馈 | RT Report socket @200Hz | `ArmInnerLoop` 线程 @50Hz |
| 遥操作 | GELLO / Pika / UMI / SpaceMouse | VR Tracker → IK → retargeting |
| 末端执行器 | 5 种夹爪（归一化到 [0,1]） | XHand 12-DOF 灵巧手 |
| 状态管理 | 布尔标志位 | `TeleopController` 状态机 |
| 数据格式 | LeRobot Dataset (HDF5/Parquet + 图片) | 自有 HDF5 格式 |
| 策略部署 | **有** (`uf_lerobot_eval.py`) | **无** — 最大 gap |
| 多机器人 | `MultipleUFRobot` 前缀命名空间 | 单臂架构 |
| 配置方式 | YAML + dataclass（可热切换） | dataclass + 可选 YAML |

### 已有等效 / 不适用（Round 1 结论，保留不做）

| # | 发现 | 结论 |
|---|------|------|
| 1 | RT Report 线程 | `ArmInnerLoop` 已等效——SDK 方式更安全，50Hz 足够 |
| 3 | 笛卡尔速度因子 | DexMani 仅用 mode 6，不涉及笛卡尔控制 |
| 4 | GripperParam 归一化 | XHand 是 12-DOF，归一化逻辑完全不同 |
| 5 | contextvars 共享状态 | 构造函数注入更明确、更可测试 |
| 6 | instantiate_from_dict | dataclass + YAML 序列化已满足需求 |
| 12 | blend_poses | EMA 滤波已等效，且 lerobot 此用法被注释 |
| 13 | precise_sleep | `time.sleep(0.02)` 在 50Hz 下精度足够 |
| 15 | 多进程图片编码 | DexMani 以原始 uint8 写入 HDF5 |
| 17 | Gripper look-ahead | 代码被注释禁用，不可依赖 |
| 18 | Mock Robot | SAPIEN 模拟 + DummyTracker 部分覆盖 |
| 19 | 多机器人协调 | 当前单臂架构，`RobotInterface` 设计支持多实例 |

---

## 三、P0 发现（12 项）

### 来自 Round 1（4 项）

#### P0-1: 首条命令慢同步（~30 LOC）

**问题**：用户按 BEGIN 进入 TELEOP 时，VR 映射的目标关节可能与当前关节差异很大，手臂以 90°/s 全速跳变。

**lerobot 参考** (`uf_robot.py:348-368`)：前 20 步限速 0.2 rad/s，之后全速。在 mode 6 下调整 speed 参数即可。

**适配** — 修改 `ArmInnerLoop`：
- `ArmInnerLoopConfig` 新增 `slow_start_steps: int = 20`、`slow_start_speed: float = 0.2`
- `_send_target()` 根据计数器选择速度
- `_hold_position()` 始终全速

```python
speed = self._cfg.slow_start_speed if self._cmd_cnt < self._cfg.slow_start_steps \
        else self._cfg.joint_max_speed
```

**文件**：`robot/inner_loop.py` | **LOC**：~30

---

#### P0-2: AsyncEpisodeSaver — 流水线化存盘（~95 LOC）

**问题**：`RecordingSession._handle_stop()` 同步执行 HDF5 I/O（2-5s），episode N+1 必须等 episode N 完全写入。

**lerobot 参考** (`uf_lerobot_record.py:115-190`)：buffer swap + 后台存盘。

**适配** — 拆分 `EpisodeRecorder.stop_episode()` 为快慢两阶段：
- **快阶段**：提取 buffer、收集 metadata、关闭 HDF5、返回 bundle、重置状态（微秒级）
- **慢阶段**（后台线程）：重开 HDF5、写 datasets、更新 attrs

**预计收益**：Episode 间死时间从 2-5s 降至 <1ms。

**文件**：`recording/` 2 个文件 | **LOC**：~95

---

#### P0-3: Chunk Boundary Smoothing（~30 LOC）

**问题**：ACT/DP 策略在 action chunk 边界产生位置跳变。DexMani 当前 EMA 滤波不专门处理 chunk 边界。

**lerobot 参考** (`uf_lerobot_eval.py:339-359`)：向量范数限幅（位置）+ 逐轴限幅（旋转）。

**适配**（优于 lerobot——旋转用 rotvec 模长限幅，旋转不变）：
- 位置：向量范数限幅（保持运动方向）
- 旋转：rotvec 模长限幅
- 推荐阈值：`max_delta_pos_mm=5`, `max_delta_rot_rad=0.05`

**接入点**：`validate_action()` 或策略推理管道直接调用。

**文件**：`robot/validate.py` | **LOC**：~30

---

#### P0-4: 策略推理入口（~270 LOC）

**问题**：DexMani 无策略部署能力——`RobotInterface` 只接受 `RobotAction(arm_qpos_cmd=...)`，策略输出 EEF 位姿。

**lerobot 参考** (`uf_lerobot_eval.py:123-409`)：完整推理循环（策略加载、pre/post processor、绝对/相对动作模式、chunk 平滑、键盘交互）。

**适配** — 8 步：
| 步骤 | 内容 | LOC |
|------|------|-----|
| 1 | `EvalConfig` dataclass | ~50 |
| 2 | EEF→关节 IK 桥接 | ~15 |
| 3 | torch 推理 + 预处理 | ~30 |
| 4 | Chunk 平滑（见 P0-3） | ~25 |
| 5 | continuous_rotvec + 旋转桥接 | ~40 |
| 6 | 相对/绝对帧转换 | ~40 |
| 7 | `RobotInterface` 适配 | ~20 |
| 8 | 入口脚本 | ~50 |

**文件**：新建 `examples/real/policy_eval.py` + 修改 `interface.py` | **LOC**：~270

---

### 来自 Round 2（8 项，经对抗性验证修正）

#### P0-5: `XArm7Config`/`XHandConfig` 缺少 `from_dict`（4 LOC）

**问题**：`RobotInterfaceConfig.from_dict()` 反序列化时静默返回 dict 而非类型化 config 对象。下游代码访问类型化字段时触发 `AttributeError`。

**原始过度声明**：声称 30 LOC，需要 type-based 多态注册表。**实际修复**：4 行——将 `xarm7.py:24` 的死 import `from_dict_helper` 改为 `FromDictMixin`，加入两个 class 声明。

**文件**：`robot/xarm7/xarm7.py`, `robot/xhand/xhand.py` | **LOC**：4

---

#### P0-6: 将 `RateManager` 接入 `TeleopController`（已完成 ✅）

**已落地**：`TeleopController` 现已使用 `RateManager`（`controller.py:35` 导入，`controller.py:153` `self.limiter = RateManager(cfg.target_hz)`），替换了原先的 `RateLimiter`（纯 `time.sleep()`）。原先访问 `self.limiter.period` 属性缺失的潜在 bug 已随之消除。

**结果**：已在 `controller.py:153` 无条件实例化 `RateManager`（混合 sleep+busy-wait，目标误差 <1ms），替换 `RateLimiter`。

**文件**：`teleop/core/controller.py` | **LOC**：5

---

#### P0-7: 合并重复的 `_quat_to_rotvec`（10 LOC）

**问题**：两个不同实现：`pose_utils.py:75-82`（缺少双覆盖保护 `w >= 0`）和 `signal_utils.py:123-137`（有保护）。第三个副本在 `tools/sweep_ema_params.py`。

**适配**：将 `w >= 0` 保护加入 `pose_utils` 版本；将 `signal_utils` 的 15 行嵌套函数替换为从 `pose_utils` 导入。

**文件**：`planning/pose_utils.py`, `utils/signal_utils.py` | **LOC**：10

---

#### P0-8: `list_cameras()` 产品线过滤（3 LOC）

**问题**：`list_cameras()` 已提取 `product_line`（`realsense.py:634`），但返回所有设备不筛选。

**适配**：添加 `by_product_line: str | None = None` 可选参数，插入大小写不敏感子串过滤器。向后兼容。

**文件**：`sensor/realsense.py` | **LOC**：3

---

#### P0-9: 分析解 rotmat→quat（消除 hot path 中的 scipy）（22 LOC）

**问题**：`quat_wxyz_to_rot6d` 和 `quat_wxyz_to_rotmat`（`RobotInterface.get_state()` @50Hz 调用）通过 scipy 往返。`rot6d_to_quat_wxyz` 零调用者——死代码。

**适配**：
- 实现纯 numpy `quat_to_rotmat`（直接四元数代数，无需迹算法）
- 实现 `rotmat_to_quat_wxyz`（4 分支迹算法，wxyz 输出）
- 删除 `rot6d_to_quat_wxyz`（~40 LOC 删除）

**文件**：`planning/pose_utils.py` | **LOC**：22（净减少 ~18 行）

---

#### P0-10: 最简测试脚手架（~420 LOC）

**问题**：15,000+ LOC 代码库零自动化测试（除 `test_recording_alignment.py` 外），涉及安全关键硬件控制。`pyproject.toml` 已有 pytest 配置（`[tool.pytest.ini_options]`）。

**适配**：
| 文件 | 内容 | LOC |
|------|------|-----|
| `tests/conftest.py` | Mock 夹具 | ~30 |
| `tests/test_rate_limiter.py` | 时序精度 | ~50 |
| `tests/test_keyboard.py` | 信号缓冲、幂等启停 | ~80 |
| `tests/test_rate_manager.py` | 混合 sleep+busy-wait | ~80 |
| `tests/test_controller_state_machine.py` | 状态转换、dryrun 集成 | ~180 |

**文件**：`tests/` 6 个文件 | **LOC**：~420

---

#### P0-11: MockRobotInterface（~350 LOC）

**问题**：无硬件则无 controller 可存在——`RobotInterface.__init__` 无条件构造 `XArm7` + `XHand`。`dry_run=True` 跳过硬件发送，但仍需真实 `RobotInterface` 实例。`TeleopController` 独立创建 `ArmInnerLoop`（直接导入 `XArmAPI`）。

**原始过度声明**：声称 250 LOC，忽略 mock arm/hand 子对象（+60 LOC）、`XArmAPI` 传递导入、HDF5 回放模式（+40 LOC）及 pinocchio/MPlib 初始化需求。

**适配**：
| 组件 | LOC |
|------|-----|
| `MockArm` + `MockHand`（形状正确的合成状态） | ~60 |
| `MockRobotInterface`（no-op `send_action`，合成/回放模式） | ~220 |
| `types.py` 重构（延迟导入 `XArm7Config`） | ~15 |
| 集成测试 | ~80 |

**替代方案**：xArm SDK wheel 可通过 pip 安装（无需硬件）——比导入重构更简单。

**文件**：`robot/mock_interface.py`（新建）+ `robot/types.py` | **LOC**：~350

---

#### P0-12: 解耦录制与遥操作激活（~85 LOC）

**问题**：BEGIN 信号原子性地转换到 TELEOP 状态并开始录制。虽然 controller 支持 `recorder=None` 构造，但遥操作中无法独立切换录制。

**原始过度声明**：声称"无法不录制就遥操作"——不实（`recorder=None` 可行）。声称 60 LOC——实际约 85 LOC，因 `vr_teleop_sim.py` 需镜像 +30 LOC。

**适配**：
- 添加 `'r' → ControlSignal.RECORD`
- BEGIN 不再启动录制（仅 `_reset_mapper()` + `state=TELEOP`）
- RECORD 独立切换 `self.recording`
- STOP/HOME/QUIT/VR-超时仍自动停止录制（安全优先）

**文件**：4 个文件（`keyboard.py`, `controller.py`, `vr_teleop_shm.py`, `vr_teleop_sim.py`） | **LOC**：~85

---

## 四、P1 发现（9 项）

### 来自 Round 1（4 项）

#### P1-1: Teleop PAUSED 重新对位（~20 LOC）

**问题**：PAUSED 状态仅冻结 IK。暂停期间移动 VR 手柄，恢复时位姿跳变。

**lerobot 方案** (`umi_teleop.py:114-133`)：`set_teleop_enabled(True, obs)` 用当前机器人位姿重置映射。

**适配**：在 `_handle_pause()` / `_handle_resume()` 中加入等效逻辑。

**文件**：`teleop/core/controller.py` | **LOC**：~20

---

#### P1-2: continuous_rotvec（~15 LOC）

**说明**：仅当策略输出是轴角（rx/ry/rz）时需要。DexMani 内部使用 rot6d 表示。作为工具函数加入 `pose_utils.py`。

**文件**：`planning/pose_utils.py` | **LOC**：~15

---

#### P1-3: 相对运动模式（~70 LOC）

**说明**：策略训练和推理使用 delta pose。好处：数据分布更集中。需配合漂移修正（EMA 回拉）。

**文件**：`planning/pose_utils.py` + eval | **LOC**：~70

---

#### P1-4: 相机 fallback（~10 LOC）

**问题**：`SharedMemoryRingBuffer.read_latest_camera()` 返回 None 时无 fallback。

**适配**：在 `FrameManager` 消费者端缓存 last_frame。

**文件**：`sensor/frame_manager.py` | **LOC**：~10

---

### 来自 Round 2（5 项，经对抗性验证修正）

#### P1-5: `KeyboardHandler.start()` 无头保护（5 LOC）

**问题**：无头服务器上崩溃并报 Xlib 错误。`dry_run=True` 不跳过键盘构造。

**原始过度声明**：声称 10 LOC，暗示可实现无头操作。**修正**：5 行保护替代密码错误为清晰 `RuntimeError`。真正的无头支持需 termios 回退（30-50+ LOC，已以 3 份内联副本存在但未共享）。

**文件**：`teleop/control/keyboard.py` | **LOC**：5

---

#### P1-6: `RealSenseConfig` 验证（12 LOC）

**问题**：`__post_init__` 仅验证 `align_mode`。缺少 FPS 边界、分辨率健全性、空序列号、L515 深度宽度 % 16 检查。

**原始过度声明**：声称 15 LOC，全部放在 `__post_init__`。**修正**：L515 可分性检查属于 `connect()`（设备发现后）。拆分为 `__post_init__`（7 LOC）+ `connect()`（5 LOC）。

**文件**：`sensor/realsense.py` | **LOC**：12

---

#### P1-7: RPY/Euler 转换工具（18 LOC）

**问题**：`rpy_to_quat_wxyz` 在 3 个文件中复制粘贴。`calibrate_camera.py` 内联使用 scipy `Rotation.from_euler` 3 次。

**适配**：在 `pose_utils.py` 中添加 `rpy_to_quat_wxyz`、`euler_xyz_to_rotmat`、`rotmat_to_euler_xyz`。替换 3 个文件中的本地定义。

**文件**：`planning/pose_utils.py` | **LOC**：18

---

#### P1-8: MultiCameraViewer 工具（~175 LOC）

**问题**：无实时多摄像头预览工具。所有构建块已存在（`MultiCameraManager`、OpenCV、`read_all_latest()`）。

**原始过度声明**：声称 120 LOC，忽略 RGB→BGR 转换、异构分辨率、None 帧处理、多进程关闭。**修正**：复用 `test_realsense.py`（625 LOC）的现有 OpenCV 模式。

**文件**：`tools/camera_viewer.py`（新建） | **LOC**：~175

---

#### P1-9: Sim 演示生成 HDF5 接入（~150 LOC）

**问题**：`pick_and_place_episode()` 已存在但不录制到 HDF5。

**原始过度声明**：声称 60 LOC，忽略基本时序不匹配——`TimestampAlignedBuffer` 按 20ms 网格分箱，但 sim 物理步进以 CPU 速度运行，导致多秒 episode 坍缩为 1-5 个网格槽。需非实时录制路径（+40 LOC）、7 个执行点需仪器化（+30 LOC）、构建器需提取（+25 LOC）。

**文件**：`examples/sim/` | **LOC**：~150

---

## 五、P2 / 延迟项

以下项目有充分文档记录，但取决于未来需求：

| 发现 | 条件 | LOC |
|------|------|-----|
| TeleopState 数据类（替代 `vr_frame: dict`） | 第二输入设备到来 | 90 |
| 配置 profile（`--dump-config` 标志） | 用户要求 YAML 工作流 | 30 |
| bimanual 架构（`DualRobotInterface`） | 第二手臂到来 | 300+ |
| TeleopDevice Protocol（完整实现） | 第三输入设备 | 350+ |
| 动作处理器管道分离 | 策略部署前 | 60 |
| V4L2 摄像头发现 | 非 RealSense 摄像头 | 50 |
| 上下文帮助键 | UX 打磨 | 8 |

---

## 六、已审查并拒绝的发现

以下 Round 2 发现经对抗性验证后被**明确拒绝**，不应实施：

| 发现 | 拒绝原因 |
|------|---------|
| **`async_read()` 及 last-frame fallback** | 等效实现在 `CameraProcess` + `RingBuffer` 层已存在。加入 `RealSense` 会鼓励绕过崩溃隔离架构。 |
| **`pose_to_matrix` / `matrix_to_pose` 辅助函数** | `pose_utils.py` 已导出 `quat_wxyz_to_rotmat`。"重复"实际是 3 行 `np.eye(4)` 样板代码。`matrix_to_pose` 零调用者——推测性开发。`config/`→`planning/` 交叉包耦合是架构倒退。 |
| **键盘遥操作"220 行重复"** | 基本特征描述错误——脚本调用共享函数（`ema_smooth_pose`、`solve_teleop_ik`、`quat_multiply`），并非重新实现。实际重叠：55-85 行编排胶水。`rpy_to_quat_wxyz`（10 行）是当前共享库中不存在的新代码。 |
| **PipelineConfig 扩展 + `from_config()`** | 会在可复现性快照中添加部署特定噪声（URDF 路径、主机/端口）。`TeleopController` 从未通过配置实例化。`CollectionConfig` 已可通过 `TeleopControllerConfig` 访问。 |
| **KeyboardHandler 按住键跟踪** | **事实错误。** 3 个目标类均已实现 `on_release` 回调和 `is_pressed()`。`KeyboardHandler`（离散 ControlSignal 事件缓冲）与 `GlobalKeyState`（持续按住键集）目的根本不同，不应合并。 |

### 被修正的 Round 1 原分析声明

| 原声明 | 修正 |
|--------|------|
| RT Report 是 DexMani 应采纳的关键技术 | ArmInnerLoop 已等效 |
| Gripper look-ahead 是可用的策略部署技术 | 代码被注释禁用 |
| continuous_rotvec 对 DexMani 是高优先级 | DexMani 用 rot6d 表示，仅策略输出轴角时需要 |
| Chunk smoothing 可消除 ~6mm 跳动 | 默认阈值=0（关闭），需手动开启 |
| blend_poses 的漂移修正有效 | 用法被注释。EMA 滤波已等效 |
| AsyncEpisodeSaver 可直接移植 | 需适配自有 HDF5 writer，非 LeRobot Dataset API |

---

## 七、统一实施路线图

```
Phase 1 (P0 快速见效, ~189 LOC)        Phase 2 (P0 基础设施, ~1,160 LOC)     Phase 3 (P1 + 策略, ~435 LOC)
┌──────────────────────────────┐       ┌──────────────────────────────┐       ┌──────────────────────────────┐
│ P0-5  from_dict gap     4    │       │ P0-10 测试脚手架      420    │       │ P0-4  Eval Loop       270    │
│ P0-6  RateManager ✅完成 5    │       │ P0-11 MockRobotIface  350    │       │ P1-1  PAUSED 重新对位   20    │
│ P0-7  _quat_to_rotvec   10    │       │ P0-12 解耦录制         85    │       │ P1-2  continuous_rotvec 15    │
│ P0-8  list_cameras       3    │       │ P0-1  慢同步           30    │       │ P1-3  相对运动          70    │
│ P0-9  scipy 移除         22    │       │ P0-2  AsyncEpisodeSaver 95   │       │ P1-4  相机 fallback      10    │
│ P0-3  Chunk 平滑         30    │       │ P0-3  (Phase 1 已做)         │       │ P1-5  无头保护            5    │
│ P1-6  Config 验证        12    │       │                              │       │ P1-7  RPY/Euler           18    │
│ P1-5  无头保护            5    │       │                              │       │ P1-8  MultiCameraViewer  175    │
│ P1-4  相机 fallback      10    │       │                              │       │ P1-9  Sim→HDF5           150    │
│ P1-7  RPY/Euler          18    │       │                              │       │                              │
│ (P1 快速见效混入)               │       │                              │       │                              │
├──────────────────────────────┤       ├──────────────────────────────┤       ├──────────────────────────────┤
│ ~119 LOC (P0) + ~45 (P1)      │       │ ~980 LOC                      │       │ ~733 LOC                      │
└──────────────────────────────┘       └──────────────────────────────┘       └──────────────────────────────┘
```

### 按文件统计

| 文件 | P0 | P1 | 类型 |
|------|----|----|------|
| `robot/inner_loop.py` | 30 | — | 修改 |
| `recording/` (2 文件) | 95 | — | 修改 |
| `robot/validate.py` | 30 | — | 修改 |
| `examples/real/policy_eval.py` | 270 | — | **新建** |
| `robot/xarm7/xarm7.py` | 2 | — | 修改 |
| `robot/xhand/xhand.py` | 2 | — | 修改 |
| `teleop/core/controller.py` | 5 | 20 | 修改 |
| `planning/pose_utils.py` | 32 | 33 | 修改 |
| `utils/signal_utils.py` | 5 | — | 修改 |
| `sensor/realsense.py` | 3 | 12 | 修改 |
| `tests/` (6 文件) | 420 | — | **新建** |
| `robot/mock_interface.py` | 350 | — | **新建** |
| `teleop/control/keyboard.py` | — | 5 | 修改 |
| `sensor/frame_manager.py` | — | 10 | 修改 |
| `tools/camera_viewer.py` | — | 175 | **新建** |
| `examples/sim/` | — | 150 | 修改 |
| `teleop/core/controller.py`（录制） | 85 | — | 修改 |
| **合计** | **~1,349** | **~435** | |

---

## 八、隐蔽成本

以下为对抗性验证中发现的、原始 LOC 估算未计入的成本：

### 架构漂移风险
- `MockRobotInterface`：`RobotInterface.__init__` 创建 `WorkspaceSafety`、pinocchio 运动学及 MPlib 碰撞模型——mock 需初始化这些（使 CI 需依赖）或完全覆盖 `__init__`
- Sim→HDF5 时序不匹配：`TimestampAlignedBuffer` 假定实时 50Hz 循环——sim 步进需在各航点间人工 `sleep(0.02)` 或单独的非实时录制路径
- 录制解耦：`vr_teleop_sim.py` 的状态机有相同耦合（原始估算未计入 +30 LOC 镜像）

### 静默失败风险
- `KeyboardHandler` "warn and skip" 模式产生永久静默的空 `poll()` 结果——比崩溃更难调试
- L515 可分性检查：在 `__post_init__` 中会破坏非 L515 配置（需摄像头型号感知守卫）
- `rotmat_to_quat` 分析解转换存在已知数值边界情况（迹接近 -1），naive 实现会静默损坏

### 测试负担（原始 LOC 估算未计入）
- Controller 状态机测试需 mock 5 个外部依赖的 20+ 方法
- 速率限制器时序精度断言在 CI 中固有地不稳定（时钟分辨率、调度器方差因机器而异）
- `MockRobotInterface`：合成状态需形状正确且有限，回放模式帧迭代需正确

### 集成涟漪
- 无头保护：3+ 入口点会有变化的故障模式（清晰 `RuntimeError` 替代 Xlib 崩溃）
- 录制解耦：STOP/HOME/QUIT/VR-超时全部自动停止录制——解耦必须保留此行为
- `from_dict`：`from_dict_helper` 已处理 list→ndarray、float/int/bool 透传及缺失键→默认值——切换至 `FromDictMixin` 必须保留此行为

---

## 附录：被修正的 Round 2 过度声明

| 发现 | 原始 LOC | 修正 LOC | 原因 |
|------|---------|---------|------|
| from_dict gap | 30 | 4 | 4 行修复 vs. 声称 30 行 |
| RateManager 接入 | — | 5 | 已确认（CONFIRMED） |
| _quat_to_rotvec | — | 10 | 已确认（CONFIRMED） |
| list_cameras 过滤 | — | 3 | 已确认（CONFIRMED） |
| scipy 移除 | — | 22 | 已确认（CONFIRMED） |
| 测试脚手架 | 200 | 420 | 估算严重不足 |
| MockRobotInterface | 250 | 350 | 缺少子对象 + 回放模式 |
| 录制解耦 | 60 | 85 | sim 镜像 + 安全交互 |
| TeleopDevice Protocol | 500 | 90（P2） | 对 1 个实现过度设计 |
| 键盘遥操作重复 | 220 | 0 | 特征描述错误的共享函数调用 |
| 演示 HDF5 接入 | 60 | 150 | 时序不匹配 + 7 个执行点 |
| MultiCameraViewer | 120 | 175 | 缺少 BGR、分辨率、关闭逻辑 |
| PipelineConfig 扩展 | 120 | 0 | 部署噪声 + 零使用 |
| 配置 profiles | 60 | 30（P2） | 基础设施已存在 |
| KeyboardHandler 按住键 | 129 | 0 | 事实错误 |
| RPY/Euler 工具 | 25 | 18 | 合并范围校正 |
| 无头保护 | 10 | 5 | 保护 vs. 实现 |
| RealSenseConfig 验证 | 15 | 12 | 拆分位置校正 |
| `async_read()` | 8 | 0 | 等效已存在 |
| `pose_to_matrix` | 15 | 0 | 3 行琐碎代码 |

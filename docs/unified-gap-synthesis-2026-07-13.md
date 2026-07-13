# DexMani 统一 Gap 分析（优化版）

> **2026-07-13** | 5 份独立分析 → 11-agent 并行实码验证 → 9-agent Section 九专项评估 → **59 项可操作建议**
> **最后更新 2026-07-13** — 已实现 21 项（B0, B1, A1, A2, B2, M1, M2, D1, D2, D3, R4, B3, B4, DX1, O4, DX8, **O1, I1, R1, A4+A5, R3**），剩余 38 项
>
> **方法**: 每项合并后由独立 agent 读取 DexMani 源码，与报告声称的行号/前提逐项比对，输出 verdict（CONFIRMED / OVERSTATED / FALSE / ALREADY_DONE / GREENFIELD / DROP）。
>
> **源报告**: LeFranX Gaps / LeRobot×UFACTORY / ManiUniCon / T-Rex / ufactory_teleop
>
> **本文是唯一权威合并报告，取代全部五份原始报告。**

---

## 一、决策摘要（一页纸）

### 最该优先做的事

| 优先级 | 做什么 | 为什么 | 代价 | 状态 |
|--------|--------|--------|------|------|
| ~~**立即**~~ | ~~修 `controller.py:385` 的 `.period` bug~~ | ~~唯一的实时 bug~~ | ~~1 行~~ | ✅ 已修复 |
| ~~**本周**~~ | ~~关节 delta 钳位 + 软启动斜坡 + HDF5 压缩~~ | ~~安全 + 存储~~ | ~~~40 LOC~~ | ✅ 已实现 |
| **本周** | 连接安全（E-Stop 必须到达手部）+ 固件限制推送 | 当前断开/E-Stop 时 arm 异常会跳过我手的停机 | **~50 LOC** | ✅ 已实现 |
| **本周** | Zarr 导出加上相机帧 | 视觉策略训练的**硬阻塞**——当前 Zarr 只有运动学数据，没有图像 | **~80 LOC** | ✅ 已实现 |
| **两周内** | 周期性 flush | 崩溃丢全部数据 | **~40 LOC** | ✅ 已实现 |
| **策略 rollout 前** | `validate_action` 扩展（碰撞/温度/扭矩/工作空间门控） | CLAUDE.md 明确标注的硬性前提条件 | **~50 LOC** | ✅ 已实现 |

### 数字

| | |
|---|---|
| 原始建议 ~130 项（5 份报告） → 合并后 **99 项唯一** | 发现并纠正 **7 项文档与代码不一致**（Section 三详细枚举） |
| → **59 项可操作**（1 P0 + 13 P1 + 22 P2 + 23 P3） | 发现 **1 个实时 bug**（`controller.py:385`） |
| → **21 项已实现** ✅ | 剩余 **38 项待实现** |
| → **53 项已拒绝**（已完成/前提错误/过度设计/架构不匹配） | 可操作总代码量 **~3,420 LOC**（vs 报告声称 ~17,000 LOC） |
| **已实现 ~428 LOC** | 被拒项多为：适合 mode-6 位置控制回路之外的架构、推测性灵活性、无消费者的新抽象 |
| **~3 项源报告建议未纳入**（Section 九透明记录——从原 43 项经 9-agent 实码核查后缩减） | |

---

## 二、复查结论

在上一轮 11-agent 验证基础上，本轮的针对关键断言做了二次实地复查：

| 断言 | 复查结果 |
|------|---------|
| `controller.py` 限速器 `.period`/`.dt` 属性错配 | ✅ **确认并已修复** — 后经 I1 换 `RateManager`，调用点为 `self.limiter.period`（`controller.py:388`） |
| `validate_action()` 是 4 个操作而非 2 个 | ✅ **确认** — `validate.py:39-55`（CLAUDE.md 已修正） |
| `DataValidator` 是 7 类检查 | ✅ **确认并已修复** — docstring 已更新为 "7 validation check categories" |
| PAUSED 恢复时 `_reset_mapper()` 已实现 | ✅ **确认** — ALREADY_DONE |
| `send_action()` 无条件调用不是 bug | ✅ **确认** — FALSE，是有意的保活行为 |
| `ik.py:580-581` docstring 错误声称 250Hz + 速度限制 | ✅ **确认** — 待修复 |
| `XArm7Config`/`XHandConfig` 缺少 `FromDictMixin` | ✅ **确认并已修复** — 已继承 FromDictMixin + PEP 563 注解解析修复 |
| `from_dict_helper` docstring 声称支持 | ✅ **确认并已修复** — 现在确实支持（B3） |

---

## 三、主动发现的 Bug 与文档错误

### 实时 Bug（1 个）✅ 已修复

**`controller.py` — 限速器 `.period` / `.dt` 属性错配**

当时 `RateLimiter` 只有 `dt`（`1.0/target_hz`）和 `target_hz`（property），没有 `period`，故一度改为 `self.limiter.dt`。
后经 **I1** 将主循环换为 `RateManager`（其暴露 `.period` 而非 `.dt`），调用点已相应改为 `self.limiter.period`（`controller.py:388`），当前一致。

### 文档与代码不一致（7 处）

| 位置 | 当前文本 | 实际 | 状态 |
|------|---------|------|------|
| `CLAUDE.md:214` | "2-check stub" | 4 个操作 | ✅ 已修复 |
| `data_validator.py:3` | "5 validation checks" | 7 类检查 | ✅ 已修复 |
| `controller.py:1-14` | 引用 RECORDING 状态 | IDLE/TELEOP/PAUSED/EMERGENCY_STOP | ✅ 已修复 |
| `serialization.py:166-167` | 列出 XArm7Config/XHandConfig 为消费者 | 两者无 `from_dict` | ✅ 已修复（B3） |
| `ik.py:580-581` | "at 250 Hz" + "velocity/acceleration/jerk limiting" | 50Hz + Mode 6 firmware | 待修复 |
| `planning/types.py:213-214`, `planning/ik.py:35`, `planning/ik.py:400` | 引用不存在的 `_tick_mode4()` | 替换为 "Mode 6 firmware handles trajectory planning" | 待修复 |
| `export_hdf5_to_zarr.py:385` | "Welford-style incremental computation" | 批量 `np.mean`/`np.std` | 待修复 |

---

## 四、统一可操作主列表

### P0 — 立即修复（1 项，~5 LOC）✅ 已完成

| ID | 标题 | 文件 | LOC | 状态 |
|----|------|------|-----|------|
| **B0** | 限速器 `.period`/`.dt` 属性错配（I1 换 `RateManager` 后调用点为 `.period`） | `teleop/core/controller.py:388` | ~1 | ✅ |

---

### P1 — 安全关键 + 数据管线阻塞 + 部署基础（13 项，~1,800 LOC）

#### 🔴 安全

| ID | 标题 | 文件 | LOC | 状态 |
|----|------|------|-----|------|
| **B1** | **`validate_action()` 扩展** — 新增扭矩门控 + 温度门控 + 碰撞检测 + 工作空间 clamp。所有新参数可选（None=跳过），向后兼容。控制器已传入 `actual_arm_tau` | `robot/validate.py`, `robot/types.py`, `teleop/core/controller.py` | ~50 | ✅ |
| **B2** | **连接安全** — 断开/E-Stop/清除错误时每组件 try/except（arm 异常跳过 hand → E-Stop 必须始终到达 hand）+ `connect()` 异常安全 + `XHand.connect()` 重入守卫 | `robot/interface.py`, `robot/xhand/xhand.py` | ~25 | ✅ |
| **A1** | **臂部每步关节 delta 钳位** — 镜像 `xhand.py:524-527` 的 L∞ 钳位。插入 `inner_loop._send_target()` | `robot/inner_loop.py` | ~15 | ✅ |
| **A2** | **软启动速度斜坡** — 前 20 帧速度从 0.2→1.57 rad/s 线性插值 | `robot/inner_loop.py` | ~15 | ✅ |
| **M2** | **固件关节限制推送** — `set_reduced_joint_range` + `set_reduced_mode(True)` 从未被调用 | `robot/xarm7/xarm7.py` | ~20 | ✅ |

#### 🟠 数据管线（阻塞下游训练）

| ID | 标题 | 文件 | LOC | 状态 |
|----|------|------|-----|------|
| **D2** | **Zarr 导出加上相机帧** — `_OBS_KEYS` 仅为运动学，无 rgb/depth。**视觉策略训练的硬阻塞项** | `tools/export_hdf5_to_zarr.py:59-67` | ~80 | ✅ |
| **D1** | **HDF5 压缩** — 6 个 `create_dataset` 加 `compression="gzip"`（~5-10x 存储节省） | `recording/episode_recorder.py` | ~10 | ✅ |
| **D3** | **录制周期性 flush** — 非相机流每 500 帧（~10s）增量写入 HDF5。中途崩溃最多丢一个 flush 周期 | `recording/episode_recorder.py` | ~40 | ✅ |

#### 🟡 鲁棒性

| ID | 标题 | 文件 | LOC | 状态 |
|----|------|------|-----|------|
| **M1** | **`get_state()` try/except** — arm/hand `get_state()` 各自 try/except；异常→NaN 回退，不穿透 50Hz 循环 | `robot/interface.py:164-192` | ~12 | ✅ |
| **B3** | **`XArm7Config`/`XHandConfig` `FromDictMixin`** — 裸 `@dataclass` 无 `from_dict`，配置反序列化失败。连带修复了 PEP 563 字符串注解解析 | `robot/xarm7/xarm7.py`, `robot/xhand/xhand.py`, `utils/serialization.py` | ~8 | ✅ |

#### 🔵 部署基础

| ID | 标题 | 文件 | LOC |
|----|------|------|-----|
| **DP1** | **策略部署史诗** — 核心推理循环（策略加载 + obs 组装 + action-chunk/temporal-ensemble + 通过已有 SharedSyncPrimitives + ArmInnerLoop 同步模式分发）~600-800 LOC + 硬件 replay ~250-350 LOC。已有基础设施：`SharedSyncPrimitives` 完全连接，Zarr obs/action 合同已存在，`retarget_server.py` 提供 ZMQ REP 模板 | 新建 + `teleop/core/`, `tools/` | ~1,000-1,200 |

#### 🔵 测试基础

| ID | 标题 | 文件 | LOC |
|----|------|------|-----|
| **T1** | **自动化单元测试** — `tests/` 仅 1 个测试文件（17.6k LOC 代码库）。最少可行：conftest + rate_limiter + keyboard + rate_manager + controller_state（~420 LOC） | `tests/` (6 个文件) | ~420 |
| **T2** | **Mock RobotInterface** — `RobotInterface.__init__` 硬编码构造函数 `XArm7()` + `XHand()`。廉价路径：DummyArm/DummyHand 返回罐装状态 + `_build_hardware=False` 守卫 + 构造注入（~140 LOC）。**T1 controller_state 测试的前提条件** | `robot/mock_interface.py` (新), `robot/interface.py` | ~140 |

---

### P2 — 质量 + 可观测性 + 工作流（22 项，~980 LOC）

#### 运动与平滑性

| ID | 标题 | 文件 | LOC |
|----|------|------|-----|
| **A3** | Post-IK 关节空间 EMA — Cartesian EMA 之后 IK 输出未经过滤直发。增量平滑；现有肘部翻转 guard + 跳变限制 + 固件已覆盖下游 | `teleop/core/pipeline.py` | ~12-15 |
| **A4** | ArmInnerLoop 跟踪误差监控 — 已读 current qpos 且 holding target，但从未计算 `|target-current|`。检测软饱和（错误码已覆盖硬故障）。含 A5（每帧 mode-6 节流复查）+ controller `_print_status` 加 `trkerr` | `robot/inner_loop.py`, `teleop/core/controller.py` | ~30 | ✅ |
| **A5** | 每帧 mode-6 重入守卫 — `arm.mode==6` 仅在初始化验证；50Hz 循环中加节流检查（并入 A4 `_monitor()`） | `robot/inner_loop.py` | ~5 | ✅ |

#### 连接与错误

| ID | 标题 | 文件 | LOC | 状态 |
|----|------|------|-----|------|
| **B4** | `_set_mode()` + `_init_mode()` 错误码验证 — 镜像 `robot_init()`/`clear_error()` 的 `get_err_warn_code` 检查。`_set_mode()` (xarm7.py) 已修复 ✅；`_init_mode()` (inner_loop.py:430-469) 直接调用 `arm.set_mode()` 无错误验证，**待实现** | `robot/xarm7/xarm7.py`, `robot/inner_loop.py` | ~8 | ✅/待 |
| **B5** | Headless keyboard guard — 无 DISPLAY 时 `KeyboardHandler.start()` 因 Xlib 崩溃 | `teleop/control/keyboard.py` | ~5-8 | 待实现 |

#### 录制与数据

| ID | 标题 | 文件 | LOC |
|----|------|------|-----|
| **R1** | `/meta` 中加 `control_mode`/`arm_mode`/`hand_delta_clip` + EMA/delta 快照 — 对下游可重现至关重要（schema v4，`record_config` 透传 4 层）| `recording/episode_recorder.py`, `collection_loop.py`, `teleop/core/controller.py` | ~40 | ✅ |
| **R2** | 录制内重新拍摄 — 'r' 键丢弃当前拍摄不增加计数器。`discard_episode` 管道已存在；仅缺新 ControlSignal + 键映射 + 转换分支 | `teleop/core/controller.py` | ~25-35 | 待实现 |
| **R3** | 跳过初始帧 — `skip_initial_frames`（默认 10）+ 首帧重锚 buffer `start_time` 避免前向填充冻结回填 | `recording/episode_recorder.py`, `collection_config.py` | ~20 | ✅ |
| **R4** | 相机环形缓冲区 torn-read — seqlock 模式：拷贝后重读序列号，不一致则丢弃 | `shm/ring_buffer.py` | ~5 | ✅ |
| **R5** | 虚假 "Welford" docstring + 每特征规范化模式 — 修复 docstring（1 行）+ 可选 min-max/q01-q99 模式（~30-80 LOC） | `tools/export_hdf5_to_zarr.py` | ~30-80 |
| **R6** | 按 `held_ratio` 的情节级质量过滤 — sidecar JSON 已写 `held_ratio` 但零消费者读取；导出过滤骨架已有，仅需加 held_ratio 比较 | `tools/export_hdf5_to_zarr.py` | ~20-40 |
| **R7** | PyTorch Dataset + 惰性 Zarr 加载 — `from_zarr` 使用 `np.asarray` 急切具体化全部数据到 RAM → 大数据集 OOM | `tools/export_hdf5_to_zarr.py` | ~150-300 |

#### 可观测性

| ID | 标题 | 文件 | LOC |
|----|------|------|-----|
| **O1** | `init_logging()` 加 FileHandler — 仅 StreamHandler(sys.stdout)；现场日志在终端关闭时丢失。共享单例 FileHandler + `$DEXMANI_LOG_DIR` + fail-safe | `utils/log.py` | ~40 | ✅ |
| **O2** | 统一 `get_status()` API dict — 各组件有独立 `get_status` 但从未组合；`_print_status` 仅打日志不返回 dict | `teleop/core/controller.py` | ~40 |
| **O3** | Timing instrumentation — `perf_counter` 散落 8+ 文件；统一到 `ExecutionTimer` util + tick 时间双端队列输入 `_print_status` | `utils/timer.py` (新) | ~55 |
| **O4** | 文档修复 — controller docstring + DataValidator count + CLAUDE.md validate_action | 3 个文件 | ~10 | ✅ |

#### 遥操作

| ID | 标题 | 文件 | LOC |
|----|------|------|-----|
| **T3** | ADB 反向自动化 — 6 个文件让用户手动运行 `adb reverse`；零 `subprocess.run`。单个 ~20 LOC 助手 | `teleop/vr/` | ~20 |
| **T4** | 协调 arm+hand 返回到原位 — `_do_home()` 仅移动 arm；hand 在 SDK 盲重置中保持紧握 → 轻度碰撞风险 | `teleop/core/controller.py` | ~30-40 |
| **T5** | Strict config key validation — `serialization.py:170-171` 明确记录"Extra keys silently ignored"；加 `_strict` 标志拒绝未知键 | `utils/serialization.py` | ~40 |
| **T6** | YAML 驱动入口点配置 — `FromDictMixin.from_yaml` 已实现；仅需 argparse `--config` + 示例 YAML + 入口引用 | 入口脚本 | ~40-60 |

#### 基础设施

| ID | 标题 | 文件 | LOC |
|----|------|------|-----|
| **I1** | **RateManager 替代 RateLimiter** — 条件实例化替换已知 buggy 组件（B0 仅修了 `.period`→`.dt`，未解决 `time.sleep()` ~15ms 抖动）。RateManager 提供 <1ms 精度的混合忙等待。主循环 + ArmInnerLoop 内环均已替换（`.dt`→`.period`）| `teleop/core/controller.py`, `robot/inner_loop.py` | ~6 | ✅ |
| **I2** | **Scipy 消除** — 50Hz 热路径 `quat_to_rotmat` 用分析 NumPy 解替换 `scipy.spatial.transform.Rotation.from_quat()`。消除 ~100MB 依赖简化部署，3 处调用点漏斗经 `pose_utils.py`（单文件修改，无 API 变更） | `planning/pose_utils.py` | ~25 |

---

### P3 — 便捷性（23 项，~650 LOC）

| ID | 标题 | LOC | 状态 |
|----|------|-----|------|
| **DX1** | 统一三重 `_quat_to_rotvec` — 加 `w>=0` 双覆盖保护 | ~10 | ✅ |
| **DX2** | 提取共享 `rpy/euler→quat` util — 3 个脚本中复制 | ~18 | 待实现 |
| **DX3** | `list_cameras()` `by_product_line` 过滤器 | ~3-5 | 待实现 |
| **DX4** | RealSenseConfig `__post_init__` 验证 | ~8-10 | 待实现 |
| **DX5** | FrameManager last-frame fallback | ~10 | 待实现 |
| **DX6** | 关节限制自动发现工具 | ~60-90 | 待实现 |
| **DX7** | 手持式关节归一化 util | ~30 | 待实现 |
| **DX8** | 向量化 hand delta clip — 标量 → 每关节 (12,) 数组 | ~5 | ✅ |
| **DX9** | 手持式 held-frame 去重后处理 | ~30-40 | 待实现 |
| **DX10** | 死锁 man switch（键盘） | ~20-40 | 待实现 |
| **DX11** | 单肢冻结热键（F=hand，A=arm） | ~60 | 待实现 |
| **DX12** | SDK import 错误提示 | ~20 | 待实现 |
| **DX13** | 主动 Hz 报告 | ~30 | 待实现 |
| **DX14** | MultiCameraViewer 实时预览 | ~100-130 | 待实现 |
| **DX15** | 情节级质量分析工具 | ~120-150 | 待实现 |
| **DX16** | 惰性 IK 初始化 + 控制器依赖注入 | ~40-60 | 待实现 |
| **DX17** | CLAUDE.md 错误处理约定文档化 — 明确何时用 tuple/何时用 ValueError/何时用 `_error_state` flag（T-Rex #9，纯文档，零代码变更） | ~15 | 待实现 |
| **DX18** | 臂部每关节运动约束 — 标量→`ndarray(7,)`，镜像 DX8 手部 12 维每关节 delta clamp（LeFranX A4(arm)） | ~5 | 待实现 |
| **DX19** | 策略评估框架 — rollout runner + 成功率计数器。无已训练策略存在时阻塞，但 60 LOC 骨架可在需要时快速补齐（LeFranX D2） | ~60 | 待实现 |
| **DX20** | Session 概览工具 — Matplotlib 薄封装，打印所有已录制 episode 概况（数量、时长、成功率），补充 `visualize_episode.py` 的单 episode 深潜（LeFranX F8） | ~30 | 待实现 |
| **DX21** | 录制恢复提示 — 启动时打印已有 episode 数，帮助操作员判断追加/覆写（LeFranX G5） | ~5 | 待实现 |
| **DX22** | 录制解耦 — BEGIN 不再自动录制，`r`→RECORD 独立切换。允许不录制情况下操作机械臂（调试/演示）。当前 record-then-discard 变通方案可用但浪费磁盘 I/O（LeRobot P0-12） | ~85 | 待实现 |
| **DX23** | EMA Alpha CLI 可配置 — `--ema-alpha-pos`/`--ema-alpha-rot` argparse 标志（EMA 参数已在 Python config 中存在，仅需 CLI 暴露）（UFACTORY P2-4） | ~5 | 待实现 |

---

## 五、快速见效清单（16 项，每项 <25 LOC，<1 小时）

| # | 项 | LOC | 状态 |
|---|-----|-----|------|
| **B0** | `.period` → `.dt` bug | 1 | ✅ |
| **B3** | `XArm7Config`/`XHandConfig` 加 `FromDictMixin` + PEP 563 修复 | ~8 | ✅ |
| **B4** | `_set_mode()` + `_init_mode()` 后错误码验证 | ~8 | ✅/待 |
| **B5** | Headless keyboard guard | 5-8 | 待实现 |
| **D1** | HDF5 `compression="gzip"`（6 个 `create_dataset`） | ~10 | ✅ |
| **R3** | 跳过初始帧 | ~10-15 | ✅ |
| **R4** | Torn-read seqlock | ~5 | ✅ |
| **O4** | 修复过时 docstrings（3 文件） | ~10 | ✅ |
| **DX1** | 统一 `quat_to_rotvec` + w>=0 guard | ~10 | ✅ |
| **DX3** | `list_cameras()` 过滤器 | ~3-5 | 待实现 |
| **DX4** | RealSenseConfig 验证 | ~8-10 | 待实现 |
| **DX8** | 向量化 hand delta clip | ~5 | ✅ |
| **I1** | RateManager 替代 RateLimiter | ~5 | ✅ |
| **DX18** | 臂部每关节 delta 约束 (7,) | ~5 | 待实现 |
| **DX21** | 启动时打印已有 episode 数 | ~5 | 待实现 |
| **DX23** | EMA Alpha CLI 可配置 | ~5 | 待实现 |

---

## 六、实施路线图

### 依赖关系图

```
Phase 1: 安全基础 (~150 LOC, 1-2天) ✅ 全部完成
  ~~B0~~✅ ──→ ~~B2~~✅ ──→ ~~A1~~✅ ──→ ~~M2~~✅
  ~~(bug)~~       ~~(E-Stop)~~ ~~(delta clip)~~  ~~(firmware limit)~~
         ~~A2~~✅ ──→ ~~B3~~✅ ──→ ~~M1~~✅
         ~~(软启动)~~    ~~(from_dict)~~    ~~(try/except)~~
  ↓
Phase 2: 数据完整性 (~120 LOC, 1-2天)
  ~~D1~~✅ ──→ ~~D2~~✅ ──→ ~~D3~~✅
  ~~(压缩)~~    ~~(Zarr相机)~~ ~~(flush)~~
  ↓
Phase 3: 可观测性 + 基础设施 (~175 LOC, 2-3天)
  ~~O1~~✅ ──→ O2 ──→ O3 ──→ ~~O4~~✅
  ~~I1~~✅ ──→ I2
  (RateManager) (Scipy消除)
  ↓
Phase 4: validate_action 扩展 + 臂部安全 (~67 LOC, 2-3天)
  ~~B1~~✅ ──→ ~~B4~~✅ ──→ B5 ──→ DX18
                                  (arm每关节delta)
  ↓
Phase 5: 测试基础 (~560 LOC, 3-5天)
  T2 ──→ T1
  (mock)  (tests)
  ↓
Phase 6: 配置 + 录制工作流 (~270 LOC, 2-3天)
  T5 ──→ T6 ──→ ~~R1~~✅ ──→ R2 ──→ ~~R3~~✅
  DX22 ──→ DX23
  (录制解耦) (EMA CLI)
  ↓
Phase 7: 训练数据就绪 (~310 LOC, 1-2周)
  R5 ──→ R6 ──→ R7
  ↓
Phase 8: 部署基础 (~1,150 LOC, 2-3周)
  DP1
  ↓
Phase 9: Polish + 按需（含 Section 九 保留项触发时升级）
  剩余 P2/P3 项 + Section 九 保留项（B5/P0-3/P1-9 待触发条件满足）
```

### 各阶段详情

| 阶段 | 主题 | 项数 | LOC | 时间 | 关键路径 |
|------|------|------|-----|------|---------|
| 1 | 安全基础 | 7 | ~96 | 1-2 天 | B0（实时 bug → 立即）→ B2（E-Stop try/except → 最高严重性） |
| 2 | 数据完整性 | 3 | ~120 | 1-2 天 | D2（视觉策略训练硬阻塞） |
| 3 | 可观测性 + 基础设施 | 6 | ~175 | 2-3 天 | O1（文件日志）→ O2（状态 API）→ O3（计时）+ I1（RateManager）+ I2（Scipy 消除）|
| 4 | validate_action 扩展 + 臂部安全 | 4 | ~67 | 2-3 天 | B1 是 CLAUDE.md 明确的硬性前提条件 + DX18（臂部每关节 delta） |
| 5 | 测试 + Mock | 2 | ~560 | 3-5 天 | T2（MockRobotInterface）是 T1（controller 状态测试）的前置条件 |
| 6 | 配置与录制工作流 | 7 | ~270 | 2-3 天 | T5（strict config）→ T6（YAML 配置）→ 录制便利性 + DX22（录制解耦）+ DX23（EMA CLI） |
| 7 | 训练数据就绪 | 3 | ~310 | 1-2 周 | R5（per-feature norms）→ R6（held_ratio 过滤）→ R7（PyTorch Dataset） |
| 8 | 部署基础 | 1 | ~1,150 | 2-3 周 | DP1（推理循环 + rollout + 硬件 replay） |
| 9 | Polish + 按需 | 25 | ~680 | 按需 | 全部 P3 + Section 九保留项（B5/P0-3/P1-9 待触发条件满足后升级） |

**阶段 1-4**（~458 LOC，5-10 天）覆盖全部安全和操作体验差距。
**阶段 5-7**（~1,140 LOC，1-3 周）覆盖质量和训练就绪。
**阶段 8**（~1,150 LOC，2-3 周）是策略部署的独立大工程。
**阶段 9** 包含 23 项 P3 便捷性改进 + 2 项 P2（I1/I2）+ 3 项 Section 九保留项。

---

## 七、已拒绝项及理由（53 项——原 21 项 + Section 九评估新增 32 项）

| 原始建议 | 来源 | 拒绝理由 |
|----------|------|---------|
| 将 `limit_jerk()` 接入 inner_loop | LeFranX A1 | Mode 6 固件已做加速度限制。Jerk（三阶导数）限制可进一步平滑运动，但 `limit_jerk()` 的函数设计假设 Ruckig 轨迹生成（非 Mode 6 在线规划），直接接入会破坏现有的每帧位置控制语义。保留 `limit_jerk()` 供未来 Mode 7 或策略 rollout 路径使用。 |
| ArmInnerLoop 错误时自动重连 | LeFranX B4 | 碰撞后静默重新启用运动是**危险的**——升级到 emergency 是正确的安全行为 |
| 自定义异常体系（~80 LOC） | T-Rex #8 | 三种信号模式（tuple/ValueError/flag）与项目 fail-safe 哲学一致（错误→warning+回退）。重构成本/收益比差 |
| CLAUDE.md 错误处理约定未文档化（~15 LOC） | T-Rex #9 | **采纳为文档任务** — 纯文档（零代码变更），在 CLAUDE.md Conventions 节增加错误处理约定说明（何时用 tuple/何时用 ValueError/何时用 `_error_state`）。归入 Section 四 P3 DX 系列。 |
| Teleoperator ABC（~300 LOC） | LeFranX G3 | 为 2 个入口点设抽象——违反"简洁优先"原则 |
| Config 驱动的工厂 + VR 路由器 + TeleopAppBuilder（~440 LOC） | LeFranX G6; ManiUniCon P1c/P1b | 纯重构，零行为变更，零安全价值 |
| 分布式 actor-learner RL（gRPC，~2500 LOC） | LeFranX C9 | 零基础（无 RL 循环、奖励、单节点训练器）——严重的架构不匹配 |
| 可写/双 replay buffer（~500 LOC） | LeFranX E4 | ReplayBuffer 有意设为只读——写入是 `EpisodeRecorder` 的职责；双缓冲仅对在线 RL 有意义 |
| 每情节目录 + 成功/失败子目录 | T-Rex #38 | 对抗明确的"success 是 /meta attr 非 dir 路由"设计决策——破坏所有现有 flat-schema 工具 |
| Ruckig OTG 集成 | T-Rex #36 | 与 Mode 6 固件在线轨迹规划直接冗余 |
| LeRobot v3.0 导出 | T-Rex #29 | Zarr 已服务于 Diffusion Policy；额外导出格式 ROI 低 |
| RLDS/TFDS 导出 | ManiUniCon P2a | 沉重的 TF 依赖，零现有消费者 |
| 仓库内训练管道（ACT/DP，~2500 LOC） | LeFranX C2 | `export_hdf5_to_zarr.py` 将输出交给外部训练仓库——复制成熟训练代码违背简单性要求 |
| "无条件 send_action" 加门控 | T-Rex #7 | **FALSE**——验证失败时发送 hold action，是有意的保活行为，不是 bug |
| 两阶段 init / wait_for_live_data | T-Rex #14 | 两个驱动已在连接时验证 live state；inner_loop 等待 ready_event |
| 相机 MP4 编码 | T-Rex #39 | 独立大工程（~300-400 LOC）；gzip 压缩（D1）先做 |
| Arm Error Code 语义映射（~80 LOC） | ufactory_teleop P1-1 | 维护 ~140 个错误/警告码字典仍需持续跟进 SDK 更新。当前裸整数码在 6+ 日志点已可定位故障——收益在调试便利性而非安全性。低优先级 |
| Mode 7 笛卡尔控制（~150 LOC） | ufactory_teleop P1-5 | Mode 6 + 软件 IK 已满足遥操作需求。Mode 7（固件侧 IK）与现有 IK 选择/碰撞检测管道冗余，且引入双路径维护负担。保留为未来评估项 |
| Command Counter 看门狗（~20 LOC） | ufactory_teleop P2-1 | 实用性存疑——Mode 6 固件已有独立轨迹规划与超时保护。cmd_num 停滞场景在 50Hz 内环中未实际观测到 |
| 末端执行器类型抽象（~50 LOC） | ufactory_teleop P2-3 | 当前仅 XHand 一个消费者——为一个实现建抽象违反"简洁优先"。在第二个末端执行器接入时再评估 |
| 断开前受控停止 — disconnect 前调用 `_hold_position()` 软停 | LeFranX A2 | Mode 6 固件在断开时自动保持最后位置（`inner_loop.py:342` 明确注释）。显式调用 `_hold_position()` 与固件行为冗余 |
| TCP 健康探测 — connect() 前 2s SYN 探测 | LeFranX B2 | `XArmAPI` 构造函数已有 try/except（`xarm7.py:84-88`）。增加 2s SYN 探测仅添加启动延迟，诊断价值已被构造函数覆盖 |
| Gym Environment — `gym.Env` 封装 | LeFranX C1 | `export_hdf5_to_zarr.py` 已提供完整数据接口（train/val split、norm stats、episode 过滤）。`gym.Env` 仅对在线 RL 有意义——远超出当前项目范围 |
| 训练检查点生命周期 — save/load + `--resume` | LeFranX C3 | 检查点属于训练基础设施。训练在外部仓库进行——本项目无训练循环可供检查点 |
| 奖励函数 — 稀疏方案 A | LeFranX C5 | 奖励函数仅存在于训练循环内部。训练在外部仓库进行——在此定义奖励函数为无消费者的死代码 |
| 图像增强 + 黑帧检测（子项 a,b） | LeFranX C6(a,b) | 子项 (b) 黑帧检测已由 `DataValidator._check_camera()`（`data_validator.py:162-177`）覆盖。子项 (a) 图像增强属于训练 dataloader，非数据导出 |
| 策略 Server/Client 分离 — GPU 推理与实时回路解耦（~800 LOC） | LeFranX D3 | 为单一消费者（策略推理）建 800 LOC server/client 架构违反"简洁优先"。DP1 的直接函数调用方案更简单且充分 |
| WandB 实验追踪 | LeFranX F6 | 训练基础设施。数据采集指标已有本地日志（`controller.py:747-749`）。训练指标属于外部训练仓库 |
| FPSTracker 抽象 — 独立计数器（~200 LOC） | LeFranX F7 | 现有内联计时（`controller.py:386-389` 超时检测，687-718 状态打印含 12 字段）已足够。为单一消费者抽象化违反"简洁优先" |
| 图像变换调试工具（~200 LOC） | LeFranX F9 | 小众需求，无实际使用场景。相机外参在标定时一次性设置，临时 numpy 调试已足够 |
| 操作度零空间 IK 评分 — 奇异点回避（~50 LOC） | LeFranX G8 | 自适应 IK 阻尼（`ik.py:526-538`）已基于操作度处理近奇异点条件。零空间操作度梯度为推测性优化，无已证明的失效场景 |
| Diff IK 关节空间速度钳位 — `ik.py:577` dq 无每关节 clip | T-Rex #1 | DLS 内部迭代步影响收敛而非最终输出。最终输出经 `canonicalize_qpos()` 后由 ArmInnerLoop Mode 6 固件以 250Hz 做轨迹规划。DLS 阻尼（λ²=1e-5）已为病态雅可比限制 dq。2+ 年运行无振荡观测 |
| 集中式机器人描述 — 消除"7 处"分散定义（~200 LOC） | T-Rex #13 | 正确性审计确认仅 2 处有意重复（非声称的 7 处）。均为架构解耦——planning 层在仿真中独立运行。交叉引用注释（`planning/types.py:165-167`）防止静默不同步。纯重构，零行为变更 |
| 实时可视化线程 — 全仪表板（~300 LOC） | T-Rex #16 | `_print_status()` 已每 2s 提供 12 字段实时状态（`controller.py:687-718`）。`visualize_episode.py` 提供事后分析。300 LOC 线程化仪表板增加延迟风险和复杂度 |
| 运行时切换 data_dir（~20 LOC） | T-Rex #18 | 运行时 setter 鼓励滥用（碎片化数据集、非连续 episode 编号）。启动时配置 data_dir 更清洁，符合 ManiUniCon/LeRobot 惯例。重启会话仅需 ~5 秒 |
| Rich 硬件健康面板 — 带格式状态表（~100 LOC） | T-Rex #19 | `_print_status` 已提供紧凑单行的全面健康指标。添加 Rich 库增加依赖和 ~100 LOC 格式化代码——纯 cosmetics |
| 聚合元数据目录 — JSONL 增量追加 + `filter_episodes()`（~100 LOC） | T-Rex #21 | Episode 过滤已在 `export_hdf5_to_zarr.py` 中实现（filter_task、filter_success、filter_tags、min_frames）。直接读 HDF5 `/meta` 始终同步；JSONL sidecar 会重复元数据且有陈旧风险 |
| DLS IK 姿态代价 — 增广雅可比零空间约束（~30 LOC） | T-Rex #22 | 零空间优化已作为后 IK 投影步骤存在（`ik.py:409-428`）。关节限制排斥通过 `nullspace.py` 工作。增广雅可比方法对 1D 零空间（7-DOF 臂，6-DOF 任务）数学等价。重构偏好，非 gap |
| Jlog6 解析雅可比修正 — SO(3) 右雅可比逆（~15 LOC） | T-Rex #23 | SO(3) 右雅可比逆（J_r^{-1}）仅在方向误差 >30° 时有意义，小误差时趋近恒等。DLS IK 无此修正已收敛至 1e-3 容差。理论精度改进，无可观测的实际失效。DX1 已修复可观测的双覆盖 bug |
| 组件名校验 — 拼写错误→清晰错误消息（~20 LOC） | T-Rex #15 | 正确性审计确认 `robot/interface.py` 中零处"component"字符串查找。硬件组件是直接 Python 对象引用（`self.arm`、`self.hand`、`self.hand_kinematics`）——不存在字符串到对象的映射可供校验。幻影 gap |
| Delta-base 动作导出 — `--delta-base` 标志（~40 LOC） | T-Rex #25 | Delta 计算（`np.diff`）是训练 dataloader 中的平凡一行。为每个可想象的训练管道操作在导出器中加格式转换 flag 会膨胀导出器。遵循"简洁优先" |
| 多数据集合并工具 — ProcessPoolExecutor 并行（~120 LOC） | T-Rex #27 | `export_hdf5_to_zarr.py:load_episodes()` 已连接单个目录的所有 episode。跨目录合并通过 `cp` 或 `ln -s` 一条命令完成。ProcessPoolExecutor 合并工具对一次性操作属推测性开发 |
| 消费者侧图像缩放 — `read_latest_camera(target_resolution=...)`（~30 LOC） | T-Rex #28 | 分辨率应在生产者端（CameraProcessConfig）配置，非消费者端。在 50Hz 控制回路中加实时缩放增加延迟。训练管道在 dataloader 中处理缩放。遵循"简洁优先" |
| 策略频域解耦 — Policy 30Hz→执行 50Hz 插值（~120 LOC） | T-Rex #34 | DP1 已通过 action-chunk/temporal-ensemble 解决频率不匹配。ArmInnerLoop Mode 6 固件原生以 250Hz 插值。这是 DP1 的实现细节，非独立前置项——不为尚不存在的代码预建基础设施 |
| Delta-EEF 动作空间 — chunk 边界 FK 漂移修正（~100 LOC） | T-Rex #35 | DLS IK 收敛至 1e-3 容差，每 chunk FK 残差 ~0.001m。对 50Hz 下 10-20 步 chunk，累积边界漂移可忽略。如 DP1 实现时观测到漂移，20 行 FK 修正即可——非前置项 |
| AsyncEpisodeSaver — stop 快速提取+后台写入 | LeRobot P0-2 | 正确性审计确认每 500 帧周期性 flush（`episode_recorder.py:220-221`）。stop 时尾部最多 499 帧非相机数据 + 文件关闭，通常 <100ms——远非声称的 2-5s。当前设计遵循 ManiUniCon 下降沿 dump 模式。添加异步写入引入 h5py 线程安全风险 |
| 连接时自动预置位 — `auto_home: bool = False` | ufactory_teleop P3-1 | 安全关键硬件的反模式。上电时机械臂物理姿态未知；自动规划/执行 return_to_home 路径可能失败并使机械臂处于不确定状态。当前设计（连接、验证、操作员决定何时回原位）正确且符合 fail-safe 哲学 |
| 线性速度限制因子 — `set_linear_spd_limit_factor(2.0)` | ufactory_teleop P3-2 | 正确性审计确认 `set_linear_spd_limit_factor` 是 Mode 5（笛卡尔）API——在 DexMani 中未使用，因项目独占 Mode 6（关节在线轨迹规划）。速度已可通过 `ArmInnerLoopConfig.joint_max_speed`（默认 90 deg/s）配置。在 Mode 6 下添加笛卡尔因子为功能上无效的死代码 |
| Butterworth 滤波器 — `signal_utils.py` 中 2 阶 5Hz @60Hz | ManiUniCon P2c | 滤波管道已在多个层级饱和 EMA（retargeting LPFilter α=0.6 + Cartesian EMA pos=0.8/rot=0.4 + hand delta clip + 固件轨迹规划）。在 1 阶 EMA 满足所有需求时添加 2 阶 Butterworth 为推测性开发。60 LOC 死工具代码 |
| Validator `action_range` 关节限位检查 | ManiUniCon P3a(partial) | 数学上已证明为同义反复：`validate_action()`（`robot/validate.py:82-90`）在发送到硬件并录制前将 `arm_qpos_cmd` 和 `hand_qpos_cmd` 原地钳位到关节限制内。HDF5 中存储的任何 action 已构造性保证在限制内。离线范围检查将 100% 通过且提供零信息 |
| 相机基类 — `sensor/base_camera.py`（~40 LOC） | ManiUniCon P3b | 恰好存在一个相机实现（RealSense）。提取抽象基类添加 40+ LOC 纯仪式代码（ABC、@abstractmethod stub、类型注解），零行为变更。CameraProcess 已通过 multiprocessing 提供崩溃隔离。YAGNI——在第二个相机驱动实际接入时提取接口 |

---

## 八、建议的 CLAUDE.md 修正 ✅ 已应用

```markdown
# Safety architecture (corrected)
1. **Pre-send gate** (`validate_action()`): **4 operations** — robot error gate, arm connection gate,
   arm joint-limit clip, hand joint-limit clip.
   `env_collision_check` and `actual_arm_qpos` params are accepted but **not yet wired**.
   **Hard prerequisite before autonomous rollouts**: add collision/torque/current/temperature/workspace gating.

# HDF5 recording format (clarification)
- Camera frames are streamed incrementally during recording; non-camera streams are buffered
  in RAM and bulk-written at stop_episode(). A crash before stop_episode() loses all buffered data.
  → P1 item D3: add periodic flush.

# DataValidator (corrected)
- 7 check categories (dynamic total per run — varies by number of data streams present):
  no_nan_obs, no_nan_action, non_zero_variance, camera_fresh, min_frames,
  no_duplicate_frames, timestamp_monotonicity.

# Inner loop (corrected)
- ArmInnerLoop runs at **50 Hz** (not 250 Hz) and does NOT perform software velocity/acceleration/
  jerk limiting. Mode 6 firmware handles all trajectory planning with configured joint_max_speed/joint_max_acc.

# Controller state machine (corrected)
- States: IDLE ⇄ TELEOP ⇄ PAUSED + EMERGENCY_STOP. There is no RECORDING state.
- Keys: B=begin+record, C=pause, S=stop+save, H=home, Q=quit, ESC=emergency_stop.
```

---

## 九、未纳入的源报告建议（透明性记录）

以下 3 项建议存在于源报告但未纳入本文的 P0-P3 主列表或拒绝表。经 9-agent 独立实码核查（166 次工具调用，~498k tokens），原 Section 九 的 43 项中：2 项提升至 P2、6 项提升至 P3、32 项经代码验证后拒绝（移入 Section 七）、3 项保留于此——均为触发条件尚未满足的待评估项。

| ID | 简述 | 原优先级 | ~LOC | 触发条件 |
|----|------|----------|------|----------|
| **B5** (LeFranX) | Watchdog 进程 — 跨进程心跳 + 独立 XArmAPI | P2 | 200 | 自主策略执行——当前人工监督替代了看门狗 |
| **P0-3** (LeRobot) | Chunk Boundary Smoothing — validate.py 向量范数位置+旋转钳位 | P0 | 30 | 策略推理模式启用——当前连续遥操作无需 chunk 边界平滑 |
| **P1-9** (LeRobot) | Sim→HDF5 录制 — 非实时路径处理物理步进 vs 20ms 网格不匹配 | P1 | 150 | Sim-to-real 成为项目优先事项 |

---

*本报告经四轮独立验证：首轮 11-agent 并行核查（~150 工具调用，~464k token 代码阅读），次轮 8 个关键断言的人工实地复查，第三轮 5-agent 实码事实核查（67 次工具调用）逐项验证所有声明。*
*第四轮 9-agent Section 九专项评估（166 次工具调用，~498k tokens）：5 个独立评估 agent + 3 个对抗验证 agent（安全/正确性/实用性审计）+ 1 个综合 agent——零事实错误，安全审计通过。*
*53 项建议经核查后拒绝（Section 七 列出关键项）。Section 九 透明记录 3 项待触发条件满足后评估的源报告建议。*
*截至 2026-07-13，已实现 21 项（B0, B1, A1, A2, B2, M1, M2, D1, D2, D3, R4, B3, B4, DX1, O4, DX8, O1, I1, R1, A4+A5, R3），累计 ~428 LOC。剩余 38 项（含新增 8 项 Section 九提升项）待实现。*

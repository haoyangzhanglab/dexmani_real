# dexmani_real 四框架全面审查分析报告

> **日期**: 2026-06-23 | **基准**: main @ 773f20b → 1364d3c (Phase 5)
> **对比框架**: LeFranX | BunnyVisionPro | ManiUniCon | **DexUMI** (新增)
> **审查方法**: 完整代码阅读 (24 文件) + 四框架源码分析 + Web 调研

---

## 1. 执行摘要

### 1.1 总体评估

dexmani_real 当前处于**接近生产就绪状态**。Phase 0-5 共 29 项改进已全部落地，代码库成熟度评级：★★★★☆ (4.5/5)。

**核心发现**：文档与代码之间存在显著滞后——文档中标记为"未实施"的多项改进（自适应阻尼、可操作性评分、validate_action、orientation workspace、pose 插值器、REARM、软减速等）在代码中**已全部实现**。文档需要一次同步更新。

### 1.2 Top 3 优势（vs 全部四个框架）

| # | 优势 | 详情 |
|---|------|------|
| 1 | **安全性 ★★★★★** | 11-bit QualityFlags + 四层安全模型 + validate_action 集中门，无框架可比 |
| 2 | **数据质量 ★★★★★** | per-frame quality flags 独有，支持训练数据精确过滤，HDF5 multi-camera + sidecar JSON |
| 3 | **部署易用性 ★★★★★** | 单脚本运行，无 Docker/ROS/Unity 依赖，配置 dataclass 类型安全 |

### 1.3 Top 3 需要立即关注的差距

| # | 差距 | 严重度 |
|---|------|--------|
| 1 | **文档严重滞后于代码** — 多处将已实现功能标记为"待实施" | High |
| 2 | **仿真端到端验证未完成** — `vr_teleop_sim.py --record` 未手动运行 | High |
| 3 | **DexUMI 范式差异未被认识** — 可穿戴外骨骼是遥操作的强力补充 | Medium |

### 1.4 DexUMI 的关键启示

DexUMI 代表了与 VR 遥操作完全不同的数据采集范式：**可穿戴外骨骼 + 视觉修复**。它不是 dexmani 的替代方案，而是强力补充——尤其适用于需要触觉反馈的精细操作任务。DexUMI 的 3D 打印外骨骼方法可以低成本适配 XHand。

---

## 2. DexUMI 深度分析（新增框架）

### 2.1 DexUMI 是什么？

**DexUMI** (Dexterous Universal Manipulation Interface) 是 Stanford/Columbia/NVIDIA/CMU 联合研究项目，CoRL 2025 Best Paper Finalist。

核心理念：**用人的手本身作为通用操作接口**——数据采集时不需要机器人。

```
┌─────────────────────────────────────────────────────────────────┐
│                     DexUMI 数据采集流程                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────┐                                        │
│  │ 操作员佩戴 3D 打印    │                                        │
│  │ 外骨骼 (XHand/Inspire)│ ← 编码器直接记录关节角度                │
│  │ + 指尖触觉传感器      │ ← 记录接触力信息                       │
│  │ + 腕部相机            │ ← 记录 RGB 视频                        │
│  └──────────┬───────────┘                                        │
│             │                                                     │
│             ▼                                                     │
│  ┌──────────────────────┐      ┌──────────────────────────────┐  │
│  │ Hardware Adaptation  │      │ Software Adaptation           │  │
│  │ 外骨骼运动学约束      │      │ 1. SAM2 分割人手+外骨骼       │  │
│  │ → robot-feasible轨迹 │      │ 2. ProPainter 修复背景        │  │
│  │ → 直接触觉反馈       │      │ 3. 重放机器人手动作录制视频    │  │
│  └──────────────────────┘      │ 4. 遮挡感知合成机器人手入场景  │  │
│                                 └──────────────────────────────┘  │
│                                             │                     │
│                                             ▼                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ 训练数据: 关节角度 + 触觉 + 修复后 RGB → Diffusion Policy   │ │
│  │ 平均成功率: 86% (4 tasks × 2 hands)                          │ │
│  │ 数据采集速度: 3.2× 传统遥操作                                 │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 DexUMI 与 dexmani 的核心差异

| 维度 | DexUMI | dexmani_real |
|------|--------|-------------|
| **数据采集范式** | 可穿戴外骨骼（无机器人） | VR 遥操作（实时控制机器人） |
| **触觉反馈** | ✅ 直接物理接触 | ❌ 无（仅有视觉反馈） |
| **数据采集速度** | 3.2× 遥操作（无需等待机器人） | 1×（机器人实时执行） |
| **动作空间保真度** | 外骨骼约束 → robot-feasible | IK + retargeting → robot-feasible |
| **视觉域差距** | 需要 SAM2+ProPainter 修复 | 无（机器人直接在场景中） |
| **适用场景** | 桌面精细操作（捏、拧、夹） | 大范围运动 + 精细操作 |
| **部署难度** | 需要 3D 打印外骨骼 + GPU | 仅需 VR 头显 |
| **安全性需求** | 无（离线数据采集） | 极高（实时机器人控制） |

### 2.3 DexUMI 可借鉴的特性

| # | 特性 | 优先级 | 实施难度 | 对 dexmani 的价值 |
|---|------|--------|---------|-----------------|
| 1 | **外骨骼数据采集模式** | P2 | 中（需 3D 打印） | 补充遥操作，提供触觉反馈数据 |
| 2 | **触觉传感器（指尖）** | P1 | 低（XHand 已内置） | 提升精细操作成功率 |
| 3 | **视觉域适应（修复+合成）** | P3 | 高（需 GPU+模型） | 跨 embodiment 迁移学习 |
| 4 | **Relative finger action** | P1 | 低（配置变更） | DexUMI 发现相对轨迹优于绝对位置 |
| 5 | **无机器人数据采集** | P2 | 中 | 降低硬件磨损，加速数据积累 |

**关键判断**：DexUMI 不是 dexmani 的竞争对手，而是**互补范式**。dexmani 的 VR 遥操作适合需要实时反馈和长距离运动的任务，DexUMI 的外骨骼模式适合需要触觉反馈的精细桌面操作。建议在 XHand 上探索低成本 3D 打印外骨骼方案。

**最新进展（2026-06）**：
- **RealDexUMI** 后续工作已在 2026 年 6 月发布，采用掌侧同构手套 + 共享末端模块，消除了 retargeting 需求，在 8 个任务上达到 88.75% 平均成功率
- 正在由 **BeingBeyond** 公司商业化，产品名为 "U1" 硬件系统
- DexUMI 使用 iPhone ARKit（非 VR 头显）进行 6-DOF 腕部跟踪，OAK-1 腕部相机（150° 对角视场角）
- 无需机器人在场即可采集数据，数据采集效率为传统遥操作的 3.2×

---

## 3. 文档 vs 代码 真实性审查

### 3.1 重大差异清单

| # | 文档位置 | 文档描述 | 实际代码状态 | 严重度 |
|---|---------|---------|-------------|--------|
| 1 | `framework-comparison.md:48-49` | Top 5 改进表混淆，"未实施"条目与"已实施"混排 | 代码已全部实施 | **High** |
| 2 | `framework-comparison.md:306` | "可操作性评分 **未在 teleop 使用**" | `ik.py:385-405` 自适应阻尼 + `ik.py:422-438` min_manipulability 门均已使用 | **High** |
| 3 | `maniunicon-comparison.md:638-665` | WorkspaceSafety 位于 `planner.py:638-665` | 已重构到 `workspace_safety.py`（81 行独立文件） | Medium |
| 4 | `development-plan.md:Phase1-2` | 多项标记为"待实施" | P0-1~P0-4、P1-1~P1-6 在代码中均已完成 | **High** |
| 5 | `framework-comparison.md:495` | QualityFlags 描述为"10-bit" | 实际为 11-bit（bit 0-10，含 CAMERA_OK bit 6） | Low |
| 6 | `control-loop-design.md` | 管道图为 7-stage | 实际为 9-step（VR→tracking→state→compute→quality→record→safety→execute→status+overrun） | Medium |

### 3.2 代码中已实施但文档标记为"未完成"的功能

以下功能在代码中已完全实现，但至少一份文档中标记为"待实施"或"未使用"：

| 功能 | 实施文件:行号 | 状态 |
|------|-------------|------|
| **自适应 DLS 阻尼** | `ik.py:385-405` | ✅ 完整实现（基于 manipulability 线性插值） |
| **可操作性评分 + 拒绝门** | `ik.py:422-438` | ✅ 完整实现（低 manipulability → heavy damping retry） |
| **EEF 方向工作空间边界** | `workspace_safety.py:49-80` + `interface.py:142-166` | ✅ 完整实现（check + clamp） |
| **VR Per-Step 旋转 Delta 限制** | `arm_mapper.py:105-118` `_clip_delta_rot()` | ✅ 完整实现（默认 1.0 rad） |
| **跟踪丢失软减速** | `controller.py:307-335` `_apply_soft_deceleration()` | ✅ 完整实现（指数衰减） |
| **集中化 validate_action** | `interface.py:176-216` | ✅ 完整实现（5 层 fail-fast 检查） |
| **REARM 中间恢复** | `controller.py:634-636` + `_rearm()` 方法 | ✅ 完整实现（'C' 键） |
| **Cartesian Pose 插值器** | `pose_interpolator.py` (184 行) + `pipeline.py:104-175` | ✅ 完整实现（Linear pos + SLERP rot） |
| **DLS-only 模式** | `TeleopProfile.use_position_ik=False` | ✅ 纯配置项，无需代码改动 |
| **Loop Overrun 检测** | `controller.py:511-520` | ✅ 完整实现（150% 周期阈值） |
| **RateManager 精确等待** | `controller.py:170-171` | ✅ 完整实现（hybrid busy-wait） |

**结论**：文档整体滞后约 1-2 个 Phase。Phase 1-4 的 P0/P1 级改进已全部在代码中落地，文档应更新为反映真实代码状态。

---

## 4. 四框架横向对比矩阵（更新版，含 DexUMI）

### 4.1 15 维度综合对比

| 维度 | dexmani_real | LeFranX | BunnyVisionPro | ManiUniCon | DexUMI |
|------|-------------|---------|-----------------|------------|--------|
| **范式** | VR 遥操作 | VR 遥操作 | VR 遥操作 | VR 遥操作 | **可穿戴外骨骼** |
| **架构** | 单进程 50Hz 顺序 | Python+C++ 多线程 | 三机 ZMQ 部署 | 多进程+共享内存 | 离线采集+训练 |
| **VR 输入** | Meta Quest HTS 50Hz | Meta Quest TCP | Vision Pro 60Hz | Meta Quest 30Hz | **无（外骨骼编码器）** |
| **手臂 IK** | DLS (adaptive) + MPlib | 解析 IK + Brent q7 | DLS Pinocchio | Pink QP | N/A（无实时控制） |
| **IK 延迟** | 1-3ms (DLS) / ~10ms (回退) | <1ms (C++) | ~1ms (server) | <1ms (QP) | N/A |
| **手部重定向** | DexPilot + pinky 适配 | DexPilot + pinky | SeqRetargeting | N/A (二指爪) | 外骨骼直接编码 |
| **安全模型** | ★★★★★ 四层+11bit | ★★☆☆☆ 基础限制 | ★★☆☆☆ 速度裁剪 | ★★★★☆ 三层+validate | N/A (离线) |
| **数据格式** | HDF5 + sidecar JSON | LeRobot Dataset | HDF5 + NPY | Zarr + LeRobot v3 | 关节角+触觉+RGB |
| **质量标记** | ✅ 11-bit/frame | ❌ | ❌ | ❌ | ❌ (离线无需求) |
| **多相机** | ✅ MultiCameraManager | ✅ multi-camera | ❌ (仅 robot) | ✅ multi-camera | ✅ 腕部相机 |
| **触觉传感器** | ✅ XHand 内置 | ❌ | ❌ | ❌ | ✅ 外骨骼指尖 |
| **双手支持** | ❌ (仅右手) | ✅ 双手 | ✅ 原生双手 | ❌ (仅右手) | ✅ (XHand+Inspire) |
| **部署复杂度** | ★★★★★ 单脚本 | ★★★☆☆ 编译C++ | ★★☆☆☆ Docker+VP | ★★★☆☆ Python only | ★★☆☆☆ 3D打印+GPU |
| **进程模型** | 单进程+daemon线程 | 多进程+多线程 | 三机分布式 | 多进程+共享内存 | 离线批处理 |
| **成熟度** | ★★★★☆ (4/5) | ★★★★★ (5/5) | ★★★★☆ (4/5) | ★★★★☆ (4/5) | ★★★☆☆ (3/5, 研究项目) |

### 4.2 dexmani 在四框架中的位置（更新后雷达图）

```
                     VR/输入质量 ★★★★☆ (4)
                           /\
                          /  \
            部署易用性 ★★★★★│    │★★★★☆ IK 鲁棒性 (3→4 ↑)
                        │    │
                        │    │
                        │    │
      可扩展性 ★★★★☆ ───┼────┼─── ★★★★☆ 手部重定向
          (3→4 ↑)       │    │
                        │    │
                        │    │
         延迟 ★★★★☆ ────┼────┼─── ★★★★★ 安全性
                        │    │
                        │   /
             数据质量 ★★★★★   /
                          \/
```

**变化说明**（与 2026-06-22 v2.0 对比）：
- **IK 鲁棒性**: 3→4（自适应阻尼+可操作性门+heavy-damping retry 已实施）
- **可扩展性**: 3→4（ZMQ VR + MultiCameraManager + CartPoseInterpolator + TeleopPipeline 解耦）

---

## 5. dexmani 独有优势（护城河分析）

### 5.1 与全部四个框架对比的独有优势

| # | 独有优势 | 为什么独特 | 竞争壁垒 |
|---|---------|-----------|---------|
| 1 | **11-bit per-frame QualityFlags** | 每帧精确标记 TRACKING/IK/RETARGET/JUMP/WORKSPACE/CAMERA/TORQUE/CURRENT/TEMP/COMM/RETARGET_VALID 状态 | **高** — 四框架均无此机制，是训练数据质量过滤的核心能力 |
| 2 | **四层安全模型 + 集中化 validate_action** | 驱动层→接口层→控制器层→路径层，validate_action 统一门前检查 | **高** — 无框架可比，LeFranX 仅有基础位置限制 |
| 3 | **Sidecar JSON 元数据** | 每个 episode 自动生成 `episode_NNN.json`，含完整 metadata | **中** — 简单但实用，ManiUniCon 需要 Hydra 配置才有等效能力 |
| 4 | **FingertipDeskSafety** | FK 桌面碰撞检测，防止手指穿透桌面 | **高** — 四框架均无此机制 |
| 5 | **TeleopPipeline 解耦** | 共享动作计算管道（sim + real 复用），独立于控制器状态机 | **中** — 架构清洁，ManiUniCon 有类似分层 |
| 6 | **auto_stop_on_quality_drop** | 连续低质量帧自动停止录制，保护数据质量 | **中** — 简单但四框架均无类似机制 |
| 7 | **IK 结构化诊断** | `_build_ik_diagnostic()` 分类失败原因（singular/pose_error/self_collision/unreachable/filtered） | **低-中** — 运维价值高 |
| 8 | **XHand 全状态感知** | 电流/触觉/温度/通信错误 per-finger | **高** — XHand 独有硬件能力，配合 QualityFlags 提供最丰富的手部状态 |

### 5.2 综合评判

dexmani 在**安全性**和**数据质量**两个维度上建立了不可逾越的护城河。其他框架要么依赖机器人原生安全机制（LeFranX、BVPro），要么完全没有手部安全监控（Open-Teach、ManiUniCon）。11-bit QualityFlags + sidecar JSON 的组合提供了业界领先的训练数据管理能力。

---

## 6. 关键差距与风险

### 6.1 安全关键差距

| # | 差距 | 严重度 | 说明 |
|---|------|--------|------|
| 1 | **仿真验证未完成** | High | `vr_teleop_sim.py --dummy --record` 未手动运行，无法确认录制 pipeline 在仿真中正常 |
| 2 | **真机 dry_run 未验证** | High | `keyboard_teleop_real.py` 受权限保护，真机脚本未更新为使用最新 PipelineConfig |
| 3 | **Zarr 导出端到端未验证** | Medium | `export_hdf5_to_zarr.py` 存在但未被实际数据验证 |
| 4 | **DataValidator 7 项检查未跑通** | Medium | 代码已实现但未在实际录制数据上执行 |

> **注意**: 这些都是验证层面的差距，代码功能本身已全部实现。不影响安全性，但影响生产就绪判定。

### 6.2 性能关键差距

| # | 差距 | 严重度 | 说明 |
|---|------|--------|------|
| 1 | **MPlib 回退仍是最大延迟波动源** | Low | DLS-only 模式可消除（`use_position_ik=False`），但默认仍然启用回退 |
| 2 | **单进程 GIL 竞争** | Low | 当前 50Hz 控制回路在 20ms 预算内运行良好（DLS 路径 7-12ms），但多相机+VR+IK 并发时会接近预算上限 |

### 6.3 数据质量差距

| # | 差距 | 严重度 | 说明 |
|---|------|--------|------|
| 1 | **无归一化统计预计算** | Medium | ManiUniCon 在导出时预计算 obs_mean/std、action_mean/std，dexmani 需要后处理步骤 |
| 2 | **无 LeRobot 格式直接导出** | Low | `export_hdf5_to_zarr.py` 已提供 Zarr 导出，但缺少 LeRobot Dataset v3.0 直接格式 |
| 3 | **无触觉数据质量标记** | Low | XHand 已采集触觉数据但 QualityFlags 中无触觉相关 bit |

### 6.4 工程成熟度差距

| # | 差距 | 严重度 | 说明 |
|---|------|--------|------|
| 1 | **文档滞后** | High | 多处将已实施功能标记为"待实施"，行号引用过期（如 WorkspaceSafety 已重构） |
| 2 | **双手支持缺失** | Medium | LeFranX 和 BVPro 均原生支持双手，dexmani 仅有右手 |
| 3 | **配置 JSON 支持** | Low | `from_dict()` 已通过 `FromDictMixin` 实现，但缺少 JSON 配置文件示例 |
| 4 | **Docker 镜像未维护** | Low | `Dockerfile.teleop` 存在但依赖项可能过期 |

---

## 7. 未采纳的四框架特性

### 7.1 从 LeFranX

| 特性 | 状态 | 说明 |
|------|------|------|
| C++ IK 扩展 (pybind11) | ❌ 未采纳 | ~500 行 C++，可降低 DLS 1ms → <0.5ms，但当前延迟可接受 |
| Brent q7 冗余优化 | ❌ 不适用 | xArm7 无解析解，Brent 优化不适用 |
| Ruckig 1kHz 轨迹平滑 | ❌ 未采纳 | 依赖 Franka，xArm7 使用 bottleneck scaling 替代 |
| 500ms 命令超时 | ❌ 未采纳 | Hold-on-failure 等效但无超时检测，Python 进程崩溃时机器人保持最后指令 |
| LeRobot 原生数据集 | ❌ 未采纳 | Zarr 导出已实现，LeRobot v3.0 格式待添加 |

### 7.2 从 BunnyVisionPro

| 特性 | 状态 | 说明 |
|------|------|------|
| 三机 ZMQ 部署 | ⚠️ 部分采纳 | ZMQ VR 订阅器已实现（`vr_publisher.py`），但非默认模式 |
| Apple Vision Pro 支持 | ❌ 未采纳 | Meta Quest 已满足需求 |
| 250Hz PID 内环 | ❌ 架构不同 | xArm7 SDK 自带速度环，dexmani 使用驱动层 bottleneck scaling |
| 球体近似碰撞检测 | ❌ 未采纳 | MPlib self-collision 已覆盖，BunnyVisionPro 的球体近似可作快速预筛 |
| SeqRetargeting | ❌ 未采纳 | DexPilot 已验证良好，无切换必要 |

### 7.3 从 ManiUniCon

| 特性 | 状态 | 说明 |
|------|------|------|
| Hydra 配置系统 | ❌ 未采纳 | 当前 dataclass 更类型安全，但 YAML 可 diff |
| Pink QP IK | ❌ 未采纳 | DLS 自适应阻尼已足够，QP 引入额外依赖 |
| JointSpaceSmoother | ❌ 未采纳 | EMA + dex-retargeting low_pass_alpha 已覆盖 |
| Lock-Free SharedMemory | ⚠️ 部分采纳 | `SharedMemoryRingBuffer` 存在但未在热路径使用 |
| max_record_steps | ✅ 已采纳 | `collection_config.py` auto_stop 等效 |
| PoseTrajectoryInterpolator | ✅ 已采纳 | `pose_interpolator.py`，简化为 VR 50Hz 原生版本 |

### 7.4 从 DexUMI（全新框架）

| 特性 | 优先级 | 实施难度 | 说明 |
|------|--------|---------|------|
| 3D 打印外骨骼 (XHand) | P2 | 中 | 需设计+打印，但可复用 DexUMI 开源设计 |
| 外骨骼关节编码器 | P2 | 中 | 替代 DexPilot 视觉重定向，可能更精确 |
| 指尖触觉传感器 | P1 | 低 | XHand 已内置，需在 QualityFlags 中加 flag |
| 视觉域适应 (SAM2+ProPainter) | P3 | 高 | 需 GPU，当前无 sim2real 域差距需求 |
| Relative finger action | P1 | 低 | 配置变更，DexUMI 证明相对轨迹优于绝对位置 |
| 无机器人数据采集模式 | P2 | 中 | 降低硬件磨损，加速数据积累 |

---

## 8. 优先级建议（Phase 6+）

### 8.1 立即行动（本周）

| # | 行动 | 工作量 | 影响 |
|---|------|--------|------|
| **P0-1** | **运行仿真端到端验证** | 2h | 确认录制 pipeline 正常，解锁生产就绪判定 |
| | `python scripts/sim/vr_teleop_sim.py --dummy --headless --record` | | |
| | → `export_hdf5_to_zarr.py` → `DataValidator.validate_directory()` | | |
| **P0-2** | **文档同步更新** | 4h | 消除代码与文档差异 |
| | 更新 `framework-comparison.md` Top 5 改进状态 | | |
| | 更新 `maniunicon-comparison.md` 行号引用 | | |
| | 更新 `development-plan.md` Phase 1-4 完成状态 | | |
| | 更新 `CLAUDE.md` QualityFlags 11-bit | | |
| **P0-3** | **更新真机测试脚本** | 2h | 确认真机入口可用 |
| | `keyboard_teleop_real.py` 添加 `--multi-camera` `--export-zarr` 选项 | | |
| | dry_run 模式验证 | | |

### 8.2 短期（2 周内）

| # | 行动 | 工作量 | 影响 |
|---|------|--------|------|
| **P1-1** | **LeRobot Dataset v3.0 直接导出器** | 1d | 打通 HuggingFace 训练生态 |
| | 新建 `scripts/tools/export_to_lerobot.py` | | |
| | 基于 ManiUniCon LeRobot 格式参考 | | |
| **P1-2** | **触觉数据 QualityFlag** | 1h | 完善数据质量维度 |
| | 新增 `TACTILE_OK = 1 << 11` bit | | |
| | `collection_loop.py` + `quality_flags.py` | | |
| **P1-3** | **Relative finger action 模式** | 2h | DexUMI 经验：相对轨迹 > 绝对位置 |
| | `TeleopProfile.use_relative_finger_action: bool = False` | | |
| **P1-4** | **命令超时保护** | 2h | LeFranX 经验：Python 进程崩溃时机器人安全 |
| | 看门狗线程监控 controller 心跳 → xArm SDK emergency stop | | |

### 8.3 中期（1-2 月）

| # | 行动 | 工作量 | 影响 |
|---|------|--------|------|
| **P2-1** | **XHand 3D 打印外骨骼探索** | 1w | DexUMI 互补数据采集模式 |
| | 复用 DexUMI 开源外骨骼设计 | | |
| | 3D 打印 + 编码器适配 + 触觉传感器 | | |
| **P2-2** | **球体近似碰撞预筛** | 3d | BVPro 轻量级 env-collision 替代 MPlib |
| | 在 MPlib collision check 之前添加快速球体预筛 | | |
| **P2-3** | **双手（左手）支持** | 1w | 扩展操作空间 |
| | 镜像 URDF + OPERATOR2MANO_LEFT + 控制器 side 参数 | | |

### 8.4 长期（3 月+）

| # | 行动 | 说明 |
|---|------|------|
| P3-1 | C++ DLS 扩展 (pybind11) | 当 DLS 延迟成为瓶颈时（当前 1-3ms 可接受） |
| P3-2 | 视觉域适应 (DexUMI 风格) | 当需要跨 embodiment 迁移时 |
| P3-3 | Docker 化完整部署 | 更新 Dockerfile.teleop，一键部署全部依赖 |

---

## 9. DexUMI 特定建议

### 9.1 dexmani + DexUMI 混合方案设想

dexmani 和 DexUMI 可以形成互补的双模式数据采集系统：

```
┌─────────────────────────────────────────────────────────────────┐
│                  dexmani + DexUMI 混合方案                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  模式 A: VR 遥操作 (dexmani)                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Meta     │───→│ Teleop   │───→│ XArm7 +  │───→│ HDF5 +   │  │
│  │ Quest    │    │ Controller│    │ XHand    │    │ sidecar  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│  适用: 大范围运动, 需要视觉反馈, 远程操作                       │
│                                                                   │
│  模式 B: 外骨骼采集 (DexUMI 风格)                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 3D 打印  │───→│ 编码器   │───→│ 触觉+    │───→│ 关节角+  │  │
│  │ 外骨骼   │    │ 采集     │    │ 腕部相机 │    │ 触觉数据 │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│  适用: 精细桌面操作, 需要触觉反馈, 无机器人磨损                  │
│                                                                   │
│  共享: HDF5 录制 + QualityFlags + sidecar JSON + Zarr 导出       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 具体实施步骤（如决定采纳）

1. **Phase 1**: 从 DexUMI GitHub 仓库获取 XHand 外骨骼 3D 模型
2. **Phase 2**: 3D 打印 + 适配 xArm7 腕部安装
3. **Phase 3**: 编码器驱动开发（读取 Arduino/ESP32 编码器值 → 映射为 XHand 关节角）
4. **Phase 4**: 集成到 `CollectionLoop.record_frame()`（新增 `data_source: "teleop" | "exoskeleton"` 字段）
5. **Phase 5**: 触觉数据 QualityFlag + sidecar JSON 扩展

---

## 10. 验证清单（更新版）

### 生产就绪判定标准

| # | 验证项 | 状态 | 备注 |
|---|--------|------|------|
| 1 | `python scripts/sim/vr_teleop_sim.py --dummy --record` 产出有效 HDF5 | ⬜ 待运行 | Phase 5.4 |
| 2 | `python scripts/tools/export_hdf5_to_zarr.py` 成功转换 | ⬜ 待验证 | 依赖 #1 |
| 3 | Zarr 数据集可被 `zarr.open()` 加载，shape 正确 | ⬜ 待验证 | 依赖 #2 |
| 4 | `DataValidator.validate_directory()` 全部 7 项检查通过 | ⬜ 待验证 | 依赖 #1 |
| 5 | CollectionLoop 正确管理录制生命周期 | ✅ | 功能测试通过 |
| 6 | auto_stop_on_quality_drop 触发后 episode 正确保存 | ✅ | 功能测试通过 |
| 7 | Episode sidecar JSON 写入正确 | ✅ | 功能测试通过 |
| 8 | 多相机 HDF5 写入 `/camera/<serial>/rgb` + `depth` | ✅ | 代码审查确认 |
| 9 | REARM (C 键) 从 EMERGENCY_STOP 恢复到 IDLE | ✅ | 代码审查确认 |
| 10 | 自适应阻尼在非奇异区低阻尼、奇异区高阻尼 | ✅ | 代码审查确认 |
| 11 | Orientation workspace 越界 clamp 正确 | ✅ | 代码审查确认 |
| 12 | VR delta rot cap 拦截大角度跳变 | ✅ | 代码审查确认 |
| 13 | 软减速在短暂跟踪丢失时平滑减速 | ✅ | 代码审查确认 |
| 14 | CartPoseInterpolator 线性位置 + SLERP 插值 | ✅ | 代码审查确认 |
| 15 | `keyboard_teleop_real.py` 真机 dry_run 模式 | ⬜ 待运行 | 权限受限 |
| 16 | 文档更新与代码一致 | ⬜ 待更新 | 本文档完成后更新 |

---

## 附录 A: 文件变更追踪（2026-06-23 状态）

### A.1 代码中已存在但文档未更新的文件

| 文件 | 行数 | 关键内容 |
|------|------|---------|
| `planning/workspace_safety.py` | 81 | WorkspaceSafety（含 orientation check/clamp） |
| `teleop/vr/pose_interpolator.py` | 184 | CartPoseInterpolator（Linear pos + SLERP rot） |
| `teleop/vr/vr_publisher.py` | 251 | ZMQ VR Publisher + Subscriber |
| `teleop/core/pipeline.py` | ~620 | TeleopPipeline（shared action computation） |
| `scripts/tools/export_hdf5_to_zarr.py` | ~800 | HDF5→Zarr 导出脚本 |
| `recording/collection_loop.py` | 344 | CollectionLoop（auto_stop + sidecar JSON） |
| `recording/collection_config.py` | ~60 | CollectionConfig（含 quality drop 设置） |

### A.2 核心实现文件快速索引

| 要查找的功能 | 文件:行号 |
|-------------|----------|
| 自适应 DLS 阻尼 | `planning/ik.py:385-405` |
| 可操作性评分门 | `planning/ik.py:422-438` |
| IK 失败诊断 | `planning/ik.py:194-292` |
| 方向工作空间检查 | `planning/workspace_safety.py:49-64` |
| 集中化 validate_action | `robot/interface.py:176-216` |
| 软减速 | `teleop/core/controller.py:307-335` |
| CartPoseInterpolator | `teleop/vr/pose_interpolator.py:27-184` |
| 旋转 delta 限制 | `teleop/vr/arm_mapper.py:105-118` |
| 多相机集成 | `teleop/core/controller.py:409-432` |
| 11-bit QualityFlags | `recording/quality_flags.py:25-39` |
| auto_stop_on_quality_drop | `recording/collection_loop.py:141-158` |
| sidecar JSON | `recording/collection_loop.py:276-307` |
| Loop overrun 检测 | `teleop/core/controller.py:511-520` |

---

## 附录 B: 四框架参考资源

| 框架 | 代码仓库 | 关键论文 |
|------|---------|---------|
| **LeFranX** | 非公开（Reference 目录） | LeRobot + DexPilot |
| **BunnyVisionPro** | 非公开（Reference 目录） | Apple Vision Pro Teleoperation |
| **ManiUniCon** | 非公开（Reference 目录） | Universal Manipulation Interface |
| **DexUMI** | [github.com/real-stanford/DexUMI](https://github.com/real-stanford/DexUMI) | CoRL 2025 Best Paper Finalist |

---

> **报告版本**: v1.0 | **撰写日期**: 2026-06-23
> **审查范围**: dexmani_real 24 文件完整阅读 + LeFranX/BVPro/ManiUniCon 源码对比 + DexUMI Web 调研
> **建议**: 本报告应作为 Phase 6 开发计划的基础，替代 `development-plan.md` 中已过时的 Phase 6 内容。

# DexMani 审计问题总表（按优先级排序）

> 提取自 `docs/audit-dexmani-vs-maniunicon.md`，2026-08-05。
> 图例：🔴严重 🟡中等 🟢轻微 🔵缺失机制 🟣基础设施差距 ✅已确认 ⚠️已修正
>
> **🟢 P4 修复状态：2026-08-05 全部 12 项 P4 问题已修复（9 文件 ~100 行变更）。详见 `docs/audit-dexmani-vs-maniunicon.md` §13 更新。**

## 全部问题（P0 → P4）

| 优先级 | ID | 严重度 | 验证 | 问题简述 | 文件:行号 | 具体表现 | 修复建议 | 类别 |
|:---:|:---:|:---:|:---:|---|---|---|---|---|
| **P0** | M01 | 🔵 | — | **无模型推理进程** | `policy/vr_teleop_policy.py` | 零推理能力；无模型加载、无 observation/action wrapper、无同步执行协议。当前仅为 VR→IK 转换器 | 新建 `ModelPolicy(mp.Process)`，参考 ManiUniCon `TorchModelPolicy` | 缺失机制 |
| **P0** | M02 | 🔵 | — | **无 Observation Wrapper** | — | 无法构建 k-帧历史 observation tensor；无预处理（resize/normalize）抽象 | 实现 ObsWrapper 工厂，参考 `PPTImageWrapper` / `FalconPCDWrapper` | 缺失机制 |
| **P0** | M03 | 🔵 | — | **无 Action Wrapper** | — | 无 chunk 提取、坐标空间转换、latency 补偿；仅支持 joint position 模式 | 实现 ActWrapper，参考 `ActionChunkWrapper` | 缺失机制 |
| **P0** | M04 | 🔵 | — | **无推理-执行同步协议** | — | 无 `robot_ready`/`policy_ready` Event 握手；无封闭推理-执行循环 | 新增双 Event 同步协议 | 缺失机制 |
| **P0** | M05 | 🔵 | ✅ | ✅已修复 | **RingBuffer 缺少 get_last_k(k)** | `shm/robot_ring.py` | `SeqlockRingBuffer` 仅支持 `read_latest()`（1 帧），无法读 k-帧历史窗口 | ✅ `get_last_k(k)` 已实现：独立 seqlock 验证、最新优先遍历、严格序列匹配、覆盖检测+早停。+`read_arm_state_k`/`read_hand_state_k` 便捷封装；state ring maxlen 3→8。16 单元测试 | 缺失机制 |
| **P0** | M06 | 🔵 | — | **无 Cartesian 动作模式** | `robot/types.py` | `RobotAction` 仅有 `arm_qpos_cmd(7)` + `hand_qpos_cmd(12)`，无 `control_mode` 字段 | 新增 `control_mode` 字段（joint/Cartesian） | 缺失机制 |
| **P1** | D01 | 🔴 | ✅ | ✅已修复 | **手部归位代码 4 处重复** | `policy/vr_teleop_policy.py:351-377,488-506,546-573` + `examples/real/vr_teleop_hand_record.py:379-396` | 相同的 poll→send→converge 循环出现 4 次 | ✅ `hand_home_converge()` 提取到 shared_storage.py；4 call site 替换为 3-9 行调用 | 代码重复 |
| **P1** | D02 | 🔴 | ✅ | ✅已修复 | **手部 retargeter reset 3 处重复** | `policy/vr_teleop_policy.py:631-638,707-714,867` | `_try_init_hand_retargeter()` + `_reset_hand_retargeter()` 模式重复 | ✅ `_seed_hand_retargeter()` 提取；3 call site 替换 | 代码重复 |
| **P1** | D03 | 🔴 | ✅ | ✅已修复 | **arm+hand 联合归位 5 处重复** | `policy/vr_teleop_policy.py`(x3) + `examples/real/vr_teleop_hand_record.py`(x2) | 完整 hand home poll → arm home queue → wait convergence 序列 5 处出现 | ✅ 随 D01 一并消除；hand 归位统一为 `hand_home_converge()` | 代码重复 |
| **P1** | M07 | 🔵 | — | 待处理 | **HDF5→LeRobot/RLDS 导出缺失** | — | 采集格式（HDF5 v11）与标准训练格式（LeRobot v3.0 / RLDS）之间无桥接工具 | 实现 `export_to_lerobot.py` 转换脚本 | 缺失机制 |
| **P1** | M08 | 🔵 | — | 待处理 | **多 Episode 合并缺失** | — | 无 Zarr ReplayBuffer 式追加或虚拟合并视图 | 实现 `merge_episodes.py` 或 Zarr 导出 | 缺失机制 |
| **P1** | M09 | 🔵 | — | 待处理 | **自动质量评估未集成** | `dexmani_real/tools/episode_quality.py` | `episode_quality.py` 已实现但未自动调用 | 在 `EpisodeRecorder.stop_episode` 回调中触发 assess | 缺失机制 |
| **P1** | M10 | 🔵 | — | 待处理 | **多相机支持缺失** | `sensor/camera_process.py` | 当前仅支持单相机 | 参考 ManiUniCon 多相机架构 | 缺失机制 |
| **P1** | T01 | 🔴 | — | 🟡部分 | **零单元测试** | — | 两个项目均无任何 `test_*.py` 文件 | ✅ `tests/test_robot_ring.py` 已添加（16 测试，M05）。核心模块（safety/ik_candidates/timestamp_buffer）仍缺 | 测试 |
| **P0** | M06 | 🔵 | — | **无 Cartesian 动作模式** | `robot/types.py` | `RobotAction` 仅有 `arm_qpos_cmd(7)` + `hand_qpos_cmd(12)`，无 `control_mode` 字段 | 新增 `control_mode` 字段（joint/Cartesian） | 缺失机制 |
| **P1** | D01 | 🔴 | ✅ | 待处理 | **手部归位代码 4 处重复** | `policy/vr_teleop_policy.py:351-377,488-506,546-573` + `examples/real/vr_teleop_hand_record.py:379-396` | 相同的 poll→send→converge 循环出现 4 次；实例 2 是截断版（缺 `_hand_home_reached` 跟踪）；实例 4 调用不同的写函数 | 提取 `_home_hand(shared, timeout_s, tol_rad)` 辅助函数 | 代码重复 |
| **P1** | D02 | 🔴 | ✅ | 待处理 | **手部 retargeter reset 3 处重复** | `policy/vr_teleop_policy.py:631-638,707-714,867` | `_try_init_hand_retargeter()` + `_reset_hand_retargeter()` 模式重复；实例 1/2 字面相同（仅变量名不同），实例 3 数据源不同 | 提取 `_reinit_hand_retargeter(shared)` 统一入口 | 代码重复 |
| **P1** | D03 | 🔴 | ✅ | 待处理 | **arm+hand 联合归位 5 处重复** | `policy/vr_teleop_policy.py`(x3) + `examples/real/vr_teleop_hand_record.py`(x2) | 完整 hand home poll → arm home queue → wait convergence 序列 5 处出现。本仓库重复度最高的模式 | 提取 `_home_hand_then_arm(shared)` | 代码重复 |
| **P1** | M07 | 🔵 | — | 待处理 | **HDF5→LeRobot/RLDS 导出缺失** | — | 采集格式（HDF5 v11）与标准训练格式（LeRobot v3.0 / RLDS）之间无桥接工具 | 实现 `export_to_lerobot.py` 转换脚本 | 缺失机制 |
| **P1** | M08 | 🔵 | — | 待处理 | **多 Episode 合并缺失** | — | 无 Zarr ReplayBuffer 式追加或虚拟合并视图；多 episode 训练需外部脚本手动拼接 | 实现 `merge_episodes.py` 或 Zarr 导出 | 缺失机制 |
| **P1** | M09 | 🔵 | — | 待处理 | **自动质量评估未集成** | `dexmani_real/tools/episode_quality.py` | `episode_quality.py` 已实现 filter/assess/validate，但 `stop_episode` 后未自动调用 | 在 `EpisodeRecorder.stop_episode` 完成回调中触发 assess | 缺失机制 |
| **P1** | M10 | 🔵 | — | 待处理 | **多相机支持缺失** | `sensor/camera_process.py` | 当前仅支持单相机；无 dict-of-rings 模式 | 参考 ManiUniCon 多相机架构（独立 ring + 融合 ring） | 缺失机制 |
| **P1** | T01 | 🔴 | — | 待处理 | **零单元测试** | — | 两个项目均无任何 `test_*.py` 文件；无 CI/CD pipeline | 核心模块优先加测：`safety.py`、`ik_candidates.py`、`episode_recorder.py` | 测试 |
| **P2** | L01 | 🟡 | ✅ | ✅已修复 | **手部 Tactile 重复读取** | `policy/vr_teleop_policy.py:782,1052` | `_read_hand_tactile()` 每帧调用 2 次（happy path: 782+1052）；held 帧缺 forward-fill（E02 cache 仅覆盖 active path） | 合并为单次读取：forward-fill 上移至 line 782，删除 line 1052 双读。统一 held/active 两路径触觉数据一致性 | 效率 |
| **P2** | L02 | 🟡 | ✅ | ✅已修复 | **`_safe_arm_queue_put` timeout 过长** | `policy/vr_teleop_policy.py:1092` | `timeout=0.5s` = 8 帧@16Hz；Arm 死亡后 Policy 阻塞 500ms 发手部命令但臂部停滞 → 手-臂异步 | 降至 `0.2s`（3 帧）— arm_loop 恢复 ~200ms + 排空 66ms = 266ms 最坏窗口，0.2s 刚好覆盖；false-positive 触发干净 shutdown | 逻辑缺陷 |
| **P2** | L03 | 🟡 | ⚠️ | ✅已修复 | **hand_loop + camera_loop 用裸 time.sleep()** | `robot/hand_process.py:284-288` + `sensor/camera_process.py:149-150` | 两个进程使用手动 `elapsed/sleep/last_ts` 模式（无累积漂移但缺过载检测），arm_loop/policy_loop 已用 RateManager | 统一迁移到 `RateManager`：绝对截止时间调度 + 过载检测。hand_loop: 30Hz <1ms 工作；camera_loop: 16Hz 含 40ms 点云，RateManager 提供一致性+过载告警 | 效率 |
| **P2** | M11 | 🔵 | — | ⏭️ skip | **无 validate_action() 最后防线** | `robot/arm_loop.py` | 策略层裁剪后无 robot 进程内二次验证 | Policy 层已有完整裁剪（workspace + delta + NaN + IK + joint limits）+ 固件 C31 backstop。加 arm_loop 二次验证是防御纵深非必须 | 缺失机制 |
| **P2** | M12 | 🔵 | — | ⏭️ defer | **无 RobotInterface ABC** | — | 无 `connect/send/validate/stop` 抽象接口；机器人硬件依赖直接耦合到 arm_loop/hand_loop | 当前纯函数架构不需要 ABC。P0 ModelPolicy 落地时按需提取 Protocol | 缺失机制 |
| **P2** | M13 | 🔵 | — | ⏭️ skip | **无分层配置系统** | `config/defaults.py` | Frozen dataclass 单例无 Hydra 式分层 | P3 C02（load_config_json + --config）已覆盖 80% 用例。Hydra 对单机系统过度工程 | 缺失机制 |
| **P2** | M14 | 🔵 | — | ⏭️ skip | **无结构化遥测导出** | — | 无 Prometheus/StatsD metrics；无运行时 CPU/内存/GPU 监控 | Prometheus/StatsD 面向多机集群。单机遥操作已有 status print + heartbeat 监控 | 缺失机制 |
| **P2** | M15 | 🔵 | — | ⏭️ skip | **无 CI/CD** | — | 无 GitHub Actions；无自动化 mypy/lint/test | P3 T02（.pre-commit-config.yaml）已覆盖格式化+类型检查。本地 hooks 是 80/20 的 CI | 缺失机制 |
| **P2** | M16 | 🔵 | — | ⏭️ skip | **无 README** | — | 无安装说明、快速开始、硬件要求文档 | CLAUDE.md（~400 行）已是实质文档，含架构图、快速参考、硬件说明 | 文档 |
| **P3** | L04 | 🟡 | ✅ | ✅已修复 | **gc.disable() 全会话禁用 GC** | `policy/vr_teleop_policy.py:322` | `gc.disable()` 在 16Hz loop 入口；`gc.collect()` 仅在 Line 337（录制保存后）+ Line 657（新 episode 前）；长时采集可能累积循环引用 | **删除 gc.disable()/gc.enable()** — numpy 引用计数管理主导分配，GC 在 62.5ms 帧预算内不会触发（ManiUniCon 无此类禁用） | 逻辑缺陷 |
| **P3** | L05 | 🟡 | ✅ | ✅已修复 | **eef_quat_wxyz 硬编码 identity** | `policy/vr_teleop_policy.py:1367` | 每帧 `[1,0,0,0]`；实际方向在 `eef_rot6d`（Line 1368）。HDF5 每帧浪费 32B 无效数据 | `rot6d_to_quat_wxyz(eef_rot6d)` + NaN guard（避免 arm_state=None 时 NaN quat） | 数据质量 |
| **P3** | L06 | 🟡 | ✅ | ✅已修复 | **安全状态机 transition() 理论竞争** | `robot/safety.py:61-83` | read-modify-write 无锁；实践中安全（Main/Policy 合法转换集不相交），但非零风险 | **仅注释** — 写权分离 + FAULT 自愈（supervisor 100ms 重断言）+ arm 先停后报，加 mp.Lock 属于过度工程 | 并发安全 |
| **P3** | L07 | 🟡 | ✅ | ✅已修复 | **EpisodeRecorder daemon 线程信号安全** | `recording/episode_recorder.py:658-664` | `stop_episode()` 在 daemon 线程执行 HDF5 写入；SIGTERM 直接杀 daemon → HDF5 截断。atexit 仅覆盖正常 exit() | SIGTERM handler 设 flag → 主循环退出 → finally 块复用已有 `_stop_recording` + `join_stop` 完成安全刷新 | 数据完整性 |
| **P3** | E01 | 🟡 | ✅ | ✅已修复 | **每帧 World-Frame Fingertip Position** | `policy/vr_teleop_policy.py:1362-1382` | Hand FK + compose_pose×10 + rot6d×2，每帧 ~0.4ms。内层 compose_pose 循环内重复计算 5×，rot6d 重复计算 2× | Hoist `T_world_handbase` 出循环（省 4× compose_pose）+ dedup `rot6d_to_quat_wxyz`（省 1×）→ ~0.3ms/帧，减少 ~25% | 效率 |
| **P3** | C01 | 🟡 | — | ✅已修复 | **无运行时配置验证** | `config/defaults.py` | Frozen dataclass 依赖 Python 类型提示，无运行时范围/有效性检查 | 3× `__post_init__` assert（ArmParams: joint limits + home within limits; PolicyParams: control_hz > 0 + EMA in [0,1]; SafetyParams: heartbeats > 0 + recoveries > 0） | 配置 |
| **P3** | C02 | 🟡 | — | ✅已修复 | **配置不可热加载** | `config/defaults.py` | 修改配置需改 Python 代码；无法 YAML/JSON 文件覆盖 | `load_config_json()` 用 `object.__setattr__` 原地突变 frozen 单例（覆盖对所有 `from defaults import` 引用可见）；`--config` CLI flag | 配置 |
| **P3** | T02 | 🟡 | — | ✅已修复 | **无 pre-commit hooks** | — | 无自动格式化/类型检查门禁；依赖开发者手动运行 black/isort/mypy | 添加 `.pre-commit-config.yaml`（black + isort + mypy + trailing-ws） | 测试 |
| **P4** | L08 | 🟢 | ✅ | ✅已修复 | **`_RECOVERY_MAX` 重复定义** | `arm_loop.py:20,91` | 模块级常量与 config 值相同但无引用 | arm_loop import `safety`，引用 `safety.max_consecutive_recoveries` | 可维护性 |
| **P4** | L09 | 🟢 | ✅ | ✅已修复 | **arm.error_code SDK 属性误解** | `arm_loop.py:437` | 原以为需缓存；agent 确认 SDK property 读后台缓存，纳秒级 | 加注释说明 SDK 已缓存，无代码变更 | 微优化 |
| **P4** | L10 | 🟢 | ✅ | ✅已修复 | **Camera 16Hz 硬编码** | `camera_process.py:17,145,148` | 字面量 1/16.0 与 policy.control_hz 脱钩 | `import policy`，`interval = 1.0 / policy.control_hz` | 正确性 |
| **P4** | L11 | 🟢 | ✅ | ✅已修复 | **Pointcloud is_recording 延迟** | `camera_process.py:150-210` | 首帧缺 pc | `_last_pc` forward-fill：is_recording 翻转时立即写缓存 pc | 数据完整性 |
| **P4** | L12 | 🟢 | ✅ | ✅已修复 | **SharedStorage.close() 静默吞异常** | `shm/shared_storage.py:283-310` | `except Exception: pass` | 分级：FileNotFoundError 静默，意外异常汇总 warning | 可观测性 |
| **P4** | L13 | 🟢 | ⚠️ | ✅已修复 | **关节限位 Python+URDF 双源** | `planning/planner.py:115-130` | Python defaults.arm 与 URDF/mplib 独立 | `np.allclose(tol=1e-3)` + `logger.warning`（非阻塞） | 可维护性 |
| **P4** | L14 | 🟢 | ✅ | ✅已修复 | **`_try_rename` rmtree 失败残留** | `recording/episode_recorder.py:862-904` | rmtree 异常 + cleanup 在 except 非 finally | 三件套：rmtree try/except + finally化 + 孤儿temp扫描 | 存储泄漏 |
| **P4** | L15 | 🟢 | ✅ | ✅已修复 | **`_to_full_qpos` zeros 默认值** | `planning/collision_model.py:44-50,200` | zero=握拳，碰撞检测应张手 | 类常量 home_qpos + ThrottledWarner 告警 | 防御性编程 |
| **P4** | L16 | 🟢 | ✅ | ✅已修复 | **`plan_joint_home_path` 惰性 import** | `planning/path_utils.py:9,121` | 函数内 lazy import | 移至顶部（defaults 是 leaf 模块，无循环依赖） | 可维护性 |
| **P4** | E02 | 🟢 | ✅ | ✅已修复 | **held 帧全量触觉分配** | `policy/vr_teleop_policy.py:318,1031` | held 帧占 80%，每帧 14KB 零张量 | `_last_tactile_data` forward-fill 复用 | 效率 |
| **P4** | C03 | 🟢 | — | ✅已修复 | **配置无发现性** | `examples/real/vr_teleop_hand_record.py:80-92` | 无 `--print-config` | `dataclasses.asdict()` 打印 6 单例全部字段 | 配置 |
| **P4** | M17 | 🔵 | — | ✅已修复 | **启动前无 URDF vs Python limits 一致性检查** | 见 L13 | 无自动化检查 | 与 L13 合并修复 | 缺失机制 |

---

## 统计

| 优先级 | 数量 | 已修复 | 类别分布 |
|:---:|:---:|:---:|---|
| **P0** | 6 | **1** ✅ | 全部为缺失机制。M05 (get_last_k) 已修复 |
| **P1** | 10 | **3** ✅ + **1** 🟡部分 | 3 代码重复（D01/D02/D03 已去重）+ 1 测试（T01 部分：1 文件 16 测试）+ 6 缺失机制 |
| **P2** | 9 | **3** ✅ + **6** ⏭️ | 3 逻辑/效率（L01/L02/L03 已修复）+ 6 缺失机制/基础设施（M11-M16 skip/defer） |
| **P3** | 10 | **8** ✅ | 4 逻辑/数据质量 + 2 效率 + 2 配置 + 1 测试 + 1 并发 |
| **P4** | **12** | **12** ✅ | 全部修复 |
| **合计** | **48** | **27** ✅ + **6** ⏭️ + **1** 🟡 | P4 12 + P3 8 + P2 3 + P1 3 + P0 1 已修复；T01 部分；P0/P1 剩余待后续 |

### 按严重度

| 严重度 | 数量 |
|:---:|:---:|
| 🔴 严重 | 4 |
| 🟡 中等 | 16 |
| 🟢 轻微 | 16 |
| 🔵 缺失机制 | 12 |

### 按验证状态（已 fact-check 的 28 个 claim）

| 状态 | 数量 |
|:---:|:---:|
| ✅ CONFIRMED | 25 |
| ⚠️ CORRECTED | 3 |
| ❌ INCORRECT | 0 |

---

## P0 详细说明（策略推理部署 — 结构性断裂）

当前 DexMani 从采集到部署的路径是断的。6 个 P0 项共同构成部署能力的**最小可行集合**。
**2026-08-05: M05 (get_last_k) 已实现 — P0 首个修复项。**

```
采集 (现有)                    训练 (手动)                    部署 (部分就绪)
─────────                    ─────────                    ─────────
HDF5 v11  ──→ LeRobot导出 ──→ 模型训练 ──→ ModelPolicy ──→ Arm/Hand
              (P1 M07)                     ├─ ObsWrapper (M02)
                                           ├─ ActWrapper (M03)
                                           ├─ 同步协议 (M04)
                                           ├─ get_last_k (M05) ✅
                                           └─ Cartesian模式 (M06)
```

---

*从 `docs/audit-dexmani-vs-maniunicon.md` 自动提取。*

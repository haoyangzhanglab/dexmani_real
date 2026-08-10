# DexMani Real `7427ab0` 全仓复核与整改报告

> 审查基线：DexMani Real `7427ab0`  
> 历史基线：`167f15a`（历史报告保持不变）  
> 对照基线：ManiUniCon `85c6f2e`（复用既有固定版本逐行证据）  
> 整改对象：`7427ab0` 之上的当前工作树  
> 日期：2026-08-10  
> 安全边界：仅静态检查、mock、共享内存、临时 HDF5 和纯数学/性能测试；未连接或探测 xArm、XHand、Quest、L515

## 1. 结论

历史 50 项 finding 中，当前软件状态为 **46 fixed、4 partially-fixed**。两个历史 P0 和 27 个历史
P1 均已有软件修复及离线回归；P2 中仍未闭合的是 F-15、F-16、PD-17、PD-20。16 项 ManiUniCon
移植 finding 为 **12 mitigated、2 obsolete-for-current-topology、2 partially-fixed**。

本轮另确认并修复四项问题：学习策略旧 chunk 被重新定时（PD-21，P1）、replay 的 `finally`
返回吞掉异常（ENG-01，P1）、IPC schema 反向依赖（ARCH-01，P2）、observation 重复扫描/无用
camera copy（PERF-01，P2）。同时发现 `7427ab0` 的 XHand stall 修复存在窗口逻辑退化；已在
XH-SDK-03 原编号下标记为 `regressed → fixed` 并增加直接回归。

当前没有已知未修复的 P0/P1 软件 finding，但这不等于真机放行。xArm Mode 6、EtherCAT 清理、
触觉单位/带载校零、真实模型 p99 和 nominal URDF 绝对精度仍必须按第 9 节人工验证。

## 2. 状态口径

- `fixed`：触发链已被软件阻断，并有直接或同边界离线测试。
- `partially-fixed`：核心风险降低，但指标、校准或测试覆盖未达到原验收条件。
- `regressed → fixed`：`7427ab0` 仍可触发，本轮已修复。
- `obsolete`：当前单 L515 拓扑或保留的 Dex 不变量使参考风险不可达；扩展拓扑时需重开。
- `unverified-on-hardware`：软件路径已修复，但厂商 SDK/固件/物理结果不能离线证明。

## 3. 历史 F findings 映射

| ID | 等级 / 状态 | 当前源码证据、触发与影响 | 整改与验证 |
|---|---|---|---|
| F-01 | P0 / fixed | `examples/real/replay_traj.py` 要求 v15 sent-stream、模型 provenance 与 `PreflightCertificate`；篡改轨迹或绑定不符会在任何 live publish 前失败。旧影响是端点合法但中点碰撞仍进入 Mode 6。 | 全入口经 `planner_action_safety_gate → publish_joint_targets`；`test_preflight_certificate_detects_tampering_and_binding_changes`、`test_hardware_entry_points_supply_geometry_aware_action_gate`、中点 workspace/path 测试验证零旁路。 |
| F-02 | P1 / fixed | `robot/arm_loop.py` 的启动和 ARMED 边沿统一检查 setter 返回码及 live mode/state 后置条件；失败 sticky fault。旧触发是 cached state 与真实 state/mode 不同。 | `test_planned_homing_fails_if_mode6_cannot_be_restored`、mode0/state failure/controller fault 测试覆盖。真机 setter 时序仍需人工确认。 |
| F-03 | P1 / fixed | `arm_loop.py` ready 前确认 state 4，只有 ARMED/RUNNING 才进入 Mode 6/state 0；DISARMED/FAULT 回到 state 4。旧影响是软件 DISARMED、控制器却可运动。 | `test_hand_disarmed_startup_validates_feedback_without_home_motion` 与 arm lifecycle mock 覆盖；cleanup 只有确认 state 4 后发 STOPPED ACK。 |
| F-04 | P1 / fixed | `ARM_STATE_DTYPE` 分开 source/publish monotonic time 与 `state_valid`；坏 shape、NaN、SDK read failure 不推进有效 source time并触发 watchdog。 | `test_arm_feedback_boundary_and_watchdog_reject_persistent_bad_reads`、causal snapshot 测试覆盖 stale/future/invalid。 |
| F-05 | P1 / fixed | `SafeCommandPublisher` 对 arm queue 使用有界 timeout；supervisor safety-first；`shutdown_processes_verified` 必须确认 kill/exit 后才能 cleanup；e-stop 在 worker loop 顶部优先。 | `test_supervisor_exit_priority_is_safety_first`、`test_shutdown_escalates_to_kill_before_shared_cleanup`、`test_shutdown_keeps_shared_memory_when_exit_is_unconfirmed`。 |
| F-06 | P1 / fixed | `arm_loop._recover_c24_measured_hold()` 丢弃故障目标，读取 fresh measured qpos，只发一次 hold；C22/C31 直接 fault。 | `test_c24_recovery_sends_exactly_one_fresh_measured_hold`、invalid measurement 测试。物理 C24 恢复幅度未制造验证。 |
| F-07 | P1 / fixed | runtime capability graph 按 `hand_enabled` 解析 worker/readiness/heartbeat，arm-only 启动不等待 hand。 | `test_optional_hand_startup_failure_does_not_fault_arm_only_entry`、resolved capability/config 测试。 |
| F-08 | P2 / fixed | schema v15 分开 observation、safe action、prepare/commit、ACK、arm accepted/applied timing与反馈 provenance。旧的 sent/accepted/executed 混义不再作为 v15 语义。 | `test_recording_provenance_*`、`test_episode_recorder_persists_arm_command_timing`、v15 round-trip。v12–v14 只标 `UNKNOWN`。 |
| F-09 | P2 / fixed | `_planned_homing` 同时要求 qpos、低 qvel 和完整 dwell，恢复 Mode 6 后才 ACK。 | `test_planned_homing_requires_low_velocity_for_full_dwell`、single-point high-velocity、timeout/fresh feedback 测试。 |
| F-10 | P2 / fixed | `config/runtime.py` 生成不可变 `ResolvedRuntimeConfig`，固定 CLI > JSON > defaults，完整交叉校验并持久化 canonical SHA-256。 | `test_runtime_config_precedence_is_immutable_and_hash_is_canonical`、invalid/capability propagation 测试。 |
| F-11 | P1 / fixed | `utils/signal_utils.py` 在 SO(3) 相对旋转上做 shortest-arc 平滑，不再线性混合绝对 rotvec。 | `test_pose_ema_takes_short_arc_across_plus_minus_pi`。 |
| F-12 | P2 / fixed, unverified-on-hardware | `arm_loop.py` 只在 committed action 到 target time 时调用 `set_servo_angle`，queue 空不再逐 tick 重发。 | action prepare/commit/expiry tests 静态证明 new-action-only；固件是否去重旧 endpoint 不再影响应用行为。 |
| F-13 | P2 / fixed | action 含 observation anchor、target、valid-until、epoch、session、action/chunk ID；worker 二次拒绝过期/旧 epoch/乱序；delta clamp 使用真实 dt。 | `test_worker_rejects_stale_or_invalid_commands`、chunk overlap/ready-order/expiry 测试。 |
| F-14 | P2 / fixed, unverified-on-hardware | v15 metadata 记录 resolved speed/acc、firmware/serial/model hash，并显式声明 `jerk_management=unmanaged`；live replay provenance 不完整或不一致即拒绝。 | replay provenance 检查与 config hash 测试；真实 jerk/固件差异仍需固定控制箱 A/B。 |
| F-15 | P2 / partially-fixed | arm state与 HDF5 已有 generated/received/applied/SDK-duration 字段，但 `episode_quality.py` 的主 tracking 指标和 replay 汇总仍保留部分同索引误差，尚未完整输出 lag-compensated residual、settled error、overshoot/settling。 | timing round-trip 已测；仍需合成固定 3-frame delay 的指标回归和统一 quality/replay 实现。 |
| F-16 | P2 / partially-fixed | episode/replay 已绑定 URDF/SRDF/arm-hand model hash、serial/firmware，但 FK 仍使用 nominal collision URDF，尚无 serial→厂商校准运动学 artifact。 | provenance mismatch 已 fail closed；绝对 TCP 精度只能通过 per-robot calibration 与外部测量闭合。 |

## 4. 历史 PD findings 映射

| ID | 等级 / 状态 | 当前源码证据、触发与影响 | 整改与验证 |
|---|---|---|---|
| PD-01 | P0 / fixed | `ActionSafetyGate` 与 `SafeCommandPublisher` 是正式边界，所有 hardware entry 均提供 geometry-aware gate；raw IPC 写入只允许 publisher。 | `test_only_safe_command_publisher_writes_raw_actuator_ipc`、missing/reduced gate、dt clamp 测试。 |
| PD-02 | P1 / fixed | canonical runtime 从统一 `mp.get_context("spawn")` 创建 primitive/process；SDK/model在 owner child 内构造。 | backend import/static boundary 与 runtime config tests；未运行 CUDA 真模型。 |
| PD-03 | P1 / fixed | policy/inference/RecorderIO 均由 main 显式监督，不依赖 daemon 自动回收；模型有独立 inference process。 | verified shutdown tests覆盖 TERM→KILL 与未确认退出禁止 unlink。 |
| PD-04 | P1 / fixed | inference manifest/resource hash、load/warmup/finite output、capability sensor mask和 component phase/generation 构成 ready 协议。 | manifest hash/shape、camera generation rewarm、startup failure tests。 |
| PD-05 | P1 / fixed | learned backend 在 inference child 执行；coordinator独立运行 heartbeat、deadline、scheduler、安全和 recorder status，mailbox/tensor block 为双槽 latest snapshot。 | tensor block、slow/expired candidate、camera reset rewarm测试。VR 的 IK仍在 policy owner中，但不存在通用模型 native hang共享线程。 |
| PD-06 | P1 / fixed | main-owned `is_running` 与结构化 `ComponentPhase/FaultCode/ExitReason` 分离；worker不再把异常写成用户 Q。 | `test_worker_modules_do_not_own_global_shutdown_flag`、supervisor priority tests。 |
| PD-07 | P1 / fixed | shutdown 必须 cooperative join→TERM→recheck→KILL→final join；存活进程存在时保留 SHM。 | 两个 shutdown fault-injection tests。 |
| PD-08 | P1 / fixed | `ObservationSnapshot` 按 host monotonic anchor 因果选帧，保存 source/receive/publish、seq、age/skew、valid mask、generation；VR recorder也使用 causal reader。 | empty/pad、future frame、cross-modal skew、camera generation、source sequence tests。 |
| PD-09 | P1 / fixed | camera header同时保存 device capture、host receive/publish monotonic、frame number和 generation；device reset只增 generation，不依赖 wall clock。 | `test_device_clock_mapping_detects_duplicate_gap_and_reset`、camera metadata round-trip。 |
| PD-10 | P1 / fixed | `ActionCandidate`/fixed command dtype包含 observation/session/epoch/action/chunk/timing/frame/unit；两 worker二次验证。 | TTL、epoch、out-of-order、identity mismatch tests。 |
| PD-11 | P1 / fixed | arm/hand共用 action identity，经 PREPARE/paired ACK 后 COMMIT；各 worker回 APPLIED/REJECTED/SDK_FAILED。 | full identity matching、partial prepare/SDK failure、record provenance tests。 |
| PD-12 | P1 / fixed | policy-side `JointActionScheduler` 有界替换未 committed future，保留 committed step，丢 expired prefix；不扩 arm queue、不做 robot interpolation。 | overlap/replace/drop、prepare window、ready action order、whole chunk expiry测试。 |
| PD-13 | P1 / fixed | session generation + policy epoch + quiesced measured hold + scheduler/backend reset；camera generation reset强制 ARMED/rewarm。 | `test_camera_generation_change_forces_armed_epoch_and_rewarm`、worker old-epoch rejection。 |
| PD-14 | P2 / fixed | `PolicyBackend`、immutable `ObservationSnapshot`、`ActionCandidate/ActionChunk`、`ActionSpec`和 isolated runner已成为正式接口；backend无 SharedStorage。 | dummy manifest/tensor/invalid contract tests和静态 import boundary。 |
| PD-15 | P2 / fixed | ring容量由 `ObservationSpec.required_ring_capacity` 推导并启动校验；history oldest-first、may-return-fewer、mask显式。 | derived capacity、short history、ring wrap/seqlock tests。 |
| PD-16 | P2 / fixed | camera先扫描小 header，只按被选 sequence/请求 modality调用一次 `read_sequence`；VR recording-off不读取 camera payload；structured同一ring每 snapshot只扫描一次。 | `test_camera_snapshot_never_mixes_payload_generations` 断言只复制 RGB；`test_observation_source_scans_each_structured_ring_once_per_snapshot`。 |
| PD-17 | P2 / partially-fixed | manifest已有 device/dtype/deadline/memory/resource hash与 tensor budget；但没有真实 checkpoint + RGB-D/PC + RecorderIO负载下固定主机 p95/p99、CPU/GPU/RSS准入报告。 | 共享 runner仅做宽松 12.5 ms snapshot/commit smoke；严格矩阵保留为验收项。 |
| PD-18 | P2 / fixed | schema v15 additive记录输入序列/age/fill、raw/safe action、command/ACK、manifest/provenance；reader保持 v12–v15 可读，训练/live replay默认只收语义 VALID v15。 | v13/legacy兼容、v15 round-trip、quality filter tests。 |
| PD-19 | P2 / fixed | immutable resolved config、typed manifest、显式 backend entrypoint和资源 SHA-256替代 mutable singleton插件配置。 | precedence/hash/unknown field/cross-invalid tests。 |
| PD-20 | P2 / partially-fixed | 离线 suite从历史90增至本轮完成后的174项，并新增 GitHub Actions、Ruff与coverage ratchet；全包实测48.57%，未达到计划70%，但本次changed-line coverage达到93%。硬件loop与VR主循环仍是主要缺口。 | CI只执行 `pytest tests`，禁止执行 `examples/real/test_*`；PR额外执行90% diff coverage门槛；仍需用fake SDK/clock提高全包覆盖。 |

## 5. 历史 XH 与 DATA findings 映射

| ID | 等级 / 状态 | 当前源码证据、触发与影响 | 整改与验证 |
|---|---|---|---|
| XH-SDK-01 | P1 / fixed, unverified-on-hardware | `XHand.connect()` 在 native open 后把 enumerate/verify/command/init 包在统一 `try/except`，任何异常调用幂等 `disconnect()`/INIT/close。 | `test_xhand_post_open_failure_always_enters_disconnect_cleanup`；真实 slave INIT/watchdog仍需 EtherCAT验证。 |
| XH-SDK-02 | P1 / fixed, unverified-on-hardware | hand loop独立追踪 send/read/board fault；持续 send fail立即 sticky `shared.error_state`，健康 read不能清该 latch。 | `test_hand_runtime_counts_boolean_command_rejection` 直接断言 fault latch。 |
| XH-SDK-03 | P1 / regressed → fixed | `7427ab0` 把 stall观察窗口递减到阈值前，静止误报虽消失但真实无进展也无法稳定触发。本轮 `_update_tracking_stall` 仅在目标未收敛且反馈无进展时累积并保持 active。 | `test_hand_tracking_stall_distinguishes_settled_feedback_from_no_progress` 覆盖已收敛静止与15帧无进展。 |
| XH-RT-01 | P1 / fixed | `validate_hand_landmarks` 检查 finite、掌宽、骨长、掌面条件与退化，失败hold且不污染 temporal state。 | zero/collinear/NaN/short-bone测试。 |
| XH-TACT-01 | P2 / fixed, unverified-on-hardware | hand worker每次成功read都写 tactile ring，包括 release；payload带 source/fresh/calibrated/unit，失败帧不会伪造fresh。 | camera/schema recorder round-trip覆盖字段；真实 contact→release需真机。 |
| XH-TACT-02 | P2 / fixed, unverified-on-hardware | startup load或无效tactile数据使 calibration fail closed；bias采样中检测载荷也拒绝，unit保持unknown而非伪称N。 | `test_xhand_tactile_startup_load_check_fails_closed`。 |
| XH-RT-02 | P2 / fixed | pinky scaling先保存原始父子骨段再统一缩放，不混用更新后的父节点。 | `test_pinky_scaling_uses_each_original_parent_child_segment`。 |
| XH-RT-03 | P2 / fixed | DexPilot保存 `np.argsort(mapping)` 的真实 SDK→internal inverse并reset完整状态。 | `test_dexpilot_reset_uses_true_sdk_to_internal_inverse_mapping`。 |
| XH-ARCH-01 | P2 / fixed | hand retarget从 defaults/model config取limits；vendor SDK只在 hand worker懒导入，backend/runtime静态禁止 native依赖。 | `test_backend_runtime_import_surface_is_device_and_gui_free`。 |
| XH-SDK-04 | P3 / fixed | SDK缺失默认 fail closed；只有 `simulation_backend=True` 才启用following simulation，command→feedback一致且identity标simulation。 | `test_xhand_missing_sdk_is_fail_closed_unless_simulation_is_explicit`。 |
| DATA-01 | P1 / fixed | `TimestampAlignedBuffer` 缺槽只用过去样本hold或invalid，不允许未来到达样本回写过去grid，并保存source index/valid。 | 三个 timestamp causal/fill tests。 |
| DATA-02 | P1 / fixed | camera freshness按source age、frame/generation/duplicate判断；停流可中止episode但不伪造新帧。 | freshness duplicate/old/cross-episode、quality flag tests。 |
| DATA-03 | P1 / fixed | v15 camera sidecar按每个grid slot写等长RGB/depth/PC与fresh/valid映射，不再仅尾部补MP4。 | equal-length sidecar、grid backfill、decoded length verification tests。 |
| DATA-04 | P1 / fixed | quality/filter在隐藏temp目录按同一mask重写所有sidecar，verify/fsync后atomic rename；失败不发布partial。 | filter camera flags、writer failure、rename failure、no overwrite tests。 |

## 6. ManiUniCon MU-01 至 MU-16 映射

这些 finding 是“不要把参考机制原样移植”的约束。这里的 `fixed` 表示 Dex 已实现对应缓解，不是修改了
ManiUniCon。对照源码不在当前 git object database；证据来自固定 `85c6f2e` 的既有逐行报告
`policy_deployment_process_sync_deep_review.md`，本轮没有重新运行参考项目。

| ID | 等级 / 状态 | Dex 决策、触发影响与验证 |
|---|---|---|
| MU-01 | P1 / fixed | 采纳 spawn，但SDK/model只在owner child构造；main只传配置和IPC。静态import边界及spawn lifecycle验证。 |
| MU-02 | P1 / fixed | backend只收 immutable snapshot、只返 candidate/chunk，不继承Process、不接触raw SharedStorage；静态raw IPC测试阻断旁路。 |
| MU-03 | P1 / fixed | 禁止双Event无超时锁步；改为target/TTL、PREPARE/ACK/COMMIT/APPLIED和有界等待。identity/prepare timeout tests覆盖。 |
| MU-04 | P1 / fixed | main监督phase/heartbeat/death/exit reason，所有wait有deadline，退出未确认不unlink。shutdown fault injection覆盖。 |
| MU-05 | P1 / fixed | history按monotonic target causal选择，短历史显式invalid padding且同source sequence不重复。snapshot/history tests覆盖。 |
| MU-06 | P1 / obsolete | 当前只有单L515，不存在多相机mean timestamp融合；未来增加相机时必须逐source保留时间/seq/generation/skew，禁止复用mean方案。 |
| MU-07 | P1 / fixed | 控制、TTL、heartbeat、age全部host monotonic ns/s；wall clock只作人类日志。device reset/wall-independent tests覆盖。 |
| MU-08 | P1 / fixed | expired whole chunk被丢弃并进入coordinated hold语义，绝不重定时旧预测；本轮PD-21回归直接覆盖。 |
| MU-09 | P1 / fixed | 保留arm queue `maxsize=2`；chunk仅在policy scheduler，未committed future可替换、expired可丢弃，不保证旧chunk全部执行。 |
| MU-10 | P1 / fixed | reset使用generation/epoch、quiesced hold、queue/ACK/backend/scheduler换代，不使用可丢Event。camera generation rewarm覆盖。 |
| MU-11 | P1 / fixed | RecorderIO集中写全部sidecar，stop/drain/verify/fsync/atomic publish；overflow/codec/IO失败abort episode。recorder process tests覆盖。 |
| MU-12 | P1 / fixed | timestamp future-fill被因果past-fill+valid/source provenance替代；reader/training默认尊重semantic validity。 |
| MU-13 | P2 / obsolete | 未移植基于拷贝时限假设的lock-free ring；继续使用seqlock verified read、wrap时可少返，不静默伪造历史。 |
| MU-14 | P2 / fixed | typed Observation/ActionSpec、manifest和resolved config启动即校验horizon/dt/shape/unit/frame；不依赖弱duck typing。 |
| MU-15 | P2 / partially-fixed | optional model仅inference child动态导入，resource有SHA-256且普通VR不加载；但backend entrypoint仍是受信本地Python代码，尚无组织级allowlist/签名供应链策略。 |
| MU-16 | P2 / partially-fixed | 已消除未请求camera payload copy、重复ring scan和部分热路径fallback allocation，并启用保守Ruff性能规则；缺固定主机全负载10–30分钟p99/RSS基线。 |

## 7. 本轮新增 findings

### PD-21 — P1 — fixed：coordinator 重新定时并复活旧 backend chunk

- **源码证据**：`7427ab0:policy/learned_coordinator.py` 在消费candidate时把
  `created/target/valid_until` 全部改写为 `now + lead + index*dt`。
- **触发**：backend输出已过期chunk，直到coordinator较晚消费。
- **影响**：旧预测获得新TTL并进入硬件scheduler，违反observation因果与“全chunk过期应hold”。
- **修复**：coordinator现在只归一化受信identity（session/epoch/action/chunk/step），完整保留backend monotonic timing。
- **验证**：`test_learned_coordinator_normalizes_identity_without_retiming_backend_action`、
  `test_learned_coordinator_never_revives_an_expired_backend_chunk`。

### ENG-01 — P1 — fixed：replay `finally` 返回吞掉控制异常

- **源码证据**：`7427ab0:examples/real/replay_traj.py` 的 `TrajectoryReplayer.run()` 在 `finally`
  中返回recorder dict；Python会压制try中任何SDK/publish/runtime异常。
- **触发**：replay loop抛异常且recorder存在。
- **影响**：危险的动作/SDK失败可能被调用者误解为正常提前结束，事故语义丢失。
- **修复**：`finally` 只做 `kb.stop()`，结果返回移到try/finally之后；prewarm early return也先停止keyboard。
- **验证**：`test_replay_never_returns_from_finally_and_swallows_motion_errors` 静态防回归；Ruff `B012`。

### ARCH-01 — P2 — fixed：SharedStorage 反向依赖高层协议实现

- **源码证据**：`7427ab0:shm/shared_storage.py` 顶层从policy inference/recording导入dtype，且
  `write_hand_cmd` 懒导入policy；这使data plane依赖业务层。
- **触发**：离线main/storage import、schema变更或可选policy依赖缺失。
- **影响**：破坏SDK/模型隔离，schema变更跨层循环，增加spawn import副作用。
- **修复**：新增NumPy-only `ipc/schema.py`，统一action/state/VR/camera/inference/record dtype；手部协调发布/回零迁到
  `policy/action_protocol.py`，SharedStorage只分配transport与通用状态读取。
- **验证**：`test_ipc_schema_layer_has_no_policy_or_recording_dependency`、mypy和全suite。

### PERF-01 — P2 — fixed：observation 重复扫描与无用camera payload复制

- **源码证据**：`7427ab0:policy/observation_sources.py` 每个modality各自 `get_last_k(maxlen)`，camera
  每个modality/sequence分开读payload；VR主loop recording-off仍读camera。
- **触发**：一个arm ring请求qpos/qvel/tau，或RGB+depth/PC history，或未录制VR遥操作。
- **影响**：重复seqlock扫描、数组分配和MiB级payload copy挤占64/16 Hz预算。
- **修复**：每物理ring只扫描一次小结构；camera先一次header scan，按被选sequence合并请求modalities，invalid/unrequested
  payload零复制；recording-off不读camera。
- **验证**：mock精确断言metadata一次、payload一次且只请求RGB；structured ring一次；10,000 tick净增长<1 MiB。

### ENG-02 — P2 — partially-fixed：CI/覆盖率验收缺口

- **证据**：基线没有GitHub Actions、Ruff/coverage依赖；实测 `dexmani_real` 总覆盖率48.57%，不是计划假设的70%。
- **影响**：硬件loop、VR主循环、规划器与可视化回归仍可能逃逸；若直接设70%会提交必失败CI。
- **整改**：新增 `requirements-dev.txt`、`requirements-ci.txt`、保守Ruff规则和offline-ci；coverage先以48% ratchet防回退，PR changed-line coverage以90%为门槛。
- **后续验证**：只用fake SDK/clock/process增加有行为意义测试，达到70%后再提高门槛；不靠omit硬件模块虚增比例。

## 8. 验证记录与未达项

本轮开始时 `7427ab0` 工作树干净，基线为164 tests；整改后为174 passed、全包coverage 48.57%、
changed-line coverage 93%。已建立的验证集合包括：compileall、Ruff、Black、isort、mypy、全量
`pytest tests`、coverage、10,000 tick内存稳定性和宽松shared-runner p99。CI明确不运行
`examples/real/test_*` 或任何hardware entry。

计划目标中尚未达到：

1. `dexmani_real` 总行覆盖率70%（当前48.57%）；本次新增/修改行为93%，PR CI门槛锁定90%。
2. 固定 `real_robot` 主机上的完整64 Hz coordinator、30 Hz worker框架、16 Hz policy、RecorderIO enqueue
   p95/p99和优化前后10%回归线。
3. 真checkpoint、GPU显存/OOM、RGB-D/pointcloud+recording背景负载的10–30分钟准入。
4. F-15完整lag-compensated tracking指标和F-16 per-robot calibrated kinematics。

## 9. 真机人工验收清单（未执行，需另行明确授权）

前置条件：低速、空载、工作区清空、有人守物理急停；任何一步异常立即FAULT/state 4并停止。

1. xArm/XHand连续20次 connect→DISARMED→ARMED→RUNNING→state-4/INIT→disconnect，无power-cycle。
2. xArm startup/mode/state/C24各点故障注入；C24后只发送一次fresh measured hold，C22/C31立即sticky fault。
3. XHand read-only与write-only故障分别触发独立health domain；静止已收敛10秒不stale，未收敛固定反馈触发stall。
4. 触觉无接触显式校零、带载启动拒绝、contact→release的raw/contact/fresh一致；确认厂商物理单位。
5. Quest遮挡、L515停流/frame freeze/camera generation reset均hold或回ARMED，episode标invalid且不混代。
6. arm/hand paired action ID、PREPARE/COMMIT/APPLIED和feedback seq可计算apply skew；一侧SDK失败不得记联合成功。
7. replay只接受v15 semantic-valid、provenance与preflight hash匹配轨迹；中点碰撞/证书篡改时零命令。
8. RGB/depth/pointcloud内部缺帧及filter后逐slot marker与state/action同时间轴。
9. ±π短弧、退化landmark、pinky全行程和DexPilot/TAG reset首帧无非意图跳变。
10. 目标模型+recording+RGB-D/PC运行10–30分钟，报告控制/推理/copy/enqueue/action age的p50/p95/p99/max、deadline miss和RSS。

## 10. 放行判定

软件层面的P0/P1整改可以进入离线review；不能据此宣称硬件已验证。训练和live replay继续默认只接受schema v15
语义有效episode。严格实时准入、70%覆盖率、F-15/F-16和第9节真机清单完成前，系统仍只适合低速、有人值守、
受控实验室运行。

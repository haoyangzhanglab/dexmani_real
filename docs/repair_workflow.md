# DexMani Real 分阶段修复 Workflow

本文落实用户提供的分阶段修复计划，负责执行顺序、agent 分工和验收协议。
原计划中的明确约束继续有效；本文不把尚未复核的问题认定为当前源码缺陷。
执行事实与测试证据记录在 [repair_progress.md](repair_progress.md)。

本轮定位：个人 PhD 真实机器人研究代码的小范围 correctness 修复。
优先级：实验结果正确 → train/deploy 语义一致 → lifecycle 正确 → 数据可追溯
→ 实现简单 → 抽象与扩展。

## 1. 入口与授权范围

制定 workflow 不等于执行修复。收到执行本 workflow 的指令后，主 agent 从 baseline 开始，
逐 Phase 完成软件修复和验收。通过 gate 后自动推进，不逐阶段重复请求确认。
任一时刻只允许一个 Phase 处于实施状态；其内部子步骤也必须先回归再推进。

Phase 1 的实施范围包括必要的 `../dexmani_policy` deployment/export、Zarr contract
及对应测试。其余 Policy 训练、评测和算法代码不属于本轮。
跨仓库任务先读目标仓库的 `AGENTS.md`；任务范围不替代实际文件系统写入权限。
如写入被环境拒绝，报告具体受阻操作，继续不依赖它的已授权工作，不绕过权限。

执行入口：

1. 阅读两仓库适用的 `AGENTS.md`、相关 coding conventions、本文和进度文件。
2. 两仓库分别运行 `git status --short`、`git rev-parse HEAD`、`git diff --check`。
3. 与原计划基线对照；HEAD 变化时重新追踪相关调用链，不机械套用旧结论。
4. 记录已有 dirty 文件和 relevant diff；已有改动保留，重叠文件须明确 patch 归属。
5. 检查测试的 import、fixture 和构造路径，确认不会初始化真实设备，再运行现有纯软件测试。
6. 记录准确命令、环境、退出结果和基线失败。环境缺失、测试未运行和代码失败分别记载。

Python 环境按目标仓库选择：Real 使用 `conda run -n real_robot`，Policy 使用
`conda run -n policy`。跨 repo agent 的任务书必须显式说明这个区别；不要把 Real
agent 模板中的环境默认值套用于 Policy。

## 2. 三档 agent 与主 agent 职责

复用 `.codex/agents/` 已有三档定义，不新建 agent framework，不修改权限配置。

| 角色 | 责任 | 典型任务 | 限制 |
| --- | --- | --- | --- |
| 主 agent | 拆任务、分配文件 owner、维护进度、接受证据、决定 Phase gate | 集成审查、最终回归、停止条件处理 | 不以子 agent 的 READY 代替整阶段验收 |
| sol-high | 复杂语义、跨仓库 contract、时序和控制安全边界 | FK identity、freeze timing、stationary sampling、clean-Q、publication 状态 | 所有决定必须对应调用链和失败测试 |
| terra-xhigh | 已明确 invariant 的多文件实现与集成 | dataset/schema/export、freshness、provenance、replay 局部修复 | 遇到未决安全或 ownership 问题提交证据给主 agent |
| luna-max | 窄范围文档、机械迁移、清单与证据核查 | callsite inventory、fixture/schema 常量同步、文档措辞、运行明确的离线检查 | 不决定安全状态、schema 含义或最终安全验收 |

档位按任务不确定性和边界风险选择，不按修改行数选择。高风险 contract 先由 sol-high
核实并定下 invariant，再由 terra-xhigh 实现明确的部分；很小的工作由主 agent 直接完成。
不要求每个 Phase 都调用三档 agent。

并行默认最多两个子任务，并服从会话实际并发上限。只并行当前 Phase 内相互独立的工作，
例如一人实现、一人只读核查另一侧 consumer。依赖同一 semantic mapping 或共享文件的
实现串行进行。不得提前实现下一 Phase。

文件所有权规则：

- 每个 production/test 文件同时只有一个 writer；只读 reviewer 可以并行。
- 测试与对应修复原则上交给同一 owner，保证先 red、后 implementation。
- reviewer 不直接修补被审文件；问题交回 writer，必要时显式转交所有权。
- 子 agent 不递归派生、不自行提交、不修改进度文件、不自行推进 Phase。
- 所有人都在共享 worktree 中工作，不回滚、覆盖或提交他人的改动。
- 主 agent 承担验收；高风险 Phase 可在 writer 完成后委派 sol-high 做只读复核。

## 3. 每个修复单元的 gate

每次任务书必须列出：Phase/子步骤、事实证据、当前与目标 invariant、允许编辑文件、
只读边界、禁止行为、环境、red test、green/related tests、交付要求。
文件清单通过源码追踪获得，不能仅照抄原计划中的建议路径。

执行顺序：

1. **Fact-check**：追踪 definition → producer → transformation → consumer → side effect，
   同时检查改变边界的两侧。明确旧代码具体违反哪个 invariant。
2. **Red**：先写 regression test，运行并记录旧实现因目标行为失败的证据。
   import error、依赖缺失、fixture 错误不算有效 red。
3. **Implement**：只修复已证明的问题，不改无关逻辑。
4. **Green**：运行该 regression 和相关模块测试；失败时先定位，不能放宽 contract。
5. **Review**：检查 focused diff、非目标行为、重复验证、资源 owner 和 schema consumers。
6. **Accept**：`git diff --check`、最终 status、测试证据齐备后，由主 agent 更新 checkpoint。

旧实现上的 regression 已 PASS、源码已修复或已有测试与计划冲突时，立即停止该修复单元，
记录矛盾并重新 fact-check。不能伪造失败、强行改代码或直接将该 Phase 标为完成。
只有核清事实后才能重定范围；需要用户改变研究决策时才请求方向。

Phase 0 和 Phase 6C/7 的纯文档部分不制造失败测试：核对源码、审查文档 diff 即可。
已有通过结果可复用，但后续改动影响其依赖时须重跑。每 Phase 的相关回归以及 Final
的两仓库全量软件回归为必需检查，不因子 agent 模板中的节省测试原则而省略。

子 agent 交付必须包含：修改文件、调用链与 invariant、有效 red、green/related commands
及结果、未验证事项、需上交的决定，最后给出 `READY` 或有证据的 `BLOCKED`。
READY 只表示子任务可验收。

## 4. Phase 顺序与分工

| Phase | 主责建议 | 辅助/复核 | 进入下一步的证据 |
| --- | --- | --- | --- |
| Baseline | 主 agent | luna-max 可做只读清单 | 两仓库状态、HEAD、相关 dirty diff、纯软件基线 |
| 0 | luna-max 或主 agent | 主 agent 核对真实 collision path | 仅 `user_design.md` 变化，runtime/tests diff 为零 |
| 1：A → B → C | sol-high 定 contract；terra-xhigh 分段实现 | sol-high 复核 freeze 与 startup；luna-max 核查 schema/callsite | v13/v7 strict contract、mismatch tests、跨仓库 smoke |
| 2 | sol-high | terra-xhigh 可负责独立 metadata persistence | stationary/freshness/drift/rejection 和持久化 tests |
| 3A | sol-high | 主 agent 核对 supervisor 退出分类 | heartbeat-only 等待、parent stop、EXPLICIT_QUIT |
| 3B | terra-xhigh | sol-high 核查启动边界 | no-hand recording 在任何 worker/camera/recorder 启动前拒绝 |
| 3C | sol-high 定状态表并收紧边界 | luna-max 清单；terra-xhigh 显式 callsite 迁移 | callsites 先全绿，再收紧签名并回归 |
| 4 | terra-xhigh | 主 agent 核对两侧 freshness | 两个 min 路径、显式 override 拒绝、JOINT 无新增限制 |
| 5 | terra-xhigh | sol-high 复核 startup/storage 边界 | 原 provenance/reader 回归 |
| 6A/6B | terra-xhigh | sol-high 核查 6B spawn 前拒绝 | lag/tie tests、output occupancy 和 no-spawn tests |
| 6C | luna-max 或主 agent | 主 agent | hand logical target 与 arm published target 措辞正确 |
| 7（可选） | luna-max | 主 agent 限定 diff | 0–6 全过；只做已列明的低风险清理 |
| Final | 主 agent | sol-high 只读边界审查；luna-max 汇总证据 | 两仓库全量回归、逐项 acceptance、人工 checklist |

### Phase 0：固化 accepted design

仅在 `user_design.md` 中补充 learned-policy 无软件 realtime collision query、现有 safety
envelope，以及 teleop 保留 endpoint collision rejection、不执行逐 tick swept query 的决定。
先从实现确认这些描述，不更改 runtime、tests 或仍有效的其他设计。

### Phase 1：一个跨仓库 Phase，内部严格 A → B → C

原计划的 1A（point-cloud）与 1B（fingertip）作为两条验收线，按跨仓库落地顺序集成，
避免一边发布新 schema、一边添加临时旧版本兼容。

**A — Real dataset / FK / export：**

- 先补三种 profile regression，刻意提供错误 raw `hand_fingertip`；证明所有 profile 都
  从各自 processed `joint_state` 计算 fingertip，joint_state 只生成一次。
- fingertip 计算只使用 caller 提供的 FK、mapping、link 顺序和安装变换，不读取文件或持久化
  几何摘要。
- quaternion 先归一化再转 matrix；测试 q、-q、scaled q 表示同一旋转，并覆盖各项几何依赖。
- 在生成 fingertip 的同一次 processing 操作记录 derivation 与 policy ID，export 只传播这些值。
- processed v13 → Zarr v7：required attrs 严格校验，多 episode 三项不一致即拒绝。
  同时确认已有 point-cloud attrs 的完整性及 export 传播；不修改点云生成算法。
- 更新直接相关的 schema 文档和导航。A 的 targeted/related tests 通过后进入 B；
  跨 repo 集成状态仍是未完成，不声称整个 Phase 已通过。

**B — Policy deployment export：**

- 只接受新的 Real Zarr v7，验证 point-cloud 和启用的 fingertip semantic identity。
- 从 Zarr attrs 返回一次验证完成的 mapping，写入 `ObservationFieldSpec.semantics`；
  不从训练 config 推断缺失值，不二次读取/解析相同 attrs。
- point-cloud 包含 frame、position_units、color_order、color_source、policy_id、
  table_plane_abcd_json、sampling、transform。
- fingertip 包含 frame、units、finger_order、derivation、policy_id。
- deployment artifact 保持 v3；更新相应 export/contract fixtures 和 tests。

**C — Real startup compatibility：**

- 从 runtime.pointcloud、runtime.environment.table 和 canonical constants 构建 expected
  point-cloud semantics；table plane JSON 使用 `separators=(",", ":")`、`allow_nan=False`。
- 当前 runtime 的点云策略、桌面平面和 fingertip policy 常量构建 expected semantics。
- 按请求 modality 严格逐项比较，缺失/mismatch 在 motion 前拒绝，报错指出具体字段。
- 覆盖同 N 下 voxel/workspace/table 改动、policy ID、sampling/transform mismatch；
  覆盖 fingertip policy mismatch 与缺失 identity。
- 最后执行 Real Zarr v7 → Policy export → Real parse/compatibility 的纯软件 smoke。
  可通过临时目录在两种 Conda 环境间传递 artifact，不引入跨环境 runtime framework。

唯一允许的版本变化：raw v24 不变；processed v12→v13；Policy Zarr v6→v7；
deployment v3 不变。无 legacy reader、missing-field fallback 或 v6-or-v7 支持。
旧数据从 raw v24 重新 process/export。

### Phase 2：stationary calibration sample

复用 `runtime.arm.homing.velocity_convergence_rad_s` 和 `convergence_rad`：
fresh/healthy arm_before 与低 qvel → capture frames → fresh/healthy arm_after 与低 qvel
→ 小 qpos drift → append。任一失败只丢弃样本并打印原因，不进入全局 FAULT。
session 保持 ARMED，不新增状态、阈值、motion revoke 或跨设备插值。

测试 stationary、before/after 高速、drift、stale、marker missing，以及 rejection 不设置
FAULT。`cameras.json` 保存 diagnostic-only `calibration_capture`：分辨率/FPS、intrinsics、
distortion、method、sample_count、位置/旋转误差 mean/std/max、UTC 时间；测试 finite JSON
和其他 camera entries 保留。runtime 不从该 metadata 获取 intrinsics。

### Phase 3：lifecycle/publication

3A 镜像 PolicyExecutor clean-Q：停止产生命令并完成当前 recording decision，设置 quit，
只续 heartbeat，等 parent 清除 is_running 后返回；等待中不能发布动作或开始录制。
测试期间仍维持 e-stop/error/FAULT 和 supervisor fault/death 优先级，不改 supervisor。

3B 在 teleop experiment 最早阶段拒绝 recording_enabled 且无 hand；保留 no-hand debug，
不建立 arm-only dataset。用 fake/spy 证明拒绝发生在任何 worker 创建与 startup 之前。

3C 必须独立执行三个 gate：

1. `rg "publish_command\(" dexmani_real tests`，按真实控制流建立所有 production callsite
   的状态表，包括 keyboard，不凭模块名推断。典型：policy/grid/replay stream RUNNING；
   replay warm-up/calibration jog/hand homing ARMED，最终以源码为准。
2. 保持原签名，先将 production callsites 全部显式传 required_safety_state，回归通过。
3. 再将 publication 签名改为必需 keyword，补齐相关测试调用，验证正确/错误状态、遗漏参数
   拒绝以及全 callsite 覆盖。不收紧纯检查 API 的 optional 参数。

### Phase 4：visual freshness

visual processing 与 deployment pointcloud worker 都取 camera/policy age 上限的 min。
visual 显式 override 超过 policy 上限则 ValueError，不 silent clamp；JOINT 保持原逻辑。
测试 (0.25, 0.15)→0.15、(0.10, 0.15)→0.10、override 0.20>0.15 拒绝、JOINT 无新依赖。

### Phase 5：minimal provenance

实验 startup 不再采集或持久化配置/源码摘要；git 查询也不进入 realtime loop。
原有 raw reader 与逐行 provenance 合同保持不变，来源管理由调用方负责。

### Phase 6：deterministic replay

6A 用 `(rmse, abs(lag), lag)` 选择 lag；测试 constant identical→0、known delay、exact tie。
不新增 excitation threshold/confidence model。

6B 在 RuntimeChannels/worker 创建前检查 output：file 或 nonempty directory 拒绝；missing
或 empty directory 允许。测试拒绝时没有 spawn，错误指向另选 `--output`。
不增加 overwrite flag、timestamp naming、registry。

6C 文档明确 arm replay 是 recorded published arm-target；hand replay 是 recorded logical
hand-target，由当前 hand worker 生成当前 intermediate SDK setpoints，不称 exact actuator replay。

### Phase 7 与 Final

Phase 7 可跳过；只修已确认 stale module/path docstring 或删除明确 complete 的一次性
progress docs。diff 扩大即停止 cleanup，不改架构、AGENTS、有效 user_design 或本轮进度证据。

Final 两仓库分别执行 `python -m pytest -q`（使用各自环境）、`git diff --check`，
Real 执行 `python -m compileall -q dexmani_real examples`；其他静态检查仅使用仓库现有工具。
软件测试必须先确认硬件隔离，禁止运行 physical examples。缺依赖不能以 skip 伪装通过。
交付列出变化、red/green 证据、未验证项目和完整 acceptance，不以“tests passed”代替硬件结论。

人工 H1–H6 留待操作者另行明确启动：低速 teleop B/S/B/Q、正常监督下 collision 行为、
calibration stable/moving、artifact identity startup、stale pointcloud 日志、replay 重复 output。
不自动连接 xArm/XHand/RealSense，不通过制造碰撞验证 controller protection。

## 5. 不可越过的范围与停止条件

始终保持：policy 无 realtime software collision；teleop endpoint collision 保留且不加 swept
query；learned arm clip/count/final jump guard 不变；PolicySpec.requires_hand 拥有 learned
hand requirement，不新增 runtime.policy.hand_enabled deployment gate。

Deferred：read_latest stale fallback、RealSense depth/color clock、keyboard 架构、PolicyParams
拆分、config resolver、RuntimeChannels 全分配、环境锁定、`.claude/settings.local.json`。
只在 checkpoint 保留 follow-up，不顺手修复。

出现以下任一条件，停止当前单元并提交证据：需要修改超过两个不相关 domain；需要 generic
registry/manager；需要旧 schema fallback；需要改变 hardware safety、learned action shaping
或 PolicySpec action/modality 含义；测试与假设冲突；当前 HEAD 已修复问题。
Phase 1 已明确授权的 producer→artifact→consumer 同一语义链不因跨目录自动视为不相关 domain。
无法从源码和既有决策消除歧义时才向用户请求具体决定，不继续扩大 patch。

## 6. Checkpoint 与恢复

只有主 agent 更新 `repair_progress.md`。每个完成单元记录：baseline、文件 owner、invariant、
red failure、green/related 命令与结果、diff review、未验证事项和下一步。
长日志可放临时目录，checkpoint 保留足够复查的断言和结果，不能只写“通过”。
恢复任务时先核对 checkpoint 与当前 status/diff，复用未失效的证据，不重做已完成工作。

最终每个软件 acceptance 项报告 PASS/FAIL 并引用证据；未执行或环境受阻的项不能填 PASS，
应标 FAIL（未完成验证）并区别于已证实的代码缺陷。人工检查单独标未执行，不能声称硬件通过。

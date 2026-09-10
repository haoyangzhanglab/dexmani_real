# DexMani Real 多模态研究数据集最终化方案

> **Implementation status: SUPERSEDED.**
>
> Current implementation is the v18/v11 multimodal dataset. Remaining
> last-mile work is tracked in:
> [`multimodal_dataset_last_mile_repair_plan.md`](multimodal_dataset_last_mile_repair_plan.md)

> 面向 Claude Code 的 canonical implementation plan。
>
> 目标：在当前 control-step 数据架构基础上，完成最后一轮**语义修正、模态补全和冗余清理**，形成适合个人 PhD 研究长期复用的数据管线。本文不把数据集裁成当前 `dexmani_policy` 某个策略的最小输入，而是构建一次采集、一次处理、后续不同策略可按需选择模态的 multimodal research dataset。
>
> 本文基于 2026-09-10 review 的最新主线：`dexmani_real` raw v28 / processed v17 / Policy Zarr v10，以及 `dexmani_policy` 当前 `horizon=16, n_obs_steps=2, n_action_steps=8, pad_before=1, pad_after=7` 训练设置。执行时若源码已经前移，以**当前源码为 authority**，不要机械套用版本号。

---

## 0. Executive decision

本轮最终设计：

```text
RAW
    = source-of-truth sensor/action archive

PROCESSED HDF5
    = row-preserving, synchronized-at-control-step, reusable multimodal research dataset

POLICY ZARR
    = 与 processed 同一套 multimodal superset，便于不同 Policy 直接选择需要的 field

DEXMANI_POLICY
    = sensor_modalities 选择子集
    = temporal sampler 负责 episode boundary padding
    = model-specific resize / crop / augmentation 留在 Policy 侧
```

Canonical timeline 仍然是 control step：

```text
T[t] = one logical control-step observation anchor

for each modality m:
    source_m[t] <= T[t]

action[t]
    = command associated with this control step
```

不恢复 camera-master synchronization，不做跨 raw row repair，不做 normal row deletion。

### 三条最重要的新决定

1. **删除 `full_history` 数据集合同。** Episode start padding 是 Policy sampling semantics，不是 dataset storage semantics。训练继续 `pad_before=n_obs_steps-1`；真机 inference 在 run-start history 不足时重复第一条本 run 内合法 observation。
2. **Processed HDF5 和 Policy Zarr 必须重新包含 dense tactile，并保留 RGB/depth/camera geometry/contact/fingertip 等当前或未来可能使用的模态。** Dataset 是 reusable superset，不由当前 encoder consumer 决定保存内容。
3. **减少 exact schema-version hard gate。** Schema version 可作为 metadata / parser discriminator，但不要在 Real→Policy→Deployment 多层反复要求 `schema_version == X`。消费者只验证自己真正依赖的 keys / shape / dtype / units / frame / action semantics。

---

# 1. Project positioning

这是个人 PhD 具身智能研究代码，目标不是生产级 dataset platform。

执行原则：

```text
correctness > simplicity > extensibility
```

但这里的 simplicity 指：

```text
少 abstraction
少 duplicated validation
少 implicit repair
少 consumer-specific dataset variants
```

而不是删除有研究价值的原始/派生模态。

### 保留的信息优先级

应该保留：

```text
昂贵且未来无法重建的 sensor information
低成本且能解释时间/validity 的 metadata
当前常用的 deterministic derived representations
```

应该删除：

```text
旧 camera-master path
row-selection / gap-repair machinery
只为 schema-version 层层认证存在的 gate
consumer-specific modality pruning
已经恒定、没有信息量的 legacy state
```

禁止新增：

```text
SensorManager
ModalityRegistry
DatasetBackend hierarchy
Schema migration graph
Compatibility framework
Generic data-quality service
```

如一个明确 helper 可以解决问题，就不要建立 framework。

---

# 2. Reference projects: what to copy, what not to copy

## 2.1 LeRobot

参考 latest reviewed commit `71a11efe77f55e61f3ab2ce45b40da8cf626afa9`。

LeRobot 的 dataset temporal query 在 episode 边界会把 query index clamp 到 episode start/end，并额外提供 padding mask。也就是说，历史不足时重复 edge frame 是正常 dataset sampling semantics，不要求源 episode 真实拥有负时间历史。

LeRobot 也把 dataset 本身作为 feature-rich storage：数据包含多个 observation/action features，policy 根据配置和 processor 选择需要的输入，而不是反向要求 dataset 只保存当前 model consumer。

**借鉴：**

```text
edge-repeat temporal padding
feature-rich dataset
model preprocessing belongs to Policy processor
```

**不照搬：**

```text
Hub/version framework
复杂 dataset backend
远端 streaming
```

## 2.2 ManiUniCon

参考 latest reviewed commit `85c6f2e32ecf9f2bed62d202b058c39623444686`。

ManiUniCon realtime `SharedMemoryRingBuffer.get_last_k(k)` 在 ring history 不足时，直接用 oldest available observation 补齐前缀。Policy runtime 调 `read_state(k=obs_horizon)` / camera history，因此 run start naturally supports repeated earliest observation。

其 demo processing 会同时保留 robot state、RGB、depth、camera intrinsics/transforms 等 observation，再写入 ReplayBuffer，而不是只为一个模型导出最小 field set。

**借鉴：**

```text
warm-start edge-repeat
multimodal reusable replay dataset
simple episode-first processing
```

**不照搬：**

```text
n_steps=min(...) 掩盖真实结构错误
缺少 DexMani 已有的 source timestamps / exact action provenance
```

---

# 3. Final data architecture

## 3.1 Raw

Raw 保持 sensor/source facts，不做 policy-specific projection。

继续保存：

```text
arm_qpos / arm_qvel / arm_tau
hand_qpos / hand_current
hand_contact
hand_tactile_force

action_arm_joint_sent
action_hand_joint
action_arm_ee

RGB-D source payload
camera calibration / geometry

observation_anchor_monotonic_ns
arm_source_monotonic_ns
hand_source_monotonic_ns
hand_contact_source_monotonic_ns
tactile_source_monotonic_ns
camera_source_monotonic_ns
VR source timestamps

sensor validity / freshness telemetry where already cheaply available
frame_status / tracking telemetry
```

不恢复：

```text
policy_observation_*
camera-aligned state/tactile copy
cross-row tactile selection
```

## 3.2 Processed HDF5

Canonical processed artifact 不再有 JOINT/RGB/PC omission semantics。标准 workflow 一次生成完整 multimodal episode。

Target arrays：

```text
joint_state             float32 [N,19]
action                  float32 [N,19]
action_ee               float32 [N,21]

rgb                     uint8   [N,H,W,3]
depth                   uint16  [N,H,W]
camera_intrinsic        float32 [N,3,3]  # 或现有 episode-broadcast representation
camera_extrinsic        float32 [N,4,4]  # T_xarm_base_from_color

point_cloud             float32 [N,P,6]

contact_force           float32 [N,5,3]
contact_force_valid     bool    [N]

tactile_force           float32 [N,5,120,3]
tactile_force_valid     bool    [N]

fingertip_points        float32 [N,5,3]
```

建议额外保留便宜且直接有研究价值的 timing arrays：

```text
observation_anchor_monotonic_ns      uint64 [N]
arm_source_monotonic_ns              uint64 [N]
hand_source_monotonic_ns             uint64 [N]
contact_source_monotonic_ns          uint64 [N]
tactile_source_monotonic_ns          uint64 [N]
camera_source_monotonic_ns           uint64 [N]
```

不要恢复旧 `/provenance` framework。它们只是 flat datasets，不做 source-row mapping，因为 row identity 永远是 1:1。

## 3.3 Policy Zarr

Policy Zarr 与 processed 使用同一 multimodal superset，方便后续 Policy：

```text
DP / VLA            -> joint_state + rgb
DP3 / R3D / SAT     -> joint_state + point_cloud
future tactile      -> tactile_force + tactile_force_valid
future visuotactile -> rgb/point_cloud + contact/tactile
geometry policy     -> depth + K + extrinsic + state
```

`dexmani_policy` 的 `sensor_modalities` 只决定加载哪些字段，不决定 Real export 保存哪些字段。

---

# 4. Episode temporal semantics: remove full_history

## 4.1 Dataset storage

Dataset 只保存真实 rows：

```text
row0 row1 row2 ...
```

不在 HDF5/Zarr 中复制 padding rows。

删除作为硬合同的：

```text
episode_start_policy="full_history"
```

不要替换成另一个必须跨层验证的 schema attribute。

## 4.2 Training

保持当前 Diffusion Policy-style sampling：

```text
horizon = 16
n_obs_steps = 2
n_action_steps = 8
pad_before = n_obs_steps - 1 = 1
pad_after = n_action_steps - 1 = 7
```

Episode 开头：

```text
real rows:      o0, o1, o2, ...
first sample:  [o0, o0]
second sample: [o0, o1]
```

这不是 fake data；它是 boundary condition，且机器人在启动阶段处于静止/hold。

## 4.3 Deployment warm start

Deployment 必须 mirror train semantics。

要求：

```text
只使用 run_started_ns 之后的合法 observation
history 不足时重复本 run 内 oldest valid complete observation
```

对于 `n_obs_steps=2`：

```text
first complete observation O0:
    history = [O0, O0]

next logical step O1:
    history = [O0, O1]
```

对于任意 `n_obs_steps=k`：

```text
available complete observations = [O0, ..., Oj], j+1 < k
history = [O0 repeated k-(j+1) times, O0, ..., Oj]
```

### Important

不要：

```text
读取 pre-run stale history 作为 padding
伪造 source timestamps
复制某个 modality 而其他 modality 不一致
```

Warm-start padding 的基本单位是**完整 Policy observation step**，不是分别给各 sensor 随机补历史。

实现优先用一个窄 helper / deque；不要建立新的 temporal manager。

---

# 5. Camera history: allow healthy reuse

当前 deployment visual selector 如果要求相邻 logical control steps 必须使用 strictly advancing camera sequence/source，会与 edge-repeat / asynchronous camera semantics 冲突。

目标：

```text
at each logical policy time T[t]:
    choose newest healthy causal visual frame <= T[t]
```

允许：

```text
T0 -> camera C10
T1 -> camera C10
T2 -> camera C12
```

只要：

```text
camera generation unchanged
health usable
source causal
age <= configured live age / grid-lag threshold
payload valid
```

不要为了“每个 policy step 必须有新相机帧”阻止 inference。

### Recording

Recording 同样应优先保留 recent healthy causal camera；如果最新 camera event 不健康但最近 healthy frame 仍在允许 age 内，可以 hold-last healthy visual rather than writing a bad visual or discarding a row。

Sustained camera stall 超过现有 runtime abort threshold 仍可 discard recording episode；不要削弱这个明显的 hardware/data failure gate。

---

# 6. Dense tactile: restore to processed and Zarr

这是本轮 mandatory change。

## 6.1 Raw semantics

保留当前 raw：

```text
hand_tactile_force     [N,5,120,3]
tactile_source_monotonic_ns
tactile_fresh
tactile_calibrated
tactile_unit_code
```

Invalid dense payload 不能用全零冒充 no-contact。

## 6.2 Processed mapping

```text
tactile_force[t]
    <- raw hand_tactile_force[t]

tactile_force_valid[t]
    <- whether raw dense payload is a real usable measurement
```

原则：

```text
valid=True  -> payload finite, source causal, semantics/unit/calibration usable
valid=False -> payload may contain NaN; row remains in episode
```

不要：

```text
previous-row fill
future interpolation
cross-row repair
whole episode reject because one dense tactile row is invalid
```

## 6.3 Fresh vs valid

不要把 `fresh` 和 `valid` 混成一个概念。

推荐语义：

```text
valid = measurement integrity / usable payload
fresh = age relative to recording control step under current runtime threshold
```

如果 raw 已有 `tactile_fresh`，可以一起复制到 processed/Zarr 作为 bool telemetry；不要再造更复杂状态机。

## 6.4 Future Policy usage

当前 `dexmani_policy` 不必立刻实现 dense tactile encoder。

未来 tactile policy 必须显式请求：

```text
tactile_force
tactile_force_valid
```

具体 invalid-row handling（mask / zero after mask / temporal hold）属于 Policy design，不属于 Real preprocessing。

---

# 7. Aggregate contact: preserve modality, decouple admission

不要删除 `contact_force`。

Target：

```text
contact_force          float32 [N,5,3]
contact_force_valid    bool    [N]
contact_source_monotonic_ns uint64 [N]
```

当前 aggregate contact 与 dense tactile 是两个独立 measurement contracts：

```text
aggregate valid, dense invalid
    -> contact_force_valid=True
    -> tactile_force_valid=False
```

以及反过来不能自动假设等价。

### Episode admission

单个 contact invalid row：

```text
KEEP episode
KEEP row
mark contact_force_valid=False
```

不要再因为 `hand_contact` 存在 NaN 而把整个 episode 当 technical corruption。

只有 required structural/action/visual corruption 或 persistent behavioral failure 才 whole-episode reject。

---

# 8. Recording BEGIN gate simplification

当前 recording BEGIN 若仍强制 dense tactile：

```text
exists
fresh
calibrated
within age
```

则与新的 optional-validity semantics 冲突。

删除 dense tactile 作为 recording admission hard gate。

BEGIN 继续要求真正与 motion safety / demonstration execution 有关的：

```text
arm feedback usable
hand qpos feedback usable when hand enabled
VR/control input usable
runtime/safety state valid
```

Tactile：

```text
available -> record it
invalid/unavailable -> record invalid mask + telemetry
```

可以 warning，但不要阻止 demo 开始。

---

# 9. Visual representation: make processed reusable

## 9.1 RGB/depth

当前 processed 若固定 resize 到 `240x320`，会把今天的 policy preprocessing 烘焙进 dataset。

最终 canonical dataset 建议保存：

```text
aligned RGB at source/native color resolution
aligned depth at source/native color resolution
corresponding aligned color intrinsics
T_xarm_base_from_color
```

不要在 Real processed 中为 DINO/CLIP/SigLIP/ResNet/VLA 提前 resize/crop。

模型侧已有 preprocessing，例如 RGB Policy 当前可：

```text
native/aligned RGB
-> rgb_preprocess_size
-> random/center crop
-> ImageProcessor
```

这应继续属于 `dexmani_policy`。

### Migration scope

这会增加 processed/Zarr 体积，但对于个人 PhD real-robot dataset 数量级是合理 trade-off；换来 future Policy 不必从 raw MP4/depth 重做基础数据处理。

## 9.2 Point cloud

`point_cloud` 是 derived cache，不可能 future-proof 所有 3D research。

本轮保留一个 canonical point cloud：

```text
point_cloud [N,P,6]
```

继续使用当前共享 `build_point_cloud*` pipeline 和明确 config。

同时因为 processed/Zarr 保存 native RGB-D + geometry，未来若研究需要不同：

```text
P
workspace crop
sampling
voxel
outlier removal
features
```

仍可从 canonical visual source derivation，而不必回到原 recorder internals。

不要为了 future-proof 同时保存多个 P 的 point cloud。

---

# 10. OutputProfile simplification

当前 `OutputProfile = JOINT / RGB / POINTCLOUD / RGB_PC` 会让同一 raw episode 产生多套不同 processed artifacts。

这与“一次处理、不同策略复用”冲突。

Target canonical workflow：

```text
one raw episode
    -> one multimodal processed episode
    -> one multimodal Zarr task store
```

因此：

1. `process_episodes.py` 默认且 canonical 路径始终生成 multimodal superset。
2. 删除或停止对外暴露 `--profile` modality omission 选项。
3. `OutputProfile` 若删除后只剩内部旧测试使用，直接删除。
4. 不保留 compatibility aliases。

如果 pointcloud 生成成本确实需要临时 skip，只允许保留一个非常明确的 development/debug flag，不让它成为正式 dataset schema variant。优先完全删除 profile。

---

# 11. Action semantics and short-IK corner case

## 11.1 Joint action

继续：

```text
action[t] = action_arm_joint_sent[t] + action_hand_joint[t]
```

`action_arm_joint_sent` 是发布到 coupled command / arm worker 的 target；不要宣称它是 physical achieved qpos。

## 11.2 EE action

正常 solved frame：

```text
action_arm_ee = actual target used by IK
```

## 11.3 Short IK hold bug

若 recording 开始后第一帧发生 short `FRAME_IK_FAIL`，`controller.last_target_eef_*` 可能尚未初始化为 finite，而当前设计又允许 1–4 帧 IK fail 保留。

因此 held frame 的 EE representation 必须定义为 finite、与实际 hold joint command 一致的量。

推荐：

```text
held action_arm_ee
    = FK(actual published held arm_qpos)
```

这样：

```text
joint action = hold(q)
EE action    = FK(hold(q))
```

禁止通过 NaN 再让 processing 把 short IK episode whole-reject。

Persistent IK threshold 保持当前 established boundary，不在本轮重新调参。

---

# 12. Schema/version simplification

用户明确不希望跨仓库反复 version certification。

新的规则：

## 12.1 Keep version as metadata

Raw/processed/Zarr 可继续有：

```text
schema_name
schema_version
```

作用：

```text
人类追踪
历史 parser dispatch
实验 provenance
```

## 12.2 Remove exact-version gates from consumers

`dexmani_policy` 不应因为：

```text
schema_version != exact current integer
```

就拒绝一个 keys/semantics 完全兼容的 dataset。

Consumer 只检查自己真正需要的：

```text
requested key exists
shape/dtype correct
action dimensionality correct
units/frame required when physically meaningful
point cloud geometry semantics correct when needed
```

不要在：

```text
ReplayBuffer
BaseDataset
training
checkpoint
Policy export
deployment restore
```

层层重复同一 schema version validation。

## 12.3 Historical exception

Raw historical reader 仍可：

```text
if current raw schema
elif normalized legacy v26
else unsupported
```

因为这是实际 layout dispatch，不属于多余 contract checking。

## 12.4 One final metadata bump

因为本轮 processed/Zarr 会增加 dense tactile/valid masks，并可能改变 RGB/depth stored resolution，建议为了 research traceability **各 bump 一次 metadata version**：

```text
processed: current + 1
Policy Zarr: current + 1
```

Raw layout若不变，不 bump raw。

这是一次 human-facing schema marker，不允许再派生出多层 `==version` gate。

---

# 13. Policy-side dataset behavior

`dexmani_policy` 的 `ReplayBuffer`/Dataset 应继续支持：

```text
load only requested keys
```

例如：

```yaml
sensor_modalities: [joint_state, point_cloud]
```

不应要求 Zarr 只有这些 keys。

### Do not add

```text
Zarr keys == encoder consumed fields
full dataset schema == current Policy config
```

这样的 restriction。

可以在 model build 时仅验证：

```text
requested sensor modalities are actually consumed by this model
```

如果这个检查已经存在于 export/restore，可以不再扩大到 training；用户优先研究效率，不需要为了防每种 config typo 加更多 framework。

### Real→Policy path

不实现自动 copy/symlink/registry。用户手动管理 Zarr path。

---

# 14. Processed/Zarr validation philosophy

本项目只需要三层验证：

## A. Producer/write boundary

验证：

```text
shape
dtype
row count
required action finite
media decode/readability
basic source causality
```

## B. Derived modality boundary

例如 point cloud：

```text
shape
finite
frame=xarm_base
xyz/rgb convention
```

Dense/contact：

```text
valid mask <-> finite payload consistency
```

## C. Consumer boundary

Policy 只验证它真正请求的 fields。

删除同一 semantic contract 在 exporter、training、artifact restore 中重复 exact checking。

---

# 15. Episode admission policy

保持 row-preserving / whole-episode architecture。

## Whole episode reject

```text
persistent IK failure
required action structurally corrupt / NaN
required robot state structure corrupt
RGB/depth source genuinely unreadable for canonical multimodal dataset
frame count / identity irrecoverably corrupt
explicit user exclude
```

## KEEP + modality-validity mask / telemetry

```text
dense tactile unavailable
aggregate contact unavailable
isolated tactile stale
isolated camera reuse
camera timing jitter
short IK hold
tracking transient
```

### Camera sustained failure

Runtime recording stall abort 属于 source acquisition failure，可以整条 discard，继续保留。

---

# 16. `pick_place_toy` legacy data

已经完成的 61 条 salvage 不重新解释、不修改原 raw。

Canonical evidence：

```text
61 source episodes
60 accepted
1 rejected: episode_20260827_224527 persistent IK
14309 source frames
14112 retained frames
accepted_episode_lost_rows = false
```

本轮新增 dense tactile 到 processed/Zarr 时，应从已经 scale-normalized legacy v26 working copies重新构建 multimodal processed/Zarr；不要再改原 v25 raw。

### Required check

重新构建后仍应：

```text
60 accepted
1 persistent IK rejected
14112 rows
```

Dense tactile invalid rows用 validity mask 表达，不允许因此新增 rejected episode。

---

# 17. Redundant/deprecated mechanisms to remove

执行前先 `rg` caller graph。确认无 live independent consumer 后删除：

```text
episode_start_policy="full_history" hard contract
Policy exact Real Zarr v10 gate and equivalent version-only tests
profile-based modality omission in canonical processing/export
processed-only fixed RGB resize config if no longer used
camera strict-advance requirement when age/health still valid
recording BEGIN dense tactile hard gate
contact/dense finite-all-rows admission gate
obsolete docs claiming v27/v16/v9/v8 current
```

下次 raw schema自然 bump 时可删除、但本轮不为它们单独 bump：

```text
fill_reason          # if always SOURCE
flag_sample_valid    # if always True
observation_valid    # if no active consumer
```

不要为清 3 个 scalar 单独制造 raw v29。

---

# 18. Phase-by-phase implementation plan

## Phase 0 — Source inventory

Claude Code 先读取：

```text
AGENTS.md
CLAUDE.md
code_style.md
repo_map.md
docs/data_schema.md
this document
```

然后在 Real + Policy 两仓库搜索：

```bash
rg -n "episode_start_policy|full_history" .
rg -n "OutputProfile|--profile" dexmani_real examples tests docs
rg -n "tactile_force|hand_tactile_force|tactile_fresh" dexmani_real tests docs
rg -n "contact_force|hand_contact" dexmani_real tests docs
rg -n "schema_version.*10|schema_version.*17|schema_version.*28" .
rg -n "_select_camera_control_grid|previous_sequence|previous_source" dexmani_real tests
rg -n "_begin_feedback_issue|RECORDING_TACTILE_MAX_AGE_NS" dexmani_real tests
rg -n "last_target_eef|FRAME_IK_FAIL" dexmani_real tests
```

Policy：

```bash
rg -n "pad_before|pad_after|SequenceSampler" dexmani_policy
rg -n "schema_version|episode_start_policy|full_history" dexmani_policy tests docs
```

输出一份 concise caller inventory，不做 framework。

---

## Phase 1 — Fix temporal start semantics first

修改：

```text
Real Zarr export metadata
Policy export/parser exact full_history requirement
Real deployment warm-start observation builder
camera control-grid visual selection
```

目标：

```text
training keeps pad_before=n_obs_steps-1
Zarr has no full_history hard contract
deployment repeats first valid complete post-run observation
visual frames may repeat when still healthy/causal/recent
```

先用 synthetic unit tests，不跑硬件。

---

## Phase 2 — Restore tactile/contact modality completeness

修改 raw→processed mapping 与 processed schema：

```text
add tactile_force
add tactile_force_valid
add contact_force_valid
add source/timing arrays selected in §3.2
```

调整 validation：

```text
invalid mask does not reject episode
valid=True requires finite payload
valid=False payload may be NaN
```

然后修改 Zarr exporter为完整复制 multimodal fields。

---

## Phase 3 — Remove profile omission and fixed visual preprocessing

将 canonical `process_episodes.py` 改为一次生成完整 multimodal artifact。

优先删除：

```text
--profile
OutputProfile branching
needs_rgb / needs_pointcloud modality omission
```

RGB/depth 保存 source/native aligned resolution。

Policy-side existing resize/crop remains。

Point cloud继续生成 canonical cache。

---

## Phase 4 — Recording admission cleanup

删除 BEGIN dense tactile hard gate。

确认：

```text
recording can start with valid robot/VR even if dense tactile temporarily missing
raw row stores tactile invalid state honestly
no fake zero tactile
```

Camera recording使用 newest healthy causal frame / recent healthy reuse；sustained stall abort不变。

---

## Phase 5 — Fix short-IK `action_ee`

让 first/early held IK frame 的 EE action始终 finite：

```text
FK(actual held arm command)
```

保留 persistent IK whole-episode rejection。

---

## Phase 6 — Historical rebuild gate

对 `pick_place_toy`：

```text
normalized legacy v26
-> new multimodal processed
-> new multimodal Zarr
```

Hard acceptance：

```text
60 accepted
1 persistent IK rejected
14112 retained rows
no row loss
no tactile/contact timing-only rejection
```

不修改原 raw。

---

## Phase 7 — Delete deprecated code/docs/tests

只有前面全部通过后删除：

```text
full_history-specific tests/docs
profile omission tests
strict camera-sequence advancement tests
version-only Policy v10 rejection tests
obsolete current-schema references
superseded design/evidence docs that now mislead current behavior
```

历史 incident 只保留必要事实，不保留第二套 canonical architecture。

---

# 19. Required tests

## T1 — Training edge padding

`n_obs_steps=2, pad_before=1`：

```text
episode [o0,o1,o2]
first sample obs == [o0,o0]
second == [o0,o1]
```

## T2 — Deployment warm start

只有一条 post-run complete observation：

```text
runtime observation history == [O0,O0]
```

禁止使用 run-start 之前的 frame。

## T3 — Visual causal reuse

```text
C0 healthy, recent
T0 -> C0
T1 -> C0
```

应该成功。

Future/stale/unhealthy/generation-mismatch仍失败。

## T4 — Dense tactile valid

```text
tactile_force_valid=True
payload finite
source causal
```

processed/Zarr同行一致。

## T5 — Dense tactile invalid

```text
tactile_force_valid=False
payload NaN
```

整条 episode仍 accepted。

## T6 — Contact independent

```text
contact valid, dense invalid
```

期待：

```text
contact_force_valid=True
tactile_force_valid=False
```

反向独立 case 也测试。

## T7 — Recording BEGIN

Dense tactile absent/invalid，但 robot/VR正常：

```text
recording BEGIN allowed
```

## T8 — Short IK at episode start

first row `FRAME_IK_FAIL`：

```text
action finite
action_ee finite
whole episode accepted if run <=4
```

## T9 — Persistent IK

连续 >=5：

```text
whole episode rejected
```

## T10 — Full multimodal HDF5/Zarr roundtrip

验证所有 arrays：

```text
row count identical
shape/dtype correct
selected source-derived fields equal after specified casts/transforms
```

## T11 — Legacy salvage

```text
60 / 61
14112 frames
```

Dense/contact invalid不增加 rejection。

## T12 — Policy subset loading

同一个 multimodal Zarr：

```text
[joint_state, rgb]
[joint_state, point_cloud]
[joint_state, tactile_force, tactile_force_valid]
```

均可按 requested keys 读取，而不要求 Zarr key set 等于 Policy modalities。

---

# 20. Validation philosophy for Claude Code

每阶段先 focused tests，再 full offline suite。

允许：

```text
unit tests
synthetic shared-memory tests
offline HDF5/Zarr rebuild
legacy read-only processing
```

禁止自动运行：

```text
robot motion
teleop hardware
RealSense live capture
hand hardware commands
live Policy rollout
```

最终至少：

```bash
python -m compileall -q dexmani_real examples tools
pytest <focused tests>
pytest

git diff --check
git status --short
```

Policy repo对应做 compile/focused/full tests，按其 AGENTS.md 环境执行。

---

# 21. Simplicity constraints during implementation

Claude Code 每个 phase结束时检查：

```text
Did I add a second way to represent the same thing?
Did I add a generic abstraction for one historical dataset?
Did I add a new schema/version gate where shape/semantic validation is enough?
Did I make dataset contents depend on the current Policy consumer?
Did I add data repair instead of validity metadata?
```

任意答案为 yes，优先再简化。

### Preferred implementation style

```text
flat arrays > nested provenance framework
explicit fields > registry
one canonical processed artifact > profiles
a validity mask > row deletion
edge repeat > synthetic pre-run data
source preservation > policy-specific preprocessing
```

---

# 22. Commit strategy

建议 coherent commits：

```text
1. refactor: align episode warm-start sampling across training and real inference
2. feat: restore dense tactile and modality validity to processed datasets
3. refactor: make processed and zarr artifacts reusable multimodal datasets
4. fix: decouple recording admission from tactile and permit healthy visual reuse
5. fix: make held action-ee finite for short ik failures
6. data: verify legacy pick-place-toy multimodal rebuild
7. cleanup: remove obsolete profile/version/full-history mechanisms
8. docs: finalize multimodal research dataset contract
```

不要 commit generated large HDF5/Zarr。

---

# 23. Final acceptance criteria

## Dataset semantics

- [ ] Control step remains canonical row timeline.
- [ ] Camera is not master clock for robot/tactile state.
- [ ] No normal raw row deletion/compaction/repair exists.
- [ ] Dataset does not require full real history before episode row0.
- [ ] Training and deployment both use edge-repeat warm start.

## Multimodal completeness

- [ ] Processed HDF5 contains joint/action/RGB/depth/camera geometry/point cloud/contact/dense tactile/fingertip.
- [ ] Policy Zarr contains the same reusable multimodal fields.
- [ ] Dense tactile has an explicit validity mask.
- [ ] Aggregate contact has an explicit validity mask.
- [ ] Invalid tactile/contact rows do not reject an otherwise usable episode.
- [ ] Raw dense tactile is never fake-zeroed into valid no-contact data.

## Reuse

- [ ] Standard processing produces one canonical multimodal artifact, not modality-specific profiles.
- [ ] Policy selects the subset it needs at load time.
- [ ] RGB/depth stored representation is not tied to one Policy backbone's crop/resize.
- [ ] Point cloud remains a cached derived modality while RGB-D+geometry allow future alternative 3D derivation.

## Runtime

- [ ] Recording BEGIN does not hard-require dense tactile.
- [ ] Healthy recent camera frames may be reused across adjacent Policy control steps.
- [ ] Future/stale/unhealthy camera data remains rejected live.
- [ ] Existing robot safety, generation, causality and age gates are not weakened.
- [ ] Short IK held frames have finite consistent `action_ee`.
- [ ] Persistent IK remains whole-episode reject.

## Versioning / simplicity

- [ ] No new schema registry/migration framework added.
- [ ] Exact version hard gates removed where physical/key/shape semantics are sufficient.
- [ ] Historical raw parser dispatch remains explicit and narrow.
- [ ] Obsolete full-history/profile/current-version docs/tests removed.

## Legacy data

- [ ] Original `pick_place_toy` raw files unmodified.
- [ ] 60 accepted / 1 persistent-IK rejected.
- [ ] 14112 accepted source rows retained if baseline unchanged.
- [ ] Dense/contact validity does not create additional rejection.

---

# 24. Final Claude Code report

最终报告必须给出：

```text
Real baseline HEAD / final HEAD
Policy baseline HEAD / final HEAD
processed/Zarr metadata schema versions
```

并列出：

```text
added multimodal fields
deleted mechanisms
temporal warm-start semantics
camera reuse semantics
recording BEGIN changes
short-IK action_ee fix
```

Historical data actual：

```text
source episodes
accepted
rejected + exact reason
source frames
retained frames
Zarr episodes
```

明确回答：

```text
Did an accepted episode lose any source row?       NO
Did legacy raw get modified?                       NO
Can invalid dense tactile alone reject an episode? NO
Can current Policy consumer remove stored fields?  NO
Does deployment use pre-run history for padding?   NO
Was robot runtime safety weakened?                 NO
```

Hardware validation若未授权必须写：

```text
Hardware validation: NOT RUN
```

---

# 25. Context recovery summary

DexMani Real 的最终目标是一个个人 PhD research-oriented reusable multimodal dataset。Control step 是 canonical row timeline；camera、robot state、contact、dense tactile 可以异步，只需 causal to the step。Raw 保存 source facts；processed HDF5 和 Policy Zarr 一次性保存 joint/action、RGB-D、camera geometry、point cloud、aggregate contact、dense tactile、fingertip 及必要 validity/timing，后续 Policy 只选择需要的 fields。不要根据当前 DP/DP3 consumer 删除 future-use modalities。Episode storage 不要求 `full_history`；training 保留 `pad_before=n_obs_steps-1`，deployment 在 run start 重复第一条 post-start complete valid observation，禁止使用 pre-run stale history。允许健康且仍新鲜的 camera frame 在相邻 policy ticks reuse。Dense/contact invalid 用 validity mask 表达，不做跨行 repair，也不因此 whole-episode reject。Recording BEGIN 不再由 dense tactile hard gate。Short IK held action_ee 用 actual held qpos 的 FK 保持 finite；persistent IK 仍 whole-episode reject。Schema version 只作 metadata/history parser discriminator，不在 Real→Policy→Deployment 反复 exact-version certification。
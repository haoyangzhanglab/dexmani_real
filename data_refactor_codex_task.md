# DexMani Real 数据层重构与 `pick_place_toy` 迁移任务

> 本文件是 Codex CLI 的实现任务规范。  
> 目标不是继续给现有 v33 打补丁，而是一次性收敛 Raw Episode 的研究语义、简化训练导出校验，并在 **不修改 `dexmani_policy`** 的前提下，使历史 `pick_place_toy` 数据可以被诚实、可审计地迁移和继续使用。

## 0. 基线、范围与硬约束

本任务编写时仓库最新提交为：

```text
0a75572ea1a70481063ef0385bb8c0b3a89b4f4c
```

实施前必须先记录实际 HEAD；若代码已继续变化，以实际 HEAD 为准重新核对调用链，不得机械套用本文行号或旧实现细节。

当前已知：

- 当前 Raw schema：v33。
- 当前 canonical Policy Zarr schema：v15。
- `episodes/pick_place_toy`：61 个 legacy raw v30 episode，共 14,309 帧。
- legacy v30 与 current v33 的详细审计见根目录 `PICK_PLACE_TOY_MIGRATION.md`。
- 当前相邻 `dexmani_policy` 已经消费 Policy Zarr v15，并对 action / observation / point cloud / tactile 等语义有现成 contract。

### 0.1 三条不可更改的产品/研究原则

1. **Raw episode 中任一影响 canonical 多模态训练正确性的坏帧，导出 Zarr 时整段 episode 丢弃。**
   - 不删除坏行。
   - 不拆 training segment。
   - 不把同一个 physical episode 的前后片段重新拼入 Zarr。
   - 不插值、不 forward-fill、不重采样。

2. **Policy Zarr 是通用的完整多模态 canonical dataset。**
   - 不按某个模型当前输入裁剪 Zarr。
   - 继续输出完整 canonical modalities。
   - 模型自己在 `dexmani_policy` 加载时选择所需模态。

3. **本任务不得修改 `dexmani_policy` 仓库。**
   - 所有兼容性工作都在 `dexmani_real` 完成。
   - 最终 Zarr 必须继续通过当前 `dexmani_policy` 的公开 contract / dataset loader。

### 0.2 绝对禁止

- 不原地修改 legacy `episodes/pick_place_toy`。
- 不把 v30 直接改版本号伪装成 v34。
- 不伪造历史不存在的 `observation_timestamp_ns` / `action_timestamp_ns`。
- 不用 source timestamp 或固定延迟推测 action publish timestamp。
- 不把旧 `action_arm_ee` 当作 canonical `action_ee`。
- 不把旧 status 数字按 v33 enum 解释。
- 不把当前 runtime calibration / table plane / hand mount 未经核实地套到历史数据。
- 不给正常 `EpisodeReader` 增加 v30/v31/... 兼容分支。
- 不修改真机安全边界、命令发布顺序、硬件 shutdown 逻辑。
- 测试和迁移验收全部离线，不连接或驱动机器人。

### 0.3 与当前 `AGENTS.md` 的已知设计冲突

当前 `AGENTS.md` 的 Research data 段仍要求 Raw 持久化 raw VR、真实 timestamps、显式 tactile validity 和较多诊断；这与本任务经过 LeRobot / Diffusion Policy 对照后确定的 v34 目标冲突。

本任务是对 **Research data 持久化 contract 的显式设计变更授权**，但不是对 hardware/runtime safety 的放宽授权。实施时：

- 先按本文件完成代码事实核对；
- 只修改与新 v34 data contract 冲突的 standing instructions；
- 在代码已经实现并验证后，同一 change 中同步更新 `AGENTS.md` 的 Research data 描述和 README；
- 保留 `AGENTS.md` 中所有 hardware safety、freshness、causal observation、run_id、shutdown 等要求；
- “sensor ring 不发布 generic validity flag”的运行时原则继续成立；Raw 的 `frame_valid` 是 control-row 研究摘要，不是 sensor validity channel；
- 仓库明确“不维护 committed tests 目录”，因此本任务不得新建 `tests/`；验证使用 focused one-off offline smoke checks、临时脚本和现有公共 contract。

---

## 1. 参考设计原则

实现时参考但不要照搬：

### 1.1 LeRobot

参考：

- https://github.com/huggingface/lerobot
- `src/lerobot/datasets/dataset_writer.py`
- `src/lerobot/datasets/feature_utils.py`
- `src/lerobot/datasets/dataset_metadata.py`

应吸收的原则：

- Dataset schema 主要描述 feature、shape、dtype、fps/episode identity。
- Writer/reader 负责结构正确，不重放机器人 runtime 的全部状态机。
- 逻辑时间可由 frame index + fps 定义；数据集不必保存每一个硬件事件 timestamp。
- 旧 major format 用专门 converter 迁移，不让主 reader 永久背负所有历史布局。
- 版本只代表真正的数据格式/语义 break，不代表每次内部实现变化。

### 1.2 官方 Diffusion Policy

参考：

- https://github.com/real-stanford/diffusion_policy
- `diffusion_policy/common/replay_buffer.py`
- `diffusion_policy/common/sampler.py`

应吸收的原则：

- 训练 buffer 核心是同长度的 `data/*` temporal arrays。
- `meta/episode_ends` 定义 episode 边界。
- 一个 episode 的所有 arrays 必须共享同一时间长度。
- sampler 只依赖 episode boundary；因此本项目应保持 physical episode 与 Zarr episode 一一对应，避免人工切段造成 padding / split 语义污染。

---

## 2. 最终数据架构

本任务完成后保持两层正式数据：

```text
Robot runtime
    │
    ▼
Raw Episode v34
    │
    │ strict whole-episode admission
    │ deterministic research transforms
    ▼
Canonical Policy Zarr v15
    │
    ▼
dexmani_policy（不修改）
```

### 2.1 Raw Episode

职责：

> 保存无法从其他数据重新得到、且对机器人学习研究仍有直接价值的原始测量、最终动作目标、最小有效性事实，以及重建派生几何所需的静态标定/运动学信息。

Raw **不是** runtime trace database，也不是 training cache。

### 2.2 Policy Zarr

职责：

> 保存已经通过整段准入、可以直接用于训练的完整多模态 canonical tensors。

Zarr **不是**历史审计档案，也不保存 runtime timestamps / debug status / migration bookkeeping。

---

## 3. Raw v34：一次最终收敛

将：

```python
EPISODE_SCHEMA_VERSION = 33
```

升级为：

```python
EPISODE_SCHEMA_VERSION = 34
```

v34 之后应长期冻结。只有 persisted scientific semantics 真正 breaking 时才允许 v35。

### 3.1 Raw v34 必需逐帧 datasets

`data.h5` 顶层必需且仅研究必要的 canonical arrays：

| Dataset | dtype | tail shape | 语义 |
| --- | --- | --- | --- |
| `arm_qpos` | float64 | (7,) | 当前 control row 选取的 xArm joint position |
| `arm_qvel` | float64 | (7,) | 同一 arm sample 的 joint velocity |
| `arm_effort` | float64 | (7,) | 同一 arm sample 的 SDK/native effort/torque telemetry，具体单位由 v34 contract 定义 |
| `hand_qpos` | float64 | (12,) | 当前 control row 选取的 XHand joint position |
| `hand_current` | float64 | (12,) | 同一 hand sample 的 current telemetry |
| `hand_contact` | float32 | (5, 3) | canonical XHand aggregate tactile payload |
| `hand_tactile_force` | float32 | (5, 120, 3) | canonical XHand dense tactile payload |
| `action_arm_joint_target` | float64 | (7,) | 本 control step 最终成功提交/发布边界所对应的 absolute arm joint target |
| `action_hand_joint_target` | float64 | (12,) | 同一步最终 hand joint target |
| `frame_valid` | bool | () | 本行是否形成正常 control solution 且对应 joint target 成功进入 command publication boundary；该 row 已由 runtime 的 observation 构造边界保证所需 sensor freshness |

媒体继续保持：

```text
rgb.mp4
depth.h5/depth   uint16 [T,H,W]
```

### 3.2 v34 tactile invalid 表达

删除：

```text
hand_contact_valid
hand_tactile_force_valid
```

v34 固定以下 invariant：

```text
valid tactile payload    <=> corresponding array is fully finite
invalid tactile payload  => corresponding array contains NaN
```

Native writer 必须保证这一 invariant。

Legacy migration 必须先验证旧 validity flag 与 finite/non-finite payload 完全一致；不一致时不得静默修复，必须拒绝该 episode 的 canonical migration并报告。

注意：

- `frame_valid` **不替代 tactile payload validity**。
- 一个 control row 可以 `frame_valid=True` 但 tactile payload invalid；由于 Zarr 是模态全集，后续 whole-episode admission 仍会因 non-finite tactile 拒绝整段。

### 3.3 `frame_valid` 的稳定语义

`frame_valid=True` 只表示核心 control-step demonstration 语义成立：

- 正常 control solution，而不是 IK / retarget / safety fallback failure；
- 该 row 已经成功通过 runtime 的 observation 构造边界；不要在 recorder 内再次实现一套 freshness/causality 判定；
- 最终 canonical joint target 成功进入定义明确的 command publication/commit boundary；
- 该行的 core state/action payload 可解释。

对当前 native teleop 路径，优先让 producer 直接从已有事实构造，例如：

```text
frame_valid =
    (control_status == FRAME_OK)
    AND (command is not None)
    AND (publication succeeded)
```

因为 `ObservationRow` 本身只有在 required arm/hand/camera/VR freshness 检查通过后才存在。不要复制第二套 observation validity state machine。

它**不表示**：

- 机器人已经物理执行/ACK/收敛；
- tactile 一定有效；
- 所有辅助 telemetry 一定 finite；
- replay 一定安全。

Native runtime 仍可保留内部 timestamp/freshness/safety mechanisms，但它们不再被持久化为 Raw schema。

### 3.4 Raw v34 必需 metadata

保留真正影响数据解释和派生研究模态重建的 metadata。

最小 episode identity：

```text
schema_version = 34
task_label
collection_source        # teleop / policy_rollout 等明确来源
control_hz
num_frames
episode_valid            # session-level validity latch
```

`episode_valid` 用来承接 **不能由单个 row 表达的整段生命周期失败**，例如 abnormal stop、operator pause 后仍保存、recording/runtime failure。它替代当前 `technical_status + had_pause` 作为训练准入所需的最小 episode-level 事实；不要仅删除旧字段而丢失这一信息。

RGB-D payload 的最小解释信息：

```text
camera_payload_mode = "depth_to_color_aligned_rgbd"
camera_color_width
camera_color_height
camera_color_intrinsics
camera_color_distortion_model
camera_color_distortion_coeffs
camera_T_xarm_base_from_color
depth_scale
```

重建 fingertip geometry 时，**随真实物理 setup 变化且无法从图像/关节数据恢复的 hand mount calibration** 必须随 Raw 保存：

```text
T_eef_handbase position + orientation
```

Raw 不保存 software provenance。以下内容均不进入 Raw episode metadata：

```text
URDF/model identity or hash
producer git SHA
fingertip link configuration
processing code/version identifiers
```

这些属于当前代码/配置，而不是实验观测事实。不要因为“可复现”再建立额外 software provenance 字段或 hash 体系。

Raw 只保存无法从实验之外恢复、且直接影响数值解释的真实物理标定，例如 camera calibration 和 `T_eef_handbase`。

### 3.5 Raw v34 不再要求/持久化的逐帧字段

从 canonical Raw schema 删除：

```text
timestamp

observation_timestamp_ns
action_timestamp_ns
arm_timestamp_ns
hand_timestamp_ns
camera_timestamp_ns
vr_timestamp_ns

arm_eef_intent

flag_frame_status

camera_depth_frame_number
camera_color_frame_number

vr_wrist_pos
vr_wrist_rot6d
vr_landmarks
head_quat_wxyz
```

同时 legacy 中以下 runtime diagnostics 不迁入 v34：

```text
arm_connected
hand_connected
hand_qpos_stale
tracking_error
flag_action_queued
observation_valid
flag_camera_fresh
camera_health
*_source_monotonic_ns
observation_anchor_monotonic_ns
action_arm_ee
```

这些信息仍永久存在于 immutable legacy source，用于历史审计；v34 不重复复制。

### 3.6 不要删除 runtime correctness mechanisms

本任务删除的是 **持久化 schema / offline validation 依赖**，不是运行时安全与同步机制。

不得因为 Raw 不再保存 timestamp 就删除：

- runtime sensor freshness；
- causal observation selection；
- command publication ordering；
- safety/rejection logic；
- worker health / shutdown correctness；
- internal profiling timestamps，只要 runtime 仍需要。

如果 current runtime 存在“暂停后仍继续写入同一个 episode，且没有任何 `frame_valid=False` 行”的路径，必须在删除 `had_pause` / timestamp admission 前修正该 lifecycle：

- 要么 pause 终止/拒绝当前 episode；
- 要么确保该 episode 后续不会被视为完整 valid demonstration。

不能仅删除检查后默认这种 episode 合格。

### 3.7 Raw reader / schema validator

`EpisodeReader` 只支持 v34。

主 reader 不支持 legacy v30。

Reader 只做结构验证：

- 目录和 `data.h5/depth.h5/rgb.mp4` 存在；
- schema version 正确；
- 10 个 required datasets 存在；
- row count / shape / dtype 正确；
- depth count/shape/dtype 正确；
- RGB 文件存在且非空；Reader 不为了打开低维 Raw 而扫描/解码整段视频；
- `control_hz > 0`；
- `episode_valid` 为 bool；
- camera intrinsics / depth scale 有效；
- base-from-camera 和 `T_eef_handbase` 为有限 rigid transform。

RGB 的实际 frame count / shape / dtype 在 whole-episode export 的 streaming pass 中一次性验证；不要让普通 Reader open 重复做媒体内容扫描。

Reader **不做**：

- training eligibility；
- `technical_status` gate；
- pause reconstruction；
- timestamp monotonicity；
- action-vs-observation timestamp ordering；
- source timestamp presence；
- frame validity判定；
- tactile finite判定；
- FK / point-cloud derivation。

### 3.8 schema extensibility

不要再用“任何未知 runtime/debug dataset 都意味着新 schema version”的模式。

实现一个简单边界：

- required canonical datasets：严格检查；
- 若项目确有 future research sensor，可加入明确的 known optional specs 而不 bump v34；
- 不允许拼写错误的未知 canonical top-level dataset 静默通过；
- debug/profiling 不应继续写进 canonical `data.h5`，优先放 runtime log。

---

## 4. 删除非研究必要的 Raw admission / bookkeeping

### 4.1 `technical_status` + `had_pause` → `episode_valid`

不再让 Reader / exporter 分别理解两套 lifecycle metadata。

将当前：

```text
technical_status
had_pause
```

收敛成一个稳定 bool：

```text
episode_valid
```

规则：

- episode START 时初始化为 true；
- abnormal stop / hardware/runtime/recording failure → false；
- operator pause 发生后如果该 physical episode 仍被保存 → false；
- normal uninterrupted operator save → true；
- clean discard 不发布 episode；
- termination reason 可以作为 optional diagnostic metadata 保留，但不作为 schema/admission 状态机。

Zarr 准入要求：

```python
episode_valid and np.all(frame_valid)
```

因此可以删除 `reader.require_valid(...)` 和 exporter 对 `had_pause` 的独立分支，但不能丢失 session-level invalid latch。

### 4.2 provenance workflow classifier

删除复杂的 workflow classifier 对训练导出的依赖。

Raw 只保留一个简单、明确的：

```text
collection_source
```

当前 Zarr v15 exporter 只接受已证明满足当前：

```text
teleop_published_joint_target
```

语义的 teleop source。

---

## 5. Policy Zarr：保持 v15 与模态全集

### 5.1 不升级 Zarr schema

保持：

```python
POLICY_ZARR_SCHEMA_NAME = "dexmani-real-policy-zarr"
POLICY_ZARR_SCHEMA_VERSION = 15
```

Raw v33→v34 不构成 Zarr tensor contract change。

### 5.2 canonical 完整数组全集保持不变

继续输出：

```text
data/
    joint_state
    arm_qvel
    arm_effort
    hand_current

    action
    action_ee

    contact_force
    tactile_force

    fingertip_points
    eef_pose

    rgb
    depth
    point_cloud

meta/
    episode_ends
```

不要按模型输入裁剪。

### 5.3 每个 Zarr episode 必须等于一个完整 physical Raw episode

`episode_ends` 继续只表示完整 physical demonstrations。

禁止：

- bad-row deletion；
- segment split；
- artificial episode boundary；
- re-stitching；
- partial episode export。

### 5.4 Zarr root attrs：只留 tensor interpretation / 当前 consumer 必需信息

在**不修改 `dexmani_policy`**的前提下，保留 current consumer contract 需要、以及完整多模态 tensor 自解释真正需要的 attrs。

至少保留：

```text
schema_name
schema_version
domain
task_name
dt

obs_alignment
observation_alignment
state_alignment
action_semantics

joint_names

action_ee_frame
action_ee_components

rgb_channels

point_cloud_frame
point_cloud_features
pointcloud_config_json

eef_pose_frame
eef_pose_components

fingertip_points_frame
fingertip_points_unit
fingertip_config_json

finger_names
tactile_sensor_ids
tactile_axis_names
tactile_point_indices

contact_force_unit
contact_force_frame
contact_force_representation   # 若当前 producer 已定义，保留其研究语义

tactile_force_representation
tactile_force_finger_order
tactile_force_sensor_order
tactile_force_point_order
tactile_force_axis_labels
tactile_force_unit

depth_scale_m_per_unit
depth_invalid_value
arm_effort_unit
hand_current_unit
```

重要：

- `pointcloud_config_json` 继续保存当前完整 point-cloud processing config；当前 `dexmani_policy` 会把它捕获进 strict resume semantics，不能为了最小化而只留下 `num_points/remove_table`。
- `fingertip_config_json` 可缩减到当前 consumer 真正需要的 ordered `fingertip_link_names`，若没有其它 consumer 依赖。
- 在删任何 attr 前，全仓搜索 `dexmani_policy` 当前 consumer；无法证明 unused 的不要删除。

### 5.5 从 Zarr attrs 删除纯 descriptive / duplicate provenance

若当前 `dexmani_policy` 不消费，删除以下类型 metadata：

```text
camera_intrinsic
camera_extrinsic
camera_geometry
camera_intrinsic_semantics
camera_extrinsic_semantics

joint_order                    # joint_names 已经定义 ordering

point_cloud_policy_id
point_cloud_transform
point_cloud_sampling
point_cloud_color_source
point_cloud_table_plane_abcd_json

fingertip_points_derivation
fingertip_points_policy_id

eef_pose_derivation
eef_pose_algorithm_id

contact_force_source
contact_force_si_verified

tactile_force_si_verified
tactile_force_spatial_geometry_verified
```

原则：

> Raw 保存每个 episode 的真实 calibration / model provenance；Zarr 保存最终 canonical tensor semantics。不要把 processing implementation prose 或 calibration 重复塞进 Zarr root attrs。

### 5.6 不要因为删 descriptive attrs 而放宽真正的 static tensor semantics

一个 Zarr 仍必须拥有统一：

- task；
- dt；
- tensor shape/dtype；
- joint/action ordering；
- point-cloud frame / features / processing config；
- tactile representation/order/unit；
- FK output frame/representation。

保持现有 `StaticSemanticsMismatch` 或等价的轻量机制。

---

## 6. Zarr whole-episode admission：从复杂 runtime proof 简化为研究必要检查

重构 `dexmani_real/dataset/processing.py::validate_episode()`。

最终只需要以下六类 gate。

### 6.1 基础 identity

要求：

- `num_frames > 0`；
- task name 有效；
- `collection_source` 为当前 exporter 支持的 teleop source。

### 6.2 Episode lifecycle 与全部 control rows clean

先要求：

```python
bool(meta["episode_valid"])
```

再要求：

```python
np.all(frame_valid)
```

任一条件失败：

```text
reject whole episode
```

### 6.3 完整多模态 source payload finite

因为 Zarr 是通用模态全集，以下会进入 canonical Zarr 的浮点 Raw payload 必须全 episode finite：

```text
arm_qpos
arm_qvel
arm_effort
hand_qpos
hand_current
hand_contact
hand_tactile_force
action_arm_joint_target
action_hand_joint_target
```

任何 non-finite：

```text
reject whole episode
```

### 6.4 RGB-D 完整

验证：

- RGB count == T；
- depth count == T；
- RGB 为预期 `uint8 [H,W,3]`；
- depth 为 `uint16 [H,W]`；
- geometry metadata 与 payload shape 一致。

### 6.5 静态 calibration / kinematics 可解释

必须能用 Raw 中 pin 的真实研究 metadata 构造：

- camera model；
- `T_xarm_base_from_color`；
- depth scale；
- arm FK；
- XHand FK；
- Raw 中记录的 hand mount calibration；
- exporter 当前显式提供的 processing configuration。

`T_eef_handbase` 必须从该 Raw episode 的物理标定读取，不要从“当前默认 runtime”静默补历史值。

URDF、model hash、git SHA、fingertip link 配置都不写入 Raw，也不建立额外 provenance 记录。若 exporter 计算 fingertip geometry 需要 link names，它们只作为运行时配置使用；最终 Zarr 仅保留当前 `dexmani_policy` contract 硬要求的最小 tensor 解释字段。

### 6.6 全部 derived modalities 成功

对完整 episode 实际生成：

- `action_ee`；
- `eef_pose`；
- `fingertip_points`；
- `point_cloud`；
- RGB/depth canonical arrays。

边界层只验证：

- expected shape；
- expected dtype；
- finite（适用时）；
- canonical rot6d validity（适用时）。

不要在 exporter 再重复一整套 point-cloud 内部 workspace / voxel / outlier 算法 invariant；这些应由 pure transform 自己保证。

任何一帧 derivation 失败：

```text
reject whole episode
```

### 6.7 删除的旧 admission checks

删除对以下 persisted runtime evidence 的训练准入依赖：

```text
technical_status
had_pause
provenance workflow classifier
technical_status / had_pause（由 episode_valid 取代）

flag_frame_status enum
hand_contact_valid
hand_tactile_force_valid

observation_timestamp_ns
action_timestamp_ns
timestamp strict monotonicity
timestamp gap <= 2dt
action_timestamp >= observation_timestamp

arm_timestamp_ns
hand_timestamp_ns
camera_timestamp_ns
vr_timestamp_ns
```

这些不是 canonical fixed-step imitation-learning dataset 必需的长期持久化语义。

---

## 7. `pick_place_toy`：legacy v30 → canonical Raw v34

### 7.1 不直接 v30 → Zarr

最终流程必须是：

```text
immutable legacy v30
        │
        ▼
one-time audited migration
        │
        ▼
canonical Raw v34
        │
        ▼
normal current Raw→Zarr v15 exporter
```

这样迁移后的历史数据与新采 v34 数据走同一条训练导出路径。

### 7.2 Legacy source 永久只读

迁移前记录每个 episode：

```text
SHA256(data.h5)
SHA256(depth.h5)
SHA256(rgb.mp4)
row count
```

迁移后再次确认 source hash 不变。

输出必须写新目录：

- staging；
- validate；
- atomic publish；
- refuse overwrite。

不得写回 `episodes/pick_place_toy`。

### 7.3 三个必须先 fact-check 的语义 gate

#### Gate A — action lineage

必须证明本批实际 lineage 下：

```text
action_arm_joint_sent
action_hand_joint
```

就是该 row 对应的最终、成功提交的 joint targets。

历史参考 `4bba54d` 的 `recording/frame.py` 明确说明：

```text
Joint targets are those actually submitted,
while action_arm_ee retains Cartesian intent.
```

但历史参考 commit 不是本批数据真实 provenance 的自动证明。

必须追：

- 原始录制版本；
- 后续 converter；
- 是否改过 action 数值、row alignment、sampling。

无法证明则不得把数据声明成 v34 canonical action semantics。

#### Gate B — tactile numeric lineage

必须彻底查清 2026-09-09 前后的 XHand 0.1 scale / bias 关系。

v34 只允许一个 canonical tactile numeric representation：

```text
current XHand SDK-native, bias-corrected representation
with fixed finger/sensor/axis ordering and unknown-SI unit
```

如果历史数据能通过**严格证明的确定性公式**转换，则迁移时规范化并逐元素验证。

如果无法证明，不要给历史数值贴 current tactile semantic label；对应 episode 不得迁入 canonical v34。

#### Gate C — arm effort / geometry / kinematics

确认：

- old `arm_tau` 与 current `arm_effort` 的来源和数值语义；
- historical camera geometry/depth scale；
- 当前 exporter 所使用的 arm / hand FK 实现能够正确解释这批 joint state；
- hand mount；
- fingertip geometry 所需的当前配置可用。

这里的 URDF / link 配置只用于执行 deterministic FK，不作为 Raw 或 migration report 的持久化 provenance。

只要能证明 deterministic equivalence，可以规范化；不能证明则 fail closed。

### 7.4 v30 → v34 字段映射

在上述 gate 通过后：

| legacy v30 | raw v34 |
| --- | --- |
| `arm_qpos` | `arm_qpos` |
| `arm_qvel` | `arm_qvel` |
| `arm_tau` | `arm_effort`，仅在语义证明后 |
| `hand_qpos` | `hand_qpos` |
| `hand_current` | `hand_current` |
| `hand_contact` | `hand_contact`，按已证明 canonical tactile transform |
| `hand_tactile_force` | `hand_tactile_force`，按已证明 canonical tactile transform |
| `action_arm_joint_sent` | `action_arm_joint_target` |
| `action_hand_joint` | `action_hand_joint_target` |
| historical status/provenance flags | 仅用于构造 `frame_valid`，不继续持久化 |
| RGB | byte-for-byte copy，能直接 copy 时禁止 decode/re-encode |
| depth | byte/numeric-identical copy |
| historical camera meta | canonicalize 为 v34 最小 aligned RGB-D camera meta |

### 7.5 legacy `frame_valid` 的保守构造

先解析本批实际历史 enum / runtime semantics，再按真实 legacy meaning 构造。

若 `4bba54d` 最终被证明与本批 lineage 等价，则至少保守要求：

```python
frame_valid = (
    old_frame_status == OLD_FRAME_OK
    and flag_action_queued
    and observation_valid
    and flag_camera_fresh
    and arm_connected
    and hand_connected
    and not hand_qpos_stale
)
```

必要时再并入已经证明属于 core control validity 的历史 flag。

不要把 tactile validity 并入 `frame_valid`；tactile invalid 由 NaN payload 表达，并在 full-modality Zarr admission 阶段整段拒绝。

### 7.6 legacy `episode_valid` 的构造

Legacy v30 没有当前 `technical_status/had_pause`，不得伪造这些历史字段。v34 migrator 应直接判断“该历史 episode 是否有足够证据满足 v34 session-level validity contract”：

- 使用真实 legacy stop/success/truncation metadata；
- 使用历史 control-grid/lifecycle 实现；
- 使用已存 row timeline / anchor continuity 作为证据之一；
- 不把“缺少旧字段”自动解释为 true。

证据足够时写 `episode_valid=True`；证据不足或存在 abnormal lifecycle 证据时写 false，并在 migration report 给出原因。

`episode_valid` 是 **v34 canonical audit result**，不是声称历史文件当时已经拥有同名字段。

### 7.7 historical status 绝不能 numeric-copy

历史参考：

```text
old FRAME_IK_FAIL = 2
old FRAME_RETARGET_FAIL = 4
```

当前 v33：

```text
FRAME_IK_FAIL = 1
FRAME_RETARGET_FAIL = 2
```

迁移只判断 historical semantic `OLD_FRAME_OK` / failure，不把旧数字复制成新 enum。

v34 不再持久化 detailed status enum。

### 7.8 不迁入 v34 的 legacy 字段

以下只留在 immutable v30 archive：

```text
timestamp
action_arm_ee

arm_connected
hand_connected
hand_qpos_stale
tracking_error

flag_action_queued
flag_frame_status

observation_anchor_monotonic_ns
observation_valid
arm_source_monotonic_ns
hand_source_monotonic_ns
vr_source_monotonic_ns
camera_source_monotonic_ns

flag_camera_fresh
camera_health
camera_depth_frame_number
camera_color_frame_number

vr_wrist_pos
vr_wrist_rot6d
vr_landmarks
head_quat_wxyz
```

### 7.9 legacy tactile validity flags

不迁入 v34，但转换前必须检查：

```text
hand_contact_valid == all-finite(hand_contact)
hand_tactile_force_valid == all-finite(hand_tactile_force)
```

语义上应理解为：

```text
valid=True  -> payload fully finite
valid=False -> payload invalid/non-finite
```

若真实历史 writer 允许更复杂情况，则按源码事实修正 invariant；不要凭想当然覆盖。

---

## 8. `pick_place_toy` 当前已知异常与预期 whole-episode 结果

当前审计已发现 10 个 episode 存在控制 / observation / camera / tactile 异常：

```text
episode_20260827_172305
episode_20260827_175240
episode_20260827_194525
episode_20260827_195951
episode_20260827_220112
episode_20260827_220747
episode_20260827_220919
episode_20260827_223607
episode_20260827_223729
episode_20260827_224527
```

当前保守分组：

```text
Raw v34 migration target:
    61 episodes
    14,309 rows
    只要全局 action/tactile/geometry semantics 可证明，bad episode 仍完整保存

Zarr v15 current clean candidates:
    51 episodes
    11,710 frames

Quarantine / currently excluded from Zarr:
    10 episodes
    2,599 frames
```

注意：

- 上述 51/11,710 只是当前审计基线，不要把数字写死进代码。
- 正式迁移后重新 dry-run。
- 7 个仅有旧 observation/camera false 的 episode，只有在历史 fact-check 明确证明旧 flag 过严/错误且整段仍满足 v34 `frame_valid` 语义时，才允许整段恢复。
- 不允许删除那一两行后保留其它行。
- `episode_20260827_195951` 已知有 tactile NaN 行；在 full-modality Zarr contract 下应整段拒绝，除非发现审计事实本身有误。

---

## 9. Migration tool

遵循仓库已有 one-off converter 风格，优先放在：

```text
examples/migrate_raw_v30_to_v34.py
```

或与已有 `examples/convert_raw_v29.py` 一致的命名/位置；不要把 legacy adapter放进 normal reader。

CLI 至少支持：

```text
SOURCE_ROOT
--output OUTPUT_ROOT
--dry-run
--report REPORT_JSON
```

### 9.1 行为

Dry-run：

- 不创建 canonical raw；
- 读取所有 source episode；
- 计算 hashes；
- 检查 lineage-dependent invariants；
- 输出逐 episode migration eligibility；
- 输出 `frame_valid=False` rows；
- 输出 tactile validity / finite mismatches；
- 输出 source→v34 field mapping；
- 输出 Zarr eligibility 的初始判断。

Actual migration：

- 拒绝 output 已存在；
- output 不得位于 source 内；
- 每 episode staging；
- 写完后用 v34 `EpisodeReader` 重开验证；
- 检查 row count；
- 检查直接映射数组；
- 检查 deterministic normalized arrays；
- 检查 RGB/depth；
- 成功后 atomic rename；
- 任一失败只删除自己 staging，不动 source。

### 9.2 Migration report

保持简单、可审计。

顶层：

```text
source_root
output_root
source_schema
target_schema
lineage evidence summary
action_mapping
tactile_normalization
camera_normalization
physical calibration decisions
```

每 episode：

```text
episode
source_hashes
source_rows
frame_valid_false_rows
nonfinite_rows
migration_result
migration_reason
zarr_eligible
zarr_rejection_reason
output_hashes
```

不要建立新的复杂 manifest schema/framework。

---

## 10. `dexmani_real` 代码修改范围

Codex 实施前先搜索实际引用，再按依赖拓扑改，不要盲删。

### 10.1 必改

```text
dexmani_real/recording/storage/schema.py
dexmani_real/recording/storage/reader.py
dexmani_real/recording/frame.py
dexmani_real/dataset/processing.py
dexmani_real/dataset/contracts.py
dexmani_real/dataset/export.py
dexmani_real/dataset/pointcloud.py
```

以及所有直接构造 `EpisodeFrame` / Raw meta / current source row 的 recorder/controller code。

### 10.2 Runtime observation / command code

允许修改 serialization plumbing，但：

- 不降低 sensor freshness；
- 不改变 causal sample selection；
- 不改变 robot command publication semantics；
- 不改变 hardware safety；
- 不为了删 Raw timestamp 删除 runtime 内部 timestamp。

### 10.3 Offline checks / examples / docs

更新所有依赖 v33 field names 的 examples / smoke-check surfaces；**不要新建 committed `tests/` 目录**。

新增 legacy migrator。

实现完成后同步更新：

- `AGENTS.md`：只更新与本次 Raw/Zarr data contract 冲突的 Research data standing instructions；
- README：稳定工作流与当前 schema；
- `PICK_PLACE_TOY_MIGRATION.md`：
  - 明确 v33 migration 不再是目标；
  - canonical path 改为 v30→v34→Zarr v15；
  - 保留历史审计事实；
  - 不把未实际运行的 migration 写成已完成。

### 10.4 不得修改

```text
dexmani_policy/*
```

如果发现当前 Zarr 简化会破坏 `dexmani_policy` contract：

- 不修改 policy；
- 回退对应 Zarr attr 删除；
- 记录该 attr 是 compatibility-required。

---

## 11. 实施顺序

严格按以下顺序，避免同时改变 schema、migration 和 exporter 后无法定位问题。

### Phase A — Fact-check 与 dependency inventory

1. 记录当前 HEAD。
2. 搜索所有 v33 datasets / metadata 的读写引用。
3. 搜索 `dexmani_policy` 当前 contract 使用的 Zarr attrs，只读核对。
4. 追 `pick_place_toy` action lineage。
5. 追 tactile 0.1 scale / bias lineage。
6. 追 arm_tau、camera calibration、kinematics / hand mount。
7. 把无法证明的项目写成明确 blocker，不猜。

### Phase B — Raw v34

1. 修改 schema。
2. 修改 frame serialization。
3. 让 native recorder生成 `frame_valid`。
4. 简化 reader。
5. 保留 runtime safety/synchronization。
6. 运行 focused one-off v34 round-trip smoke checks。

### Phase C — Raw→Zarr v15

1. 简化 `validate_episode()`。
2. 删除 timestamp/status/validity duplicated gates。
3. 保持 whole-episode transactional export。
4. 保持完整 modalities。
5. 减少 Zarr descriptive attrs，但严格保持 current `dexmani_policy` compatibility。
6. 运行 focused one-off exporter smoke checks。

### Phase D — legacy migrator

1. 实现 v30 reader/adaptor，仅存在 migrator 内。
2. dry-run。
3. small fixture migration。
4. actual local `pick_place_toy` migration（如果数据可用；输出不得提交大型二进制到 Git）。
5. v34 reader revalidation。
6. Zarr dry-run。
7. 记录完整结果。

### Phase E — Final verification

1. 完成所有 required offline smoke checks。
2. `pick_place_toy` source hashes unchanged。
3. Zarr contract smoke test。
4. 文档更新。
5. 搜索确认 normal runtime 无 legacy version branches。

---

## 12. 离线验证要求

遵循 `AGENTS.md`：不新增 committed `tests/` 目录，不恢复庞大 smoke-suite。使用临时 Python 脚本、one-off assertions、migrator `--dry-run` 和现有公共 contract 完成以下 focused offline checks。

### 12.1 Raw schema smoke checks

必须覆盖：

- valid v34 accepted；
- missing each required dataset rejected；
- wrong row count/shape/dtype rejected；
- timestamps / VR / frame number 不再 required；
- tactile NaN raw episode structurally可读；
- depth count mismatch 在 Reader 结构校验阶段 rejected；
- RGB file 缺失/空文件在 Reader 阶段 rejected；
- RGB 实际 frame count / shape / dtype mismatch 在 whole-episode exporter streaming pass 中 rejected；
- invalid camera calibration rejected；
- unknown/typo canonical datasets 不被静默接受。

### 12.2 `frame_valid / episode_valid` smoke checks

覆盖 native runtime source conditions：

- normal uninterrupted episode + normal row → episode_valid=true, frame_valid=true；
- IK/retarget/control failure → false；
- action publication failure → false；
- observation invalid → false；
- camera invalid/freshness failure → false；
- tactile invalid alone不应偷偷改变 core `frame_valid`，但 full Zarr export 会拒绝；
- abnormal stop → episode_valid=false；
- pause 后仍保存 → episode_valid=false。

### 12.3 Whole-episode Zarr smoke checks

至少：

1. 一个 episode_valid=true 且所有 frame_valid=true 的 clean episode → 全部 rows 写入。
2. episode_valid=false → 整段 0 rows。
3. 中间一行 `frame_valid=False` → 整段 0 rows。
4. 中间一行 tactile NaN → 整段 0 rows。
5. point-cloud derivation 一帧失败 → 整段 0 rows。
6. 两个 clean episodes → `episode_ends` 精确累计。
7. 第一段失败、第二段成功 → 不残留 partial first-episode arrays。
8. Zarr array key set仍为完整 canonical modality全集。
9. Zarr schema name/version仍为 v15。
10. 对当前 `dexmani_policy` consumer-required attrs 做 compatibility check。

### 12.4 Legacy migration smoke checks

构造最小 v30 fixture，覆盖：

- correct mapping；
- historical status semantic mapping；
- action queued false；
- observation/camera false；
- tactile invalid + NaN；
- tactile validity flag / payload不一致 → fail；
- source unchanged；
- output staging cleanup；
- refuse overwrite；
- no fake timestamp fields；
- no legacy diagnostic fields in v34。

### 12.5 `pick_place_toy` local acceptance

如果本机有真实数据：

```text
source:
61 episodes
14,309 rows
```

Raw migration完成后要求：

```text
target raw:
61 episodes
14,309 rows
```

且：

- source SHA256 不变；
- RGB 不重编码；
- depth 不重采样；
- row order 完全相同；
- direct-map arrays一致；
- normalized tactile/effort 逐元素满足被证明的公式；
- no interpolation / smoothing / clipping / resampling。

Zarr：

- included episode 要么完整出现，要么完全不存在；
- `episode_ends` 只对应完整 source episode；
- 当前已知异常 episode 默认整段 reject；
- 最终 count 以实际 dry-run 为准。

---

## 13. Zarr 数值验收

对于每个 included episode：

```text
joint_state =
concat(raw.arm_qpos, raw.hand_qpos)

action =
concat(raw.action_arm_joint_target,
       raw.action_hand_joint_target)

action_ee =
concat(FK(raw.action_arm_joint_target),
       raw.action_hand_joint_target)

eef_pose =
FK(raw.arm_qpos)

fingertip_points =
FK(raw.arm_qpos,
   raw.hand_qpos,
   pinned hand model/mount)

point_cloud =
derive(raw RGB-D,
       raw camera calibration,
       explicit ProcessingConfig)
```

所有 canonical arrays：

- 时间长度相同；
- shape/dtype符合 v15 contract；
- 需要 finite 的必须 finite；
- rot6d 合法；
- point cloud frame/features 正确。

---

## 14. Point-cloud / table-plane 原则

不得调用当前 runtime config 后静默把今天的 calibration 当历史事实。

`ProcessingConfig` 必须来自显式、可审计的参数。

对于历史 `pick_place_toy`：

- camera geometry 使用 raw/legacy 真实 metadata；
- hand mount 使用 Raw v34 中该 episode 自己的物理标定；URDF / fingertip links 仅作为 exporter 当前配置使用，不写入 Raw 或 migration provenance；
- table plane 无法证明时，优先显式 `remove_table=False`，而不是套当前 table plane；
- 一旦选择 processing config，同一个 Zarr 的 point-cloud tensor semantics必须统一；
- Zarr 不需要重复保存每 episode camera calibration / table plane descriptive provenance。

---

## 15. Version policy

完成后在代码注释/文档中明确：

### Raw

```text
v34 = stable research raw contract
```

只有下面情况才 bump：

- action semantic break；
- joint/tactile core tensor representation break；
- row alignment model break；
- required core array shape/order break。

以下不 bump：

- runtime internal timestamp；
- debug telemetry；
- controller refactor；
- new logging；
- point-cloud algorithm；
- Zarr processing；
- optional non-breaking research metadata。

### Zarr

保持 v15。

只有 consumer-visible tensor contract 变化才讨论 v16：

- tensor shape/dtype；
- action representation；
- ordering；
- frame/unit semantics；
- observation/action temporal alignment。

Raw v34 本身不是 v16 理由。

---

## 16. Code quality / implementation constraints

- 优先删除代码和分支，不新增抽象框架来“管理简化”。
- Legacy logic 只存在 one-off migrator。
- 不建立 generic migration registry。
- 不建立 schema adapter hierarchy。
- 不建立 generalized per-modality validity framework。
- 不新增 background service / database / catalog。
- 不为了少量 attrs 建独立 metadata subsystem。
- pure numerical transforms继续保持 pure。
- source mutation、staging publication、atomic rename 的边界必须清楚。
- 错误必须 fail closed，错误信息包含 episode name + 首个具体原因。
- 避免读取同一 RGB/video 多次；export 一次 pass 完成 validation + transform。
- 控制工作内存，继续 chunk/block 写 Zarr。
- 第一段 episode 失败不能留下错误 static attrs / staging arrays。

---

## 17. Definition of Done

只有以下全部满足才算完成：

- [ ] Raw v34 只有精简后的 canonical research arrays；v33 runtime trace fields 不再持久化。
- [ ] Native recorder 的 `frame_valid` 与 episode-level `episode_valid` 语义均通过 focused offline checks。
- [ ] Runtime freshness / causal / safety 没有因 storage 简化而降低。
- [ ] `EpisodeReader` 只负责 v34 结构验证，不承担 training admission。
- [ ] 正常 reader 没有 legacy v30 分支。
- [ ] Policy Zarr 保持 schema v15。
- [ ] Policy Zarr 保持完整多模态全集。
- [ ] Zarr exporter 对坏 Raw episode 整段拒绝，无 partial row/segment salvage。
- [ ] Zarr descriptive attrs 被精简，但当前 `dexmani_policy` contract 无需任何修改即可通过。
- [ ] v30→v34 migrator 是独立离线工具。
- [ ] legacy source 永不修改、不覆盖。
- [ ] 无 synthetic observation/action timestamps。
- [ ] action mapping 已通过真实历史 lineage 证明。
- [ ] tactile 0.1 scale / bias lineage 已证明或明确阻塞 canonical migration。
- [ ] historical camera / kinematics / hand mount 已 pin。
- [ ] migration report 可追踪每个 episode 的 source hash、frame_valid 异常和 Zarr 决定。
- [ ] 若真实 `pick_place_toy` 数据可用，Raw migration 保持 61 episodes / 14,309 rows。
- [ ] current known bad episodes 不通过删行/切段进入 Zarr。
- [ ] 当前 `dexmani_policy` 原样可以读取最终 Zarr。
- [ ] 更新 `AGENTS.md` / README / `PICK_PLACE_TOY_MIGRATION.md`，standing instructions 与 v34 实现一致，且文档只陈述实际执行过的结果。

---

## 18. 最终设计意图

实现完成后的长期心智模型必须非常简单：

```text
runtime:
    负责 causal / freshness / safety / publication correctness

Raw v34:
    保存不可再生研究事实
    + frame_valid（row-level）
    + episode_valid（session-level）
    + camera / hand-mount 等真实物理 calibration
    不保存 runtime trace

Zarr v15:
    只保存完整 clean physical episodes
    保存通用多模态 canonical tensors
    不保存 raw diagnostics / migration provenance

legacy:
    immutable
    只通过 one-off offline migrator 进入 v34
```

不要再围绕“如何让 v30 伪装成 v33”设计系统；本任务的目标是用一个稳定、简洁、可复现的 v34 Raw contract 结束 schema churn，并让历史真机数据通过可审计的 semantic compaction 继续服务当前完整多模态训练链路。

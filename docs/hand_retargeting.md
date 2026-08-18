# Hand retarget 当前合同

本文只描述当前实现的稳定边界。源码和运行配置是最终事实来源；静态阅读不需要连接硬件。

## 一条调用链

```text
Quest 右手 21×3 landmarks
  → Unity→FLU / VR ring
  → causal snapshot（控制网格）
  → geometry gate
  → TAG 或 DexPilot
  → startup ramp + operational clip
  → SafetyGate
  → latest-wins hand command ring
  → hand worker
  → XHand SDK
```

相关实现：

| 责任 | 文件 |
|---|---|
| VR payload 与关节顺序 | `utils/schema.py` |
| 因果读取 | `shm/causal_reader.py`、`teleop/snapshot.py` |
| 控制网格与候选 | `teleop/loop.py`、`teleop/hand_control.py` |
| retarget wrapper | `teleop/hand_retarget.py` |
| TAG | `teleop/tag_retargeting/` |
| 安全发布 | `policy/safety.py` |
| worker/SDK | `robot/hand_process.py`、`robot/xhand.py` |

## 当前语义

- 只消费右手 21 个关键点；wrist 姿态供 arm mapping 使用，不参与手指 retarget。
- 默认后端是 TAG；`hand_retargeting_type=dexpilot` 选择 DexPilot 外部后端。
- 控制网格默认 16 Hz；hand worker 频率和所有数值默认值以 `config/defaults.py` 为准。
- 同一个已验证 VR ring sequence 最多调用一次有状态 retargeter；后续控制 tick 复用缓存结果。
- 缓存成功和失败结果，避免同一观测被重复求解；ramp 仍按控制 tick 推进。
- wrapper 没有额外 output EMA。TAG 使用优化器时间正则/捏合权重，DexPilot 保留外部库自身 LPFilter。
- operational bounds 是发布前整形；结构、finite、机械范围或生命周期错误必须拒绝/hold，不能靠 clip 掩盖。
- 应用层不对 hand command 做 command-to-command delta clamp；worker 和 SDK 仍独立复验范围、generation、有效期和状态。

## 状态与时间

用下面的词区分不同阶段：

```text
observed → solved → shaped → published → accepted → measured
```

- `observed`：因果控制网格选中的 VR frame。
- `solved`：retargeter 返回的 SDK-order qpos；TAG 可提供 `last_raw_qpos`。
- `shaped`：startup ramp 和 operational clip 后的 endpoint。
- `published`：通过 SafetyGate 并写入 hand ring 的 endpoint。
- `accepted`：XHand send 成功后 driver 记录的 endpoint。
- `measured`：设备反馈 qpos；不等于以上任何一个候选。

`run_generation` 负责跨暂停、BEGIN、回零和反馈故障隔离；valid-until 负责同一 generation 内的过期命令。worker 在 SDK 边界前都要复验。ordinary pause 是 command quiescence，不发布测量位置作为替代 hold；旧 ring 命令必须被丢弃。

状态 owner：teleop 管理 retarget/cache/ramp；hand worker 管理 ring cursor 和设备反馈；XHand driver 只在 SDK send 成功后更新 `last_qpos_cmd`；main/lifecycle 管理安全状态、generation 边界和 shutdown。

## 输入和几何 gate

进入 solver 前至少检查：

- shape 为 `(21, 3)`；
- 所有值 finite；
- 掌部基线和骨段长度不退化；
- VR frame 新鲜、sequence 可验证并符合当前 generation。

掌心坐标系、MANO 轴变换和 pinky compensation 都在 `teleop/hand_retarget.py` 及后端调用链中完成。输入数组不得原地修改。schema 中的 `side` 字段不代表当前已经支持左右手对称控制。

## 两个后端

| 维度 | TAG | DexPilot |
|---|---|---|
| 目标 | 五指 wrist-to-tip | 外部库的 reference-vector graph |
| 状态 | 仓库内 Pinocchio/NLopt | 外部 `dex-retargeting` |
| 捏合 | 独立第二阶段，失败可回退 Stage 1 | 外部 projected-vector 机制 |
| raw 语义 | solver 的 SDK-order 输出 | 外部 retarget 返回值 |
| 可审计性 | 目标、梯度和回退在仓库内 | 依赖已安装库版本 |

两个后端共用 XHand SDK 关节顺序和机械 URDF，但不能假定 raw、滤波或失败状态的含义完全相同。更换后端时同步检查配置、warm start、reset、记录解释和离线评估。

## 失败和耦合发布

retarget 失败通常返回上一条合法 hand endpoint，并令 `retarget_ok=False`；这和“endpoint 非法”不同。非法 shape/finite、mechanical envelope、IPC 或 lifecycle 错误则不跨越命令边界。

| arm IK | hand retarget | 结果 |
|---|---|---|
| 成功 | 成功 | 发布 arm 与 shaped hand |
| 成功 | 失败 | arm 可继续，hand 使用合法 hold |
| 失败 | 成功 | arm hold；允许独立 hand-only（若调用路径允许） |
| 任意 | 非法/IPC 错误 | 不发布不一致的 coupled action |

teleop 会先得到本周期 shaped hand endpoint，再把它用于 arm IK/collision；因此规划看到的是最终候选，而不是 solver raw。在线路径和 replay dense preflight 的碰撞覆盖范围不同，不要把二者混为同一保证。

## 记录与修改检查

v17 中与 hand 相关的字段大致分为：VR observation、`action_hand_joint_raw`、最终 `action_hand_joint`、`hand_qpos`、retarget/frame flags 和 timing。held/failure 路径的 raw 字段可能是兼容 fallback；解释 raw 前使用 `EpisodeReader` 提供的 validity mask。

修改以下任一项时，至少检查：

1. `utils/schema.py` 的 shape、关节顺序和 invalid value。
2. cache key、reset/reanchor、generation 和 ramp 推进时机。
3. TAG/DexPilot 的 raw 输出和失败语义。
4. `policy/safety.py`、hand worker、XHand driver 的重复校验。
5. `episode_samples.py`、EpisodeReader、processing/replay 对字段和 flags 的解释。

离线调参入口是 `examples/tune_hand_retarget.py`；不要以 teleop 硬件运行代替 solver 的 deterministic check。任何实时预算、默认参数或外部依赖版本变化，都应回到配置和代码核对后再更新本文。
